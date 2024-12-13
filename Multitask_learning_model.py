import torch
import torch.nn as nn
import torch.optim as optim
import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import numpy as np
import psutil
import time
import os
import torch.nn.functional as F


def measure_performance(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        process = psutil.Process(os.getpid())
        start_memory = process.memory_info().rss
        cpu_start = process.cpu_percent(interval=None)

        result = func(*args, **kwargs)

        end_time = time.time()
        end_memory = process.memory_info().rss
        cpu_end = process.cpu_percent(interval=None)

        print(f"Execution Time: {end_time - start_time:.2f}s")
        print(f"Memory Usage: {end_memory - start_memory} bytes")
        print(f"CPU Usage: {cpu_end - cpu_start:.2f}%")

        return result
    return wrapper


# Define a custom dataset for loading data from SQLite
class RetailDataset(Dataset):
    def __init__(self, db_file, transform=None, sample_frac=0.0001):
        print("Connecting to the database...")
        conn = sqlite3.connect(db_file)
        print("Database connection successful.")
        
        # Load the data
        print("Loading data from the database...")
        self.data = pd.read_sql_query("""
            SELECT c.*, p.*
            FROM transactions t
            JOIN customers c ON t.customer_id = c.customer_id
            JOIN products p ON t.product_id = p.product_id
        """, conn)
        print(f"Data loaded: {len(self.data)} rows.")
        
        conn.close()

        # Use only a sample of the data
        print(f"Sampling {sample_frac * 100}% of the data...")
        self.data = self.data.sample(frac=sample_frac).reset_index(drop=True)
        print(f"Sampled data: {len(self.data)} rows.")
        
        self.transform = transform

        # Define input and output columns based on your data structure
        self.input_columns = [
            'age', 'number_of_children', 'gender_male', 'gender_other',
            'income_bracket_low', 'income_bracket_medium', 'marital_status_married', 
            'marital_status_single', 'education_level_high_school', 'education_level_master', 
            'education_level_phd', 'occupation_retired', 'occupation_self_employed', 
            'occupation_unemployed'
        ]
        
        self.output_columns = [
            'product_rating', 'product_review_count', 'product_weight', 'unit_price',
            'product_brand_y', 'product_brand_z', 'product_size_medium', 
            'product_size_small', 'product_color_blue', 'product_color_green', 
            'product_color_red', 'product_color_white', 'product_material_metal', 
            'product_material_plastic', 'product_material_wood', 'product_category_electronics', 
            'product_category_furniture', 'product_category_groceries', 'product_category_toys'
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Ensure we properly access the input and output columns
        inputs = self.data.loc[idx, self.input_columns].values.astype(float)
        outputs = self.data.loc[idx, self.output_columns].values.astype(float)

        # Return the data as tensors
        return torch.tensor(inputs, dtype=torch.float32), torch.tensor(outputs, dtype=torch.float32)

class MultiTaskModel(nn.Module):
    def __init__(self, input_size):
        super(MultiTaskModel, self).__init__()

        # Shared layers
        self.shared_layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # Task-specific heads
        # Regression heads (Task Group 1)
        self.regression_heads = nn.ModuleDict({
            'rating': nn.Linear(32, 1),
            'review_count': nn.Linear(32, 1),
            'weight': nn.Linear(32, 1),
            'price': nn.Linear(32, 1)
        })

        # Classification heads (Task Group 2)
        self.classification_heads = nn.ModuleDict({
            'brand': nn.Linear(32, 2),        # y, z (fallback x)
            'size': nn.Linear(32, 2),         # medium, small (fallback large)
            'color': nn.Linear(32, 4),        # blue, green, red, white (fallback black)
            'material': nn.Linear(32, 3),     # metal, plastic, wood (fallback glass)
            'category': nn.Linear(32, 4)      # electronics, furniture, groceries, toys (fallback clothing)
        })

    def forward(self, x):
        # Pass input through shared layers
        shared_representation = self.shared_layers(x)

        # Compute regression tasks
        regression_outputs = {
            task: head(shared_representation)
            for task, head in self.regression_heads.items()
        }

        # Compute classification tasks with softmax for probabilistic outputs
        classification_outputs = {
            task: F.softmax(head(shared_representation), dim=-1)
            for task, head in self.classification_heads.items()
        }

        # Combine all outputs into a single dictionary
        return {**regression_outputs, **classification_outputs}

# Training setup
@measure_performance
def train_model(model, dataloader, epochs=10, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn_regression = nn.MSELoss()
    loss_fn_classification = nn.CrossEntropyLoss()

    # Task-specific mappings
    regression_tasks = ['rating', 'review_count', 'weight', 'price']
    classification_tasks = {
        'brand': (4, 6),
        'size': (6, 8),
        'color': (8, 12),
        'material': (12, 15),
        'category': (15, 19)
    }

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)

            # Compute combined loss
            loss = 0

            # Add regression losses
            for idx, task in enumerate(regression_tasks):
                loss += loss_fn_regression(outputs[task], targets[:, idx].unsqueeze(1))

            # Add classification losses
            for task, (start, end) in classification_tasks.items():
                loss += loss_fn_classification(outputs[task], targets[:, start:end].argmax(dim=1))

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(dataloader):.4f}")


def evaluate_model(model, dataloader):
    model.eval()
    loss_fn_regression = nn.MSELoss(reduction='sum')

    regression_tasks = ['rating', 'review_count', 'weight', 'price']
    classification_tasks = {
        'brand': (4, 6),
        'size': (6, 8),
        'color': (8, 12),
        'material': (12, 15),
        'category': (15, 19)
    }

    total_samples = len(dataloader.dataset)
    total_regression_loss = 0
    total_mse = {task: 0 for task in regression_tasks}
    correct_classifications = {task: 0 for task in classification_tasks}

    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs)

            # Compute regression losses
            for idx, task in enumerate(regression_tasks):
                mse_loss = loss_fn_regression(outputs[task], targets[:, idx].unsqueeze(1))
                total_regression_loss += mse_loss.item()
                total_mse[task] += mse_loss.item()

            # Compute classification accuracies
            for task, (start, end) in classification_tasks.items():
                _, predicted = torch.max(outputs[task], dim=1)
                correct_classifications[task] += (predicted == targets[:, start:end].argmax(dim=1)).sum().item()

    # Calculate average regression metrics
    print(f"Avg. Regression Loss: {total_regression_loss / total_samples:.4f}")
    for task in regression_tasks:
        mse = total_mse[task] / total_samples
        rmse = np.sqrt(mse)
        print(f"{task} - MSE: {mse:.4f}, RMSE: {rmse:.4f}")

    # Calculate classification accuracies
    for task, correct in correct_classifications.items():
        accuracy = correct / total_samples
        print(f"Accuracy for {task}: {accuracy:.2%}")

if __name__ == "__main__":
    # Load the data
    print("Loading dataset...")
    dataset = RetailDataset(db_file='retail_sample_100000.db', sample_frac=1.0)

    # Determine input size from the dataset
    input_size = dataset[0][0].shape[0]  # Assuming dataset[0][0] contains input features
    print(f"Input size: {input_size}")

    print(f"Total dataset size (before splitting): {len(dataset)}")
    print("Sample of the dataset:\n", dataset.data.head())

    # Dataset splitting
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # DataLoader setup
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Instantiate the model
    print("Instantiating the model...")
    model = MultiTaskModel(input_size=input_size).to(device)

    # Train the model
    print("Training the model...")
    train_model(model, train_dataloader, epochs=10, lr=0.001)

    # Save the trained model
    print("Saving the trained model...")
    torch.save(model.state_dict(), "multi_task_model.pth")

    # Evaluate the model
    print("Evaluating the model...")
    evaluate_model(model, test_dataloader)
