import torch
import torch.nn as nn
import torch.optim as optim
import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from model_evaluation import evaluate_model
import numpy as np


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

# Define the multi-task model
class MultiTaskModel(nn.Module):
    def __init__(self):
        super(MultiTaskModel, self).__init__()
        # Shared layers
        self.shared_layers = nn.Sequential(
            nn.Linear(len(dataset.input_columns), 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # Task-specific heads
        self.rating_head = nn.Linear(32, 1)
        self.review_count_head = nn.Linear(32, 1)
        self.weight_head = nn.Linear(32, 1)
        self.price_head = nn.Linear(32, 1)

        # Classification heads (using single output per class)
        self.brand_head = nn.Linear(32, 2)  # y, z, with x as fallback
        self.size_head = nn.Linear(32, 2)   # medium, small, with large as fallback
        self.color_head = nn.Linear(32, 4)  # blue, green, red, white, with black as fallback
        self.material_head = nn.Linear(32, 3)  # metal, plastic, wood, with glass as fallback
        self.category_head = nn.Linear(32, 4)  # electronics, furniture, groceries, toys, with clothing as fallback

    def forward(self, x):
        shared_representation = self.shared_layers(x)

        # Regression tasks
        rating = self.rating_head(shared_representation)
        review_count = self.review_count_head(shared_representation)
        weight = self.weight_head(shared_representation)
        price = self.price_head(shared_representation)

        # Classification tasks with softmax for probabilistic outputs
        brand = self.brand_head(shared_representation)
        size = self.size_head(shared_representation)
        color = self.color_head(shared_representation)
        material = self.material_head(shared_representation)
        category = self.category_head(shared_representation)

        return {
            'rating': rating,
            'review_count': review_count,
            'weight': weight,
            'price': price,
            'brand': brand,
            'size': size,
            'color': color,
            'material': material,
            'category': category
        }

# Training setup
def train_model(model, dataloader, epochs=10, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn_regression = nn.MSELoss()
    loss_fn_classification = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)

            # Compute combined loss
            loss = (
                loss_fn_regression(outputs['rating'], targets[:, 0].unsqueeze(1)) +
                loss_fn_regression(outputs['review_count'], targets[:, 1].unsqueeze(1)) +
                loss_fn_regression(outputs['weight'], targets[:, 2].unsqueeze(1)) +
                loss_fn_regression(outputs['price'], targets[:, 3].unsqueeze(1)) +
                loss_fn_classification(outputs['brand'], targets[:, 4:6].argmax(dim=1)) +
                loss_fn_classification(outputs['size'], targets[:, 6:8].argmax(dim=1)) +
                loss_fn_classification(outputs['color'], targets[:, 8:12].argmax(dim=1)) +
                loss_fn_classification(outputs['material'], targets[:, 12:15].argmax(dim=1)) +
                loss_fn_classification(outputs['category'], targets[:, 15:19].argmax(dim=1))
            )
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(dataloader)}")

def evaluate_model(model, dataloader):
    model.eval()
    total_regression_loss = 0
    total_mse = {
        'rating': 0,
        'review_count': 0,
        'weight': 0,
        'price': 0
    }
    correct_classifications = {
        'brand': 0, 'size': 0, 'color': 0, 'material': 0, 'category': 0
    }
    total_samples = len(dataloader.dataset)
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs)

            # Compute regression losses and accumulate MSE for each task
            for idx, task in enumerate(['rating', 'review_count', 'weight', 'price']):
                mse_loss = nn.functional.mse_loss(outputs[task], targets[:, idx].unsqueeze(1), reduction='sum')
                total_regression_loss += mse_loss.item()
                total_mse[task] += mse_loss.item()

            # Compute classification accuracies
            for task, index_range in [('brand', (4, 6)), ('size', (6, 8)), 
                                      ('color', (8, 12)), ('material', (12, 15)), 
                                      ('category', (15, 19))]:
                _, predicted = torch.max(outputs[task], 1)
                correct_classifications[task] += (predicted == targets[:, index_range[0]:index_range[1]].argmax(1)).sum().item()

    # Calculate MSE and RMSE for each regression task
    print(f"Avg. Regression Loss: {total_regression_loss / total_samples:.4f}")
    for task in total_mse:
        mse = total_mse[task] / total_samples
        rmse = np.sqrt(mse)
        print(f"{task} - MSE: {mse:.4f}, RMSE: {rmse:.4f}")
    
    # Print classification accuracies
    for task, correct in correct_classifications.items():
        print(f"Accuracy for {task}: {correct / total_samples:.2%}")

# Instantiate and train the model
# Instantiate and train the model
if __name__ == "__main__":
    # Load the data
    print("Loading dataset...")
    dataset = RetailDataset(db_file='retail_sample_100000.db', sample_frac=1.0)

    # Check if the dataset is loaded correctly
    print(f"Total dataset size (before splitting): {len(dataset)}")

    # Print the first few rows to verify data
    print("Sample of the dataset:\n", dataset.data.head())

    train_size = int(0.8 * len(dataset))  # Use 80% for training
    test_size = len(dataset) - train_size

    # Split the dataset
    print(f"Train size: {train_size}, Test size: {test_size}")
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # DataLoader setup
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Instantiate the model
    print("Instantiating the model...")
    model = MultiTaskModel()

    # Train the model
    print("Training the model...")
    train_model(model, train_dataloader, epochs=10, lr=0.001)

    # Evaluate the model
    print("Evaluating the model...")
    evaluate_model(model, test_dataloader)
