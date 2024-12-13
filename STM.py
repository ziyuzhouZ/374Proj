import torch
import torch.nn as nn
import torch.optim as optim
import psutil
import os
import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time

# Define a custom dataset for loading data from SQLite
class RetailDataset(Dataset):
    def __init__(self, db_file, transform=None, sample_frac=0.0001):
        print("Connecting to the database...")
        conn = sqlite3.connect(db_file)
        print("Database connection successful.")

        # Load the data
        print("Loading data from the database...")
        self.data = pd.read_sql_query(
            """
            SELECT c.*, p.*
            FROM transactions t
            JOIN customers c ON t.customer_id = c.customer_id
            JOIN products p ON t.product_id = p.product_id
            """, conn
        )
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

# Define the single-task model
class SingleTaskModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(SingleTaskModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        )

    def forward(self, x):
        return self.network(x)

# Performance measurement decorator
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

# Training function
@measure_performance
def train_single_task_model(model, dataloader, task_idx, output_size, epochs=10, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss() if output_size == 1 else nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for inputs, targets in dataloader:
            optimizer.zero_grad()

            outputs = model(inputs)

            if output_size == 1:
                # For regression tasks
                loss = loss_fn(outputs, targets[:, task_idx].unsqueeze(1))
            else:
                # For classification tasks
                target_indices = targets[:, task_idx:task_idx + output_size].argmax(dim=1)
                loss = loss_fn(outputs, target_indices)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(dataloader)}")

if __name__ == "__main__":
    # Load the dataset
    dataset = RetailDataset(db_file='retail_sample_100000.db', sample_frac=1.0)
    input_size = len(dataset.input_columns)

    # Task-specific configurations
    tasks = [
        ("rating", 1), ("review_count", 1), ("weight", 1), ("price", 1),
        ("brand", 2), ("size", 2), ("color", 4), ("material", 3), ("category", 4)
    ]

    for idx, (task_name, output_size) in enumerate(tasks):
        print(f"Training Single Task Model for: {task_name}")
        model = SingleTaskModel(input_size, output_size)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        train_single_task_model(model, dataloader, idx, output_size)
