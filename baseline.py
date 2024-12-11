from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import numpy as np
import sqlite3
import pandas as pd

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

def evaluate_baseline(X, y, task_type):
    if task_type == 'regression':
        model = DummyRegressor(strategy="mean")
    else:
        model = DummyClassifier(strategy="most_frequent")
    
    model.fit(X, y)
    score = model.score(X, y)
    print(f"Baseline {task_type} score: {score:.2f}")

if __name__ == "__main__":
    dataset = RetailDataset(db_file='retail_sample_100000.db', sample_frac=1.0)
    input_data = dataset.data[dataset.input_columns]
    output_data = dataset.data[dataset.output_columns]

    tasks = [
        ("rating", "regression"), ("review_count", "regression"),
        ("weight", "regression"), ("price", "regression"),
        ("brand", "classification"), ("size", "classification"),
        ("color", "classification"), ("material", "classification"),
        ("category", "classification")
    ]

    for idx, (task_name, task_type) in enumerate(tasks):
        X_train, X_test, y_train, y_test = train_test_split(
            input_data, output_data.iloc[:, idx], test_size=0.2, random_state=42
        )
        print(f"Evaluating baseline for task: {task_name}")
        evaluate_baseline(X_train, y_train, task_type)
