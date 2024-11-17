import pandas as pd
import sqlite3

def load_data_to_db(db_file):
    """Loads data from CSV files into a SQLite database.

    Args:
        db_file: The path to the SQLite database file.
    """

    # Read the CSV files into DataFrames
    customer_info = pd.read_csv('customer_info_cleaned.csv')
    product_info = pd.read_csv('product_info_cleaned.csv')
    transactions = pd.read_csv('linking_table_cleaned.csv')

    # Limit to 1000 samples
    customer_info = customer_info.head(100000)
    product_info = product_info.head(100000)
    transactions = transactions.head(100000)

    # Create a database connection
    conn = sqlite3.connect(db_file)

    # Create tables (if not exist)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS products (
            product_id INT PRIMARY KEY NOT NULL,
            product_rating DECIMAL(3,1),
            product_review_count INT,
            product_weight DECIMAL(5,2),
            unit_price DECIMAL(10,2),
            product_brand_y BOOLEAN,
            product_brand_z BOOLEAN,
            product_size_medium BOOLEAN,
            product_size_small BOOLEAN,
            product_color_blue BOOLEAN,
            product_color_green BOOLEAN,
            product_color_red BOOLEAN,
            product_color_white BOOLEAN,
            product_material_metal BOOLEAN,
            product_material_plastic BOOLEAN,
            product_material_wood BOOLEAN,
            product_category_electronics BOOLEAN,
            product_category_furniture BOOLEAN,
            product_category_groceries BOOLEAN,
            product_category_toys BOOLEAN
        );
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS customers (
            customer_id INT PRIMARY KEY NOT NULL,
            age INT,
            number_of_children INT,
            gender_male BOOLEAN,
            gender_other BOOLEAN,
            income_bracket_low BOOLEAN,
            income_bracket_medium BOOLEAN,
            marital_status_married BOOLEAN,
            marital_status_single BOOLEAN,
            education_level_high_school BOOLEAN,
            education_level_master BOOLEAN,
            education_level_phd BOOLEAN,
            occupation_retired BOOLEAN,
            occupation_self_employed BOOLEAN,
            occupation_unemployed BOOLEAN
        );
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS transactions (
            transaction_id INT PRIMARY KEY NOT NULL,
            customer_id INT,
            product_id INT,
            FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
            FOREIGN KEY (product_id) REFERENCES products(product_id)
        );
    ''')

    # Insert data into tables
    product_info.to_sql('products', conn, index=False, if_exists='replace')
    customer_info.to_sql('customers', conn, index=False, if_exists='replace')
    transactions.to_sql('transactions', conn, index=False, if_exists='replace')

    conn.commit()
    conn.close()

# Example usage:
db_file = 'retail_sample_100000.db'
load_data_to_db(db_file)
