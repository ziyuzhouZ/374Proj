"""
Filename: clean_data.py

Functions:
    clean the original dataset 
    customer_info = pd.read_csv('customer_info.csv')
    product_info = pd.read_csv('product_info.csv')
    linking_table = pd.read_csv('linking_table.csv')
    and preprocess the data

Author name: Ziyu Zhou
Appreciation: ChatGPT, Gemini
"""

import pandas as pd

# Load the datasets
customer_info = pd.read_csv('customer_info.csv')
product_info = pd.read_csv('product_info.csv')
linking_table = pd.read_csv('linking_table.csv')

# Function to preprocess a DataFrame
def preprocess_dataframe(df, df_name):
    # Check for null/missing data
    if df.isnull().values.any():
        print(f"{df_name} has missing data.")
    else:
        print(f"{df_name} passes the null check.")

    # Check for whitespace
    whitespace_issues = df.apply(lambda x: x.str.contains('^\s*$', na=False).any() if x.dtype == "object" else False)
    if whitespace_issues.any():
        print(f"{df_name} has whitespace issues.")
    else:
        print(f"{df_name} passes the whitespace check.")

    # Check for duplicates
    if df.duplicated().any():
        print(f"{df_name} has duplicates.")
    else:
        print(f"{df_name} passes the duplicate check.")

    # Standardize format (e.g., strip whitespace, lower case for string columns)
    df = df.applymap(lambda x: x.strip().lower() if isinstance(x, str) else x)

    # Handling unexpected values in categorical columns
    if df_name == "customer_info":
        valid_genders = ['male', 'female', 'other']
        df['gender'] = df['gender'].where(df['gender'].isin(valid_genders), 'other')
        
        valid_income_brackets = ['low', 'medium', 'high']
        df['income_bracket'] = df['income_bracket'].where(df['income_bracket'].isin(valid_income_brackets), 'medium')

        valid_marital_statuses = ['single', 'married', 'divorced']
        df['marital_status'] = df['marital_status'].where(df['marital_status'].isin(valid_marital_statuses), 'single')
        
        valid_education_levels = ["bachelor's", 'phd', "master's", 'high school']
        df['education_level'] = df['education_level'].where(df['education_level'].isin(valid_education_levels), 'high school')

    elif df_name == "product_info":
        valid_sizes = ['small', 'medium', 'large']
        df['product_size'] = df['product_size'].where(df['product_size'].isin(valid_sizes), 'medium')

        valid_colors = ['red', 'blue', 'green', 'white', 'black']
        df['product_color'] = df['product_color'].where(df['product_color'].isin(valid_colors), 'unknown')

        valid_materials = ['metal', 'plastic', 'wood', 'glass']
        df['product_material'] = df['product_material'].where(df['product_material'].isin(valid_materials), 'unknown')

        valid_categories = ['electronics', 'groceries', 'toys', 'clothing', 'furniture']
        df['product_category'] = df['product_category'].where(df['product_category'].isin(valid_categories), 'other')

    # Convert categories into numeric values using One-Hot Encoding
    if df_name == "customer_info":
        df = pd.get_dummies(df, columns=['gender', 'income_bracket', 'marital_status', 'education_level', 'occupation'], drop_first=True)
        
    elif df_name == "product_info":
        df = pd.get_dummies(df, columns=['product_brand', 'product_size', 'product_color', 'product_material', 'product_category'], drop_first=True)

    # Standardize column names
    df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
    
    # Convert boolean values to 0 and 1
    df = df.astype(int)

    print(f"{df_name} preprocessing completed.")
    return df

# Preprocess each DataFrame
customer_info_cleaned = preprocess_dataframe(customer_info, "customer_info")
product_info_cleaned = preprocess_dataframe(product_info, "product_info")
linking_table_cleaned = preprocess_dataframe(linking_table, "linking_table")

# Save the cleaned DataFrames to new CSV files
customer_info_cleaned.to_csv('customer_info_cleaned.csv', index=False)
product_info_cleaned.to_csv('product_info_cleaned.csv', index=False)
linking_table_cleaned.to_csv('linking_table_cleaned.csv', index=False)

import pandas as pd

# Load each CSV file into a DataFrame
customer_info = pd.read_csv('customer_info_cleaned.csv')
product_info = pd.read_csv('product_info_cleaned.csv')
linking_table = pd.read_csv('linking_table_cleaned.csv')


print("All datasets have been cleaned and saved.")
