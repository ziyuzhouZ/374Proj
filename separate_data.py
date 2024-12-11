import pandas as pd

# Load the original dataset
df = pd.read_csv('retail_data.csv')

# Define columns for Customer Info
customer_columns = [
    'customer_id',
    'age',
    'gender',
    'income_bracket',
    'marital_status',
    'number_of_children',
    'education_level',
    'occupation'
]

# Define columns for Product Info
product_columns = [
    'product_id',
    'product_brand',
    'product_rating',
    'product_review_count',
    'product_size',
    'product_weight',
    'product_color',
    'product_material',
    'product_category',
    'unit_price'
]

# Define columns for the Linking Table
linking_columns = [
    'transaction_id',
    'customer_id',
    'product_id'
]

# Extract Customer Info
customer_info = df[customer_columns].drop_duplicates()

# Extract Product Info
product_info = df[product_columns].drop_duplicates()

# Extract Linking Table
linking_table = df[linking_columns].drop_duplicates()

# Save each DataFrame to a separate CSV file
customer_info.to_csv('customer_info.csv', index=False)
product_info.to_csv('product_info.csv', index=False)
linking_table.to_csv('linking_table.csv', index=False)

print("Extraction complete. The new CSV files are 'customer_info.csv', 'product_info.csv', and 'linking_table.csv'.")
