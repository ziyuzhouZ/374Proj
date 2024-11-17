/*
Filename: clean_data.py

Functions:
    based on the cleaned data, create the db table for all the data
    product_id	product_rating	product_review_count	product_weight	unit_price	product_brand_brand_y	product_brand_brand_z	product_size_medium	product_size_small	product_color_blue	product_color_green	product_color_red	product_color_white	product_material_metal	product_material_plastic	product_material_wood	product_category_electronics	product_category_furniture	product_category_groceries	product_category_toys
    1480	2	560	4	49	1	0	0	1	0	0	1	0	1	0	0	1	0	0	0

    customer_id	age	number_of_children	gender_male	gender_other	income_bracket_low	income_bracket_medium	marital_status_married	marital_status_single	education_level_high_school	education_level_master's	education_level_phd	occupation_retired	occupation_self-employed	occupation_unemployed
    1	56	3	0	1	0	0	0	0	0	0	0	0	1	0

    transaction_id	customer_id	product_id
    503290	1	1480

Author name: Ziyu Zhou
Appreciation: ChatGPT, Gemini
*/
CREATE TABLE products (
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

CREATE TABLE transactions (
    transaction_id INT PRIMARY KEY  NOT NULL,
    customer_id INT,
    product_id INT,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);

CREATE TABLE customers (
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
