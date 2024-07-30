def preprocess_data(df):
    # Handle missing values
    df = df.dropna()

    # Additional preprocessing steps (if needed)
    return df

# Example function to merge additional product data
def merge_product_data(df, product_data_path):
    product_df = pd.read_csv(product_data_path)
    df = df.merge(product_df, on='product_id', how='left')
    return df
