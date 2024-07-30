from src.data_loader import load_data
from src.data_preprocessing import preprocess_data

# Load data
file_path = 'data/ratings_Electronics (1)'
df = load_data(file_path)

# Preprocess data
df = preprocess_data(df)

# Save the processed data for further use
df.to_csv('data/processed_ratings_Electronics (1)', index=False)
