import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
from sklearn.model_selection import train_test_split
from src.evaluation import get_rmse, precision_at_k, recall_at_k

# Load the processed data
df = pd.read_csv('data/processed_amazon_reviews.csv')

# Split the data into train and test sets
train_data, test_data = train_test_split(df, test_size=0.2)

# Create train user-item matrix
train_user_item_matrix = train_data.pivot(index='user_id', columns='product_id', values='rating').fillna(0)

# Perform SVD
U, sigma, Vt = svds(train_user_item_matrix, k=50)
sigma = np.diag(sigma)

# Reconstruct the matrix
predicted_ratings = np.dot(np.dot(U, sigma), Vt)
predicted_ratings_df = pd.DataFrame(predicted_ratings, columns=train_user_item_matrix.columns)

# Evaluate the model on the training set
train_rmse = get_rmse(predicted_ratings_df.values, train_user_item_matrix.values)
print(f'Train RMSE: {train_rmse}')

# Create test user-item matrix
test_user_item_matrix = test_data.pivot(index='user_id', columns='product_id', values='rating').fillna(0)
test_user_item_matrix = test_user_item_matrix.loc[train_user_item_matrix.index, train_user_item_matrix.columns].fillna(0)

# Evaluate the model on the test set
U_test, sigma_test, Vt_test = svds(test_user_item_matrix, k=50)
sigma_test = np.diag(sigma_test)
predicted_test_ratings = np.dot(np.dot(U_test, sigma_test), Vt_test)
predicted_test_ratings_df = pd.DataFrame(predicted_test_ratings, columns=train_user_item_matrix.columns)

test_rmse = get_rmse(predicted_test_ratings_df.values, test_user_item_matrix.values)
print(f'Test RMSE: {test_rmse}')

# Evaluate Precision and Recall at k
k = 5
user_id = df['user_id'].unique()[0]  # Example user for evaluation
recommended_items = predicted_test_ratings_df.loc[user_id].sort_values(ascending=False).index.tolist()
relevant_items = test_user_item_matrix.loc[user_id][test_user_item_matrix.loc[user_id] > 0].index.tolist()

precision = precision_at_k(recommended_items, relevant_items, k)
recall = recall_at_k(recommended_items, relevant_items, k)

print(f'Precision at {k}: {precision}')
print(f'Recall at {k}: {recall}')
