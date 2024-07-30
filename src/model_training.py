import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Load the processed data
df = pd.read_csv('../data/processed_amazon_reviews.csv')

# Create user-item matrix
user_item_matrix = df.pivot(index='user_id', columns='product_id', values='rating').fillna(0)

# Split the data into train and test sets
train_data, test_data = train_test_split(df, test_size=0.2)

# Create train user-item matrix
train_user_item_matrix = train_data.pivot(index='user_id', columns='product_id', values='rating').fillna(0)

# Perform SVD
U, sigma, Vt = svds(train_user_item_matrix, k=50)
sigma = np.diag(sigma)

# Reconstruct the matrix
predicted_ratings = np.dot(np.dot(U, sigma), Vt)
predicted_ratings_df = pd.DataFrame(predicted_ratings, columns=user_item_matrix.columns)

# Evaluate the model
def get_rmse(predictions, truth):
    predictions = predictions[truth.nonzero()].flatten()
    truth = truth[truth.nonzero()].flatten()
    return np.sqrt(mean_squared_error(predictions, truth))

train_rmse = get_rmse(predicted_ratings_df.values, train_user_item_matrix.values)
print(f'Train RMSE: {train_rmse}')

# Create test user-item matrix
test_user_item_matrix = test_data.pivot(index='user_id', columns='product_id', values='rating').fillna(0)
test_user_item_matrix = test_user_item_matrix.loc[user_item_matrix.index, user_item_matrix.columns].fillna(0)

# Reconstruct the matrix for test data
U_test, sigma_test, Vt_test = svds(test_user_item_matrix, k=50)
sigma_test = np.diag(sigma_test)
predicted_test_ratings = np.dot(np.dot(U_test, sigma_test), Vt_test)
predicted_test_ratings_df = pd.DataFrame(predicted_test_ratings, columns=user_item_matrix.columns)

# Evaluate the model on test data
test_rmse = get_rmse(predicted_test_ratings_df.values, test_user_item_matrix.values)
print(f'Test RMSE: {test_rmse}')
