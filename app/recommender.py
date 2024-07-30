import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds

class Recommender:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = pd.read_csv(data_path)
        self.user_item_matrix = self.df.pivot(index='user_id', columns='product_id', values='rating').fillna(0)
        self.U, self.sigma, self.Vt = self._train_svd()

    def _train_svd(self, k=50):
        U, sigma, Vt = svds(self.user_item_matrix, k=k)
        sigma = np.diag(sigma)
        return U, sigma, Vt

    def recommend_products(self, user_id, num_recommendations=5):
        user_idx = self.user_item_matrix.index.get_loc(user_id)
        predicted_ratings = np.dot(np.dot(self.U, self.sigma), self.Vt)
        predicted_ratings_df = pd.DataFrame(predicted_ratings, columns=self.user_item_matrix.columns)
        user_ratings = predicted_ratings_df.iloc[user_idx].sort_values(ascending=False)
        return user_ratings.head(num_recommendations).index.tolist()
