import numpy as np
from sklearn.metrics import mean_squared_error, precision_score, recall_score

def get_rmse(predictions, truth):
    predictions = predictions[truth.nonzero()].flatten()
    truth = truth[truth.nonzero()].flatten()
    return np.sqrt(mean_squared_error(predictions, truth))

def precision_at_k(recommended_items, relevant_items, k):
    recommended_at_k = recommended_items[:k]
    relevant_set = set(relevant_items)
    recommended_set = set(recommended_at_k)
    relevant_recommended_set = recommended_set.intersection(relevant_set)
    return len(relevant_recommended_set) / k

def recall_at_k(recommended_items, relevant_items, k):
    recommended_at_k = recommended_items[:k]
    relevant_set = set(relevant_items)
    recommended_set = set(recommended_at_k)
    relevant_recommended_set = recommended_set.intersection(relevant_set)
    return len(relevant_recommended_set) / len(relevant_set)
