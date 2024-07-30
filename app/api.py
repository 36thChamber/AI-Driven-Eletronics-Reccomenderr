from flask import request, jsonify
from app import app
from app.recommender import Recommender

# Initialize the recommender
recommender = Recommender(data_path='data/processed_amazon_reviews.csv')

@app.route('/recommend', methods=['POST'])
def recommend():
    user_id = request.json['user_id']
    recommendations = recommender.recommend_products(user_id)
    return jsonify(recommendations)
