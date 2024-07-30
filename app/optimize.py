# Caching
from flask_caching import Cache

cache = Cache(app, config={'CACHE_TYPE': 'simple'})

@app.route('/recommend', methods=['POST'])
@cache.cached(timeout=60, query_string=True)
def recommend():
    user_id = request.json['user_id']
    recommendations = recommender.recommend_products(user_id)
    return jsonify(recommendations)
