# E-Commerce Recommendation System

## Overview

This project implements a recommendation engine for an e-commerce platform, designed to provide personalized product recommendations based on user behavior and purchase history. The recommendation engine uses a collaborative filtering approach with Singular Value Decomposition (SVD) to predict user preferences and suggest products they are likely to be interested in.

## Features

- **Personalized Recommendations**: Generates product recommendations tailored to individual user preferences.
- **Scalability**: Optimized to handle high traffic with caching and a production-ready server.
- **Continuous Evaluation**: Regularly evaluates and retrains the model to ensure accuracy and relevance.

## Project Structure

```
ecommender-system/
├── data/
│   ├── amazon_reviews.csv
│   ├── processed_amazon_reviews.csv
│   └── additional_product_data.csv (if any)
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── evaluation.py
│   ├── continuous_evaluation.py
├── app/
│   ├── __init__.py
│   ├── recommender.py
│   ├── api.py
│   └── optimize.py
├── data_processing.py
├── train_model.py
├── evaluate_model.py
├── requirements.txt
├── README.md
└── run.py
```

## Setup

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Installation

1. **Clone the Repository**

   Clone the project repository to your local machine:

   ```bash
   git clone https://github.com/your-username/recommender-system.git
   cd recommender-system
   ```

2. **Install Required Packages**

   Install the necessary Python packages using pip:

   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the Data**

   - Download the dataset `ratings_eletronics (1).csv` from `https://www.kaggle.com/datasets/saurav9786/amazon-product-reviews` and place it in the `data` directory.
   - If additional product data is available, place it in the `data` directory as well.

4. **Process the Data**

   Run the data processing script to clean and preprocess the data:

   ```bash
   python data_processing.py
   ```

   This will create the `processed_amazon_reviews.csv` file used for training the model.

5. **Train the Model**

   Execute the model training script to build and train the recommendation model:

   ```bash
   python train_model.py
   ```

6. **Evaluate the Model**

   Run the evaluation script to assess the performance of the trained model:

   ```bash
   python evaluate_model.py
   ```

## Running the Application

### Development Server

To run the Flask application in development mode, use the following command:

```bash
python run.py
```

### Production Server

For a production-ready environment, use Gunicorn to serve the Flask application:

1. **Install Gunicorn**

   ```bash
   pip install gunicorn
   ```

2. **Run Gunicorn**

   ```bash
   gunicorn --workers 4 run:app
   ```

   This command runs the application with 4 worker processes, which helps handle high traffic and improve performance.

## API Endpoints

### `/recommend` (POST)

Provides personalized product recommendations for a given user.

- **Request Body**:

  ```json
  {
    "user_id": "USER_ID"
  }
  ```

- **Response**:

  ```json
  [
    "PRODUCT_ID_1",
    "PRODUCT_ID_2",
    ...
  ]
  ```

  The response contains a list of recommended product IDs for the specified user.

## Continuous Evaluation and Retraining

The model is periodically retrained to incorporate new data and maintain accuracy. This is handled by the APScheduler library:

- The retraining process is scheduled to run daily at midnight.
- The script `continuous_evaluation.py` manages this periodic retraining.

## Testing the API

To test the API endpoint, you can use `curl` or a tool like Postman. Here’s an example using `curl`:

```bash
curl -X POST -H "Content-Type: application/json" -d '{"user_id": "A2SUAM1J3GNN3B"}' http://127.0.0.1:5000/recommend
```

This command sends a POST request to the `/recommend` endpoint with a user ID and retrieves recommendations.

## Additional Information

- **Collaborative Filtering**: The recommendation system uses collaborative filtering with Singular Value Decomposition (SVD) to predict user preferences based on past interactions.
- **Evaluation Metrics**: The model’s performance is evaluated using RMSE, Precision, and Recall.

## Contributing

Contributions are welcome! If you have suggestions or improvements, please submit issues or pull requests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions or support, please contact [arturcf00@gmail.com](mailto:arturcf00@gmail.com).
