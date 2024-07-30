import pandas as pd

def load_data(file_path):
    df = pd.read_csv(file_path, header=None)
    df.columns = ['user_id', 'product_id', 'rating', 'timestamp']
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    return df
