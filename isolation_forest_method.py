"""
This script detects anomalies in time-series data using the Isolation Forest algorithm. 
It reads the data from a CSV file, applies scaling, and fits the Isolation Forest model to identify 
anomalies based on an anomaly score. The results, including the anomaly score and detection labels, 
are saved to a CSV file.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

def read_data(file_path):
    
    df = pd.read_csv(file_path, parse_dates=['Time'], index_col='Time')
    return df

def detect_anomalies(df, contamination=0.1):
    """
    Detects anomalies in the time-series data using the Isolation Forest algorithm.
    
    Args:
        df (pd.DataFrame): The input time-series data.
        contamination (float): The proportion of anomalies in the data, used to adjust sensitivity.

    Returns:
        pd.DataFrame: A DataFrame containing the original data, anomaly scores, and anomaly labels.
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(df[['Value']])

    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    anomaly_labels = iso_forest.fit_predict(X)

    df['anomaly_score'] = iso_forest.decision_function(X)
    df['is_anomaly'] = anomaly_labels == -1

    return df

def save_results(df, output_file='anomaly_results_iforest.csv'):
    
    df.to_csv(output_file)
    print(f"Results saved to '{output_file}'")

def main():
    
    input_file = 'initial_data_stream.csv'
    output_file = 'anomaly_results_iforest.csv'
    contamination = 0.01  # Adjust this value to change the proportion of anomalies

    df = read_data(input_file)

    result_df = detect_anomalies(df, contamination)

    print(result_df.describe())
    print(f"\nTotal anomalies detected: {result_df['is_anomaly'].sum()}")

    save_results(result_df, output_file)

if __name__ == "__main__":
    main()
