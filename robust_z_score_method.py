"""
This script performs anomaly detection on time-series data using robust z-scores. 
It reads data from a CSV file, calculates robust z-scores based on the median and median absolute deviation, 
and flags anomalies that deviate significantly. The results are saved to a CSV file.
"""

import pandas as pd
import numpy as np
from scipy import stats

def read_data(file_path):
    
    df = pd.read_csv(file_path, parse_dates=['Time'], index_col='Time')
    return df

def robust_zscore(data):
    """
    Calculates the robust z-scores for a given dataset, using the median and median absolute deviation (MAD).

    Args:
        data (np.ndarray or pd.Series): The time-series data (or any numerical data).

    Returns:
        np.ndarray: An array of robust z-scores.
    """
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    modified_zscore = 0.6745 * (data - median) / mad
    return modified_zscore

def detect_anomalies(df, threshold=2.75):
    """
    Detects anomalies in the time-series data based on the calculated robust z-scores.

    Args:
        df (pd.DataFrame): The input time-series data.
        threshold (float): The z-score threshold above which a point is considered an anomaly.

    Returns:
        pd.DataFrame: A DataFrame containing the original data along with z-scores and anomaly flags.
    """
    df['robust_zscore'] = robust_zscore(df['Value'])
    df['is_anomaly'] = np.abs(df['robust_zscore']) > threshold
    return df

def save_results(df, output_file='anomaly_results_z_score.csv'):
    
    df.to_csv(output_file)
    print(f"Results saved to '{output_file}'")

def main():
    
    input_file = 'initial_data_stream.csv'
    output_file = 'anomaly_results_z_score.csv'
    threshold = 2.75  # Adjust this value to change sensitivity

    df = read_data(input_file)

    result_df = detect_anomalies(df, threshold)

    print(result_df.describe())
    print(f"\nTotal anomalies detected: {result_df['is_anomaly'].sum()}")

    save_results(result_df, output_file)

if __name__ == "__main__":
    main()
