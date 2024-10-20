import pandas as pd
import numpy as np
from scipy import stats

def read_data(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path, parse_dates=['Time'], index_col='Time')
    return df

def robust_zscore(data):
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    modified_zscore = 0.6745 * (data - median) / mad
    return modified_zscore

def detect_anomalies(df, threshold=3):
    # Calculate robust z-scores
    df['robust_zscore'] = robust_zscore(df['Value'])
    
    # Identify anomalies
    df['is_anomaly'] = np.abs(df['robust_zscore']) > threshold
    
    return df

def save_results(df, output_file='anomaly_results_z_score.csv'):
    # Save all data points with anomaly information to a CSV file
    df.to_csv(output_file)
    print(f"Results saved to '{output_file}'")

def main():
    input_file = 'initial_data_stream.csv'
    output_file = 'anomaly_results_z_score.csv'
    threshold = 3  # Adjust this value to change sensitivity

    # Read the data
    df = read_data(input_file)

    # Detect anomalies
    result_df = detect_anomalies(df, threshold)

    # Display summary of results
    print(result_df.describe())
    print(f"\nTotal anomalies detected: {result_df['is_anomaly'].sum()}")

    # Save results
    save_results(result_df, output_file)

if __name__ == "__main__":
    main()