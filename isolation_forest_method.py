import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

def read_data(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path, parse_dates=['Time'], index_col='Time')
    return df

def detect_anomalies(df, contamination=0.1):
    # Prepare the data
    scaler = StandardScaler()
    X = scaler.fit_transform(df[['Value']])
    
    # Create and fit the Isolation Forest
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    anomaly_labels = iso_forest.fit_predict(X)
    
    # Add results to the dataframe
    df['anomaly_score'] = iso_forest.decision_function(X)
    df['is_anomaly'] = anomaly_labels == -1
    
    return df

def save_results(df, output_file='anomaly_results_iforest.csv'):
    # Save all data points with anomaly information to a CSV file
    df.to_csv(output_file)
    print(f"Results saved to '{output_file}'")

def main():
    input_file = 'initial_data_stream.csv'
    output_file = 'anomaly_results_iforest.csv'
    contamination = 0.01# Adjust this value to change the proportion of anomalies

    # Read the data
    df = read_data(input_file)

    # Detect anomalies
    result_df = detect_anomalies(df, contamination)

    # Display summary of results
    print(result_df.describe())
    print(f"\nTotal anomalies detected: {result_df['is_anomaly'].sum()}")

    # Save results
    save_results(result_df, output_file)

if __name__ == "__main__":
    main()