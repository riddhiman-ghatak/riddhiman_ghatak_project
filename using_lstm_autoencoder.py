"""
This script is designed to detect anomalies in time-series data using an LSTM-based autoencoder. 
It processes the data from a CSV file, trains a model to reconstruct the input sequences, and 
detects anomalies based on the reconstruction error. The results are saved to a CSV file for further analysis.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping

def process_data(file_path):
    
    df = pd.read_csv(file_path, parse_dates=['Time'], index_col='Time')
    return df

def create_sequences(data, seq_length):
    """
    Creates sequences of a specified length from the input data.

    Args:
        data (np.ndarray): The normalized time-series data.
        seq_length (int): The length of each sequence.

    Returns:
        np.ndarray: An array of sequences.
    """
    sequences = []
    for i in range(len(data) - seq_length + 1):
        sequences.append(data[i:i+seq_length])
    return np.array(sequences)

def build_model(seq_length, n_features):
    """
    Builds an LSTM autoencoder model for sequence reconstruction.

    Args:
        seq_length (int): The length of each input sequence.
        n_features (int): The number of features in each sequence.

    Returns:
        keras.models.Sequential: The compiled LSTM autoencoder model.
    """
    model = Sequential([
        LSTM(64, activation='relu', input_shape=(seq_length, n_features), return_sequences=True),
        LSTM(32, activation='relu', return_sequences=False),
        RepeatVector(seq_length),
        LSTM(32, activation='relu', return_sequences=True),
        LSTM(64, activation='relu', return_sequences=True),
        TimeDistributed(Dense(n_features))
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def detect_anomalies(df, seq_length=10, threshold_factor=3):
    """
    Detects anomalies in the time-series data based on reconstruction error from an LSTM autoencoder.

    Args:
        df (pd.DataFrame): The input time-series data.
        seq_length (int): The length of sequences for the model.
        threshold_factor (float): The multiplier for defining the anomaly threshold.

    Returns:
        pd.DataFrame: A DataFrame containing time, value, reconstruction error, and anomaly flags.
    """
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[['Value']])

    sequences = create_sequences(scaled_data, seq_length)

    model = build_model(seq_length, 1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, mode='min')
    model.fit(sequences, sequences, epochs=50, batch_size=32, validation_split=0.1, callbacks=[early_stopping], shuffle=False)

    reconstructions = model.predict(sequences)
    mse = np.mean(np.power(sequences - reconstructions, 2), axis=(1,2))

    threshold = np.mean(mse) + threshold_factor * np.std(mse)

    anomalies = mse > threshold

    result_df = pd.DataFrame({
        'Time': df.index[seq_length-1:],
        'Value': df['Value'][seq_length-1:],
        'reconstruction_error': mse,
        'is_anomaly': anomalies
    })

    return result_df

def save_results(result_df, output_file='anomaly_results_lstm.csv'):
    
    result_df.to_csv(output_file)
    print(f"Results saved to '{output_file}'")

def main():
    """
    Main function to load data, detect anomalies, and save the results.
    """
    input_file = 'initial_data_stream.csv'
    output_file = 'anomaly_results_lstm.csv'
    seq_length = 10
    threshold_factor = 3

    df = process_data(input_file)

    result_df = detect_anomalies(df, seq_length, threshold_factor)

    print(result_df)

    save_results(result_df, output_file)

if __name__ == "__main__":
    main()
