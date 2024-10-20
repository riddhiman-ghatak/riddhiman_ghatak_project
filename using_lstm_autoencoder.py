import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping

def process_data(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path, parse_dates=['Time'], index_col='Time')
    return df

def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length + 1):
        sequences.append(data[i:i+seq_length])
    return np.array(sequences)

def build_model(seq_length, n_features):
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
    # Normalize the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[['Value']])

    # Create sequences
    sequences = create_sequences(scaled_data, seq_length)

    # Build and train the model
    model = build_model(seq_length, 1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, mode='min')
    model.fit(sequences, sequences, epochs=50, batch_size=32, validation_split=0.1, callbacks=[early_stopping], shuffle=False)

    # Get reconstruction error
    reconstructions = model.predict(sequences)
    mse = np.mean(np.power(sequences - reconstructions, 2), axis=(1,2))

    # Define threshold for anomaly detection
    threshold = np.mean(mse) + threshold_factor * np.std(mse)

    # Identify anomalies
    anomalies = mse > threshold

    # Create a DataFrame with all data points and anomaly information
    result_df = pd.DataFrame({
        'Time': df.index[seq_length-1:],
        'Value': df['Value'][seq_length-1:],
        'reconstruction_error': mse,
        'is_anomaly': anomalies
    })

    return result_df

def save_results(result_df, output_file='anomaly_results_lstm.csv'):
    # Save all data points with anomaly information to a CSV file
    result_df.to_csv(output_file)
    print(f"Results saved to '{output_file}'")

def main():
    input_file = 'initial_data_stream.csv'
    output_file = 'anomaly_results_lstm.csv'
    seq_length = 10
    threshold_factor = 3

    # Process the data
    df = process_data(input_file)

    # Detect anomalies
    result_df = detect_anomalies(df, seq_length, threshold_factor)

    # Display results
    print(result_df)

    # Save results
    save_results(result_df, output_file)

if __name__ == "__main__":
    main()