import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_data_stream(total_points=1000, start_date="2024-01-01"):
    # Convert start_date to datetime
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    
    # Generate time values as a sequence of dates (one per day)
    time_values = [start_date + timedelta(days=i) for i in range(total_points)]
    
    # Generate patterns
    regular_pattern = 10 * np.sin(2 * np.pi * 0.05 * np.arange(total_points))
    seasonal_pattern = 5 * np.sin(2 * np.pi * 0.005 * np.arange(total_points))
    noise = np.random.normal(0, 1, total_points)
    signal = regular_pattern + seasonal_pattern + noise
    
    # Introduce various types of anomalies
    anomalies = np.zeros(total_points)
    
    # 1. Spike anomalies
    spike_indexes = np.random.choice(total_points, size=int(0.02 * total_points), replace=False)
    anomalies[spike_indexes] += np.random.uniform(15, 25, size=len(spike_indexes))
    
    # 2. Dip anomalies
    dip_indexes = np.random.choice(total_points, size=int(0.02 * total_points), replace=False)
    anomalies[dip_indexes] -= np.random.uniform(15, 25, size=len(dip_indexes))
    
    # 3. Level shift anomalies
    shift_start = np.random.randint(total_points // 4, 3 * total_points // 4)
    shift_duration = np.random.randint(20, 50)
    anomalies[shift_start:shift_start+shift_duration] += 15
    
    # 4. Trend anomalies
    trend_start = np.random.randint(total_points // 4, 3 * total_points // 4)
    trend_duration = np.random.randint(30, 70)
    trend = np.linspace(0, 20, trend_duration)
    anomalies[trend_start:trend_start+trend_duration] += trend
    
    # 5. Variance change anomalies
    variance_start = np.random.randint(total_points // 4, 3 * total_points // 4)
    variance_duration = np.random.randint(40, 80)
    anomalies[variance_start:variance_start+variance_duration] += np.random.normal(0, 5, variance_duration)
    
    signal_with_anomalies = signal + anomalies
    
    return time_values, signal_with_anomalies

def save_data_to_csv(time_values, data_stream, filename='data_stream.csv'):
    df = pd.DataFrame({
        'Time': pd.to_datetime(time_values),
        'Value': data_stream
    })
    df.set_index('Time', inplace=True)
    df.to_csv(filename)
    print(f"Data stream saved to {filename}")

def main():
    total_points = 1000
    time_values, data_stream = generate_data_stream(total_points=total_points, start_date="2024-01-01")
    save_data_to_csv(time_values, data_stream, filename='initial_data_stream.csv')

if __name__ == "__main__":
    main()