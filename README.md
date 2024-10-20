# Anomaly Detection in Time-Series Data Project

## Project Overview

This project aims to detect anomalies in time-series data using three different algorithms. It includes generating synthetic time-series data with various types of anomalies and applying distinct anomaly detection methods. Below are the key components and methods used in the project:

### 1. Data Generation

We create synthetic time-series data that mimics real-world scenarios and inject different types of anomalies such as:
- **Spikes**: Sudden upward movements in the data.
- **Dips**: Sharp downward movements in the data.
- **Level Shifts**: Long-term shifts in the data level.
- **Trends**: Gradual increase or decrease over a period.
- **Variance Changes**: Periods with increased volatility or variance.

These anomalies are intentionally added to test the effectiveness of each detection method.

### 2. Anomaly Detection Algorithms

We implement three anomaly detection algorithms to find and mark unusual patterns in the time-series data:

#### a. **Isolation Forest**
Isolation Forest is a machine learning-based algorithm that isolates anomalies by recursively partitioning the data. It works by randomly selecting a feature and splitting the data, making it easier to isolate anomalies. The algorithm scores each point based on how easily it can be separated from the rest. Points with the lowest scores are considered anomalies.

- **Advantages**: Efficient for large datasets, works well with high-dimensional data.
- **Results**: Anomaly scores and binary labels (anomaly or not) are saved in `anomaly_results_iforest.csv`.

#### b. **Robust Z-Score Method**
This method is a statistical approach to anomaly detection, which uses the **median absolute deviation (MAD)** to calculate robust Z-scores. Unlike traditional Z-scores, which rely on mean and standard deviation, the robust Z-score method uses the median and MAD, making it resistant to outliers in the data.

- **Advantages**: Simple and effective for normally distributed data, less sensitive to outliers compared to standard Z-score.
- **Results**: Robust Z-scores and binary anomaly labels are saved in `anomaly_results_z_score.csv`.

#### c. **LSTM Autoencoder**
The **LSTM Autoencoder** is a deep learning model designed for sequence data. It consists of an encoder that compresses the input time-series into a fixed-size representation and a decoder that reconstructs the input from this compressed form. The reconstruction error (i.e., the difference between the original and reconstructed data) is used to detect anomalies—higher errors indicate anomalies.

- **Advantages**: Capable of capturing complex patterns in sequential data, useful for detecting subtle and complex anomalies.
- **Results**: Reconstruction errors and binary labels for anomalies are saved in `anomaly_results_lstm.csv`.

### 3. Visualization

The project also includes interactive visualizations of the time-series data and detected anomalies. We use `Plotly` to create these visualizations, which highlight the detected anomalies across the time-series data for each detection method. The visualizations are saved as HTML files, and can be viewed in any browser.

- **Files**:
    - `anomaly_results_iforest.html`: Interactive plot showing anomalies detected by Isolation Forest.
    - `anomaly_results_z_score.html`: Interactive plot showing anomalies detected by Robust Z-Score.
    - `anomaly_results_lstm.html`: Interactive plot showing anomalies detected by LSTM Autoencoder.

Each plot marks the anomaly points in red, making it easy to visualize where the anomalies occur in the time-series.

---

This project demonstrates the power and flexibility of different algorithms for anomaly detection in time-series data, providing useful insights for tasks such as fraud detection, sensor monitoring, and more.


## Installation and Setup

Follow these steps to set up the project on your local machine:

### Step 1: Clone the repository

To get started, clone the repository from GitHub using the following command:

```bash
git clone https://github.com/riddhiman-ghatak/riddhiman_ghatak_project.git
cd riddhiman_ghatak_project
```
### Step 2: install dependencies
```bash
pip install -r requirements.txt

```
### Step 3: generate synthetic time series data 
```bash
python data_simulation.py

```
### Step 4: run different scripts for anomaly detection
```bash
python robust_z_score_method.py

python isolation_forest_method.py

python using_lstm_autoencoder.py


```
### Step 5: Visualize the detected anomalies
```bash
python visualization.py

```

## Repository Structure

```bash
├── results                       # this folder contains the output html file of visualization.py file
├── data_simulation.py            # Generates synthetic time-series data with anomalies
├── robust_z_score_method.py      # Detects anomalies using the Robust Z-Score method
├── isolation_forest_method.py    # Detects anomalies using Isolation Forest
├── using_lstm_autoencoder.py     # Detects anomalies using LSTM Autoencoder
├── visualization.py              # Visualizes the anomalies detected by each method
├── requirements.txt              # Required dependencies
└── README.md                     # Project documentation
