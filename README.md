# Anomaly Detection in Time-Series Data Project

## Project Overview

This project focuses on detecting anomalies in time-series data using different algorithms. We generate synthetic time-series data with various types of anomalies (spikes, dips, level shifts, trends, variance changes) and apply three distinct anomaly detection techniques:
1. **Isolation Forest** - A machine learning-based algorithm that detects anomalies by randomly partitioning the data.
2. **Robust Z-Score** - A statistical method using median absolute deviation for outlier detection.
3. **LSTM Autoencoder** - A deep learning model that reconstructs the input data and uses reconstruction error to detect anomalies.

The project visualizes these anomalies using interactive plots, and each method's results are saved in separate CSV files.

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
├── data_simulation.py            # Generates synthetic time-series data with anomalies
├── robust_z_score_method.py      # Detects anomalies using the Robust Z-Score method
├── isolation_forest_method.py    # Detects anomalies using Isolation Forest
├── using_lstm_autoencoder.py     # Detects anomalies using LSTM Autoencoder
├── visualization.py              # Visualizes the anomalies detected by each method
├── requirements.txt              # Required dependencies
└── README.md                     # Project documentation
