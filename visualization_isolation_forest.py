import pandas as pd
import plotly.express as px

# Load the CSV file
df = pd.read_csv('anomaly_results_iforest.csv')  # Replace 'your_file.csv' with your actual file path

# Convert 'Time' column to datetime
df['Time'] = pd.to_datetime(df['Time'])

# Create the plot
fig = px.line(df, x='Time', y='Value', title='Time vs Value with Anomalies marked using isolation forest')

# Add scatter points for anomalies
anomalies = df[df['is_anomaly']]
fig.add_scatter(x=anomalies['Time'], y=anomalies['Value'], mode='markers', 
                 marker=dict(color='red', size=10), name='Anomalies')

# Update layout for dark background
fig.update_layout(
    plot_bgcolor='black',  # Background color for the plot area
    paper_bgcolor='black',  # Background color for the entire figure
    font_color='white'      # Font color
)

# Show the plot
fig.show()
