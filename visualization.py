import pandas as pd
import plotly.express as px



df_iforest = pd.read_csv('anomaly_results_iforest.csv')
df_iforest['Time'] = pd.to_datetime(df_iforest['Time'])
fig_iforest = px.line(df_iforest, x='Time', y='Value', title='Time vs Value with Anomalies marked using isolation forest')
anomalies_iforest = df_iforest[df_iforest['is_anomaly']]
fig_iforest.add_scatter(x=anomalies_iforest['Time'], y=anomalies_iforest['Value'], mode='markers', 
                         marker=dict(color='red', size=10), name='Anomalies')
fig_iforest.update_layout(plot_bgcolor='black', paper_bgcolor='black', font_color='white')
fig_iforest.show()
fig_iforest.write_html('anomaly_results_iforest.html')




df_lstm = pd.read_csv('anomaly_results_lstm.csv')
df_lstm['Time'] = pd.to_datetime(df_lstm['Time'])
fig_lstm = px.line(df_lstm, x='Time', y='Value', title='Time vs Value with Anomalies Marked using lstm autoencoder')
anomalies_lstm = df_lstm[df_lstm['is_anomaly']]
fig_lstm.add_scatter(x=anomalies_lstm['Time'], y=anomalies_lstm['Value'], mode='markers', 
                      marker=dict(color='red', size=10), name='Anomalies')
fig_lstm.update_layout(plot_bgcolor='black', paper_bgcolor='black', font_color='white')
fig_lstm.show()
fig_lstm.write_html('anomaly_results_lstm.html')





df_z_score = pd.read_csv('anomaly_results_z_score.csv')
df_z_score['Time'] = pd.to_datetime(df_z_score['Time'])
fig_z_score = px.line(df_z_score, x='Time', y='Value', title='Time vs Value with Anomalies Marked using robust z_score method')
anomalies_z_score = df_z_score[df_z_score['is_anomaly']]
fig_z_score.add_scatter(x=anomalies_z_score['Time'], y=anomalies_z_score['Value'], mode='markers', 
                         marker=dict(color='red', size=10), name='Anomalies')
fig_z_score.update_layout(plot_bgcolor='black', paper_bgcolor='black', font_color='white')
fig_z_score.show()
fig_z_score.write_html('anomaly_results_z_score.html')
