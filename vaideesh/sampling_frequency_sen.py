import pandas as pd
import numpy as np

df = pd.read_csv("sensor_data.csv")

df['timestamp'] = pd.to_datetime(df['timestamp'])

# FIX: divide by 1e6 (microseconds → seconds)
time_seconds = df['timestamp'].astype('int64') / 1e6

time_diffs = np.diff(time_seconds)

print("Mean interval:", np.mean(time_diffs))
print("Sampling rate:", 1 / np.mean(time_diffs))