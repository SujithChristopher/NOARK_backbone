import pandas as pd

df = pd.read_csv("camera_data.csv")

# Convert timestamps
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Compute time differences
time_diffs = df['timestamp'].diff().dt.total_seconds().dropna()

# Remove zero or negative intervals
time_diffs = time_diffs[time_diffs > 0]

print(f"Average FPS: {1/time_diffs.mean():.2f}")
print(f"Min FPS: {1/time_diffs.max():.2f}")
print(f"Max FPS: {1/time_diffs.min():.2f}")