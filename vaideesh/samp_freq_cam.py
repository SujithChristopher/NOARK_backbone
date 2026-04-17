import pandas as pd

df = pd.read_csv("camera_data.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Each row IS one frame — just diff the timestamps directly
time_diffs = df['timestamp'].diff().dt.total_seconds().dropna()
time_diffs = time_diffs[time_diffs > 0]      # remove zeros
time_diffs = time_diffs[time_diffs < 0.2]    # remove spikes >200ms

print(f"Total frames        : {len(df)}")
print(f"Average FPS         : {1 / time_diffs.mean():.2f}")
print(f"Min FPS (slowest)   : {1 / time_diffs.max():.2f}")
print(f"Max FPS (fastest)   : {1 / time_diffs.min():.2f}")
print(f"Avg interval        : {time_diffs.mean()*1000:.2f} ms")
print(f"Spike frames >200ms : {(df['timestamp'].diff().dt.total_seconds() > 0.2).sum()}")