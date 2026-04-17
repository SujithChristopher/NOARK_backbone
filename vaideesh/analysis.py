# Load and analyze the uploaded CSVs
import pandas as pd

cam = pd.read_csv('/home/sujith/Documents/NOARK_backbone/camera_data.csv')
sen = pd.read_csv('/home/sujith/Documents/NOARK_backbone/sensor_data.csv')

# Convert timestamps
cam['timestamp'] = pd.to_datetime(cam['timestamp'])
sen['timestamp'] = pd.to_datetime(sen['timestamp'])

# Compute sampling intervals
cam_dt = cam['timestamp'].diff().dt.total_seconds().dropna()
sen_dt = sen['timestamp'].diff().dt.total_seconds().dropna()

cam_freq = 1 / cam_dt.mean()
sen_freq = 1 / sen_dt.mean()

cam_freq, sen_freq