import json

with open("/home/sujith/imx219_waveshare.json") as f:
    tuning = json.load(f)

for algo in tuning.get("algorithms", []):
    print(list(algo.keys()))