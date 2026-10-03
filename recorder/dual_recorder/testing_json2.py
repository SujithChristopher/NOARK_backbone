import json

with open("/home/sujith/imx219_waveshare.json") as f:
    tuning = json.load(f)

for algo in tuning.get("algorithms", []):
    if "rpi.alsc" in algo:
        print(json.dumps(algo["rpi.alsc"], indent=2))