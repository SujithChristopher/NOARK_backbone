import json, shutil, os

src = "/usr/share/libcamera/ipa/rpi/pisp/imx219.json"
dst = os.path.expanduser("~/imx219_waveshare.json")

shutil.copy(src, dst)

with open(dst, "r") as f:
    tuning = json.load(f)

# Identity CCM — no colour correction, neutral output
identity_ccm = [
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0
]

for algo in tuning.get("algorithms", []):

    # Flatten all ALSC tables
    if "rpi.alsc" in algo:
        alsc = algo["rpi.alsc"]
        alsc["omega"] = 0.0
        alsc["n_iter"] = 0
        alsc["luminance_strength"] = 0.0
        if "luminance_lut" in alsc:
            alsc["luminance_lut"] = [1.0] * len(alsc["luminance_lut"])
        for key in ["calibrations_Cr", "calibrations_Cb"]:
            if key in alsc:
                for entry in alsc[key]:
                    entry["table"] = [1.0] * len(entry["table"])
        print("Flattened rpi.alsc")

    # Replace all CCM entries with identity
    if "rpi.ccm" in algo:
        for entry in algo["rpi.ccm"]["ccms"]:
            entry["ccm"] = identity_ccm
        print("Replaced rpi.ccm with identity")

with open(dst, "w") as f:
    json.dump(tuning, f, indent=4)

print("Done")