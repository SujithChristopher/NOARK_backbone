import os
import sys
import toml
import json

from platform import system

OS_TYPE = system()

"Check if path exists"
if os.path.exists(os.path.expanduser("~/Documents/NOARK")):
    print("Path exists")
else:
    os.makedirs(os.path.expanduser("~/Documents/NOARK"))
    print("Path created")

_hospital_ids = []
"Check if file exists"
if os.path.exists(os.path.expanduser("~/Documents/NOARK/data.json")):
    print("File exists")
    with open(os.path.expanduser("~/Documents/NOARK/data.json"), "r") as f:
        data = json.load(f)

        for _d in data:
            _hospital_ids.append(_d["hospital_id"])
            "check if folder with hospital id name exists"
            if not os.path.exists(
                os.path.expanduser(f'~/Documents/NOARK/data/{_d["hospital_id"]}')
            ):
                os.makedirs(
                    os.path.expanduser(f'~/Documents/NOARK/data/{_d["hospital_id"]}')
                )
                print(f'Folder created for {_d["hospital_id"]}')

else:
    print("File does not exist")
