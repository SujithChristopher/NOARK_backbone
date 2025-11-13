#!/usr/bin/env python3
"""
Convert data.json (array format) to patients.json (nested format)
"""

import json
from pathlib import Path


def convert_data_to_patients_format():
    """Convert array format to patients.json nested format"""
    data_file = Path("data.json")
    patients_file = Path("data_converted.json")

    if not data_file.exists():
        print(f"Error: {data_file} not found")
        return

    try:
        print(f"Reading {data_file}...")
        with open(data_file, 'r', encoding='utf-8') as f:
            data_array = json.load(f)

        print(f"Found {len(data_array)} patients")

        # Convert array to patients format
        patients_register = {}
        for patient in data_array:
            hospital_id = patient.get("hospital_id", "UNKNOWN")
            # Create a copy without hospital_id since it's the key
            patient_data = patient.copy()
            patient_data.pop("hospital_id", None)
            patients_register[hospital_id] = patient_data

        # Create the final structure
        patients_data = {
            "current_patient_id": "",
            "patient_register": patients_register
        }

        print(f"Writing to {patients_file}...")
        with open(patients_file, 'w', encoding='utf-8') as f:
            json.dump(patients_data, f, indent=2)

        print(f"Success! Converted to {patients_file}")

        # Print sample
        print(f"\nSample structure:")
        sample_keys = list(patients_register.keys())[:2]
        for key in sample_keys:
            print(f'"{key}": {json.dumps(patients_register[key], indent=2)}')

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    convert_data_to_patients_format()
