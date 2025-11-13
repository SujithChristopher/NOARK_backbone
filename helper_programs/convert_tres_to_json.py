#!/usr/bin/env python3
"""
Convert patient_register.tres (Godot resource file) to data.json format
"""

import json
import re
from pathlib import Path


def parse_tres_file(tres_path):
    """Parse the patient_register.tres file and extract the patient dictionary"""
    with open(tres_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract the patient_register dictionary from the file
    # Find the start of patient_register and parse it
    start_idx = content.find('patient_register = {')
    if start_idx == -1:
        raise ValueError("Could not find patient_register in .tres file")

    # Find matching closing brace
    brace_count = 0
    in_string = False
    escape_next = False
    end_idx = start_idx + len('patient_register = ')

    for i in range(end_idx, len(content)):
        char = content[i]

        if escape_next:
            escape_next = False
            continue

        if char == '\\':
            escape_next = True
            continue

        if char == '"' and not escape_next:
            in_string = not in_string
            continue

        if not in_string:
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i + 1
                    break

    dict_str = content[start_idx + len('patient_register = '):end_idx]

    # Parse the GDScript dictionary to Python dict
    patients_dict = {}

    # Find all patient entries: "ID": { ... }
    i = 0
    while i < len(dict_str):
        # Find opening quote for hospital ID
        if dict_str[i] == '"':
            j = i + 1
            while j < len(dict_str) and dict_str[j] != '"':
                j += 1
            hospital_id = dict_str[i + 1:j]

            # Find colon
            k = j + 1
            while k < len(dict_str) and dict_str[k] != ':':
                k += 1

            # Find opening brace
            m = k + 1
            while m < len(dict_str) and dict_str[m] != '{':
                m += 1

            # Find matching closing brace
            brace_count = 0
            in_string = False
            escape_next = False
            n = m

            for idx in range(m, len(dict_str)):
                char = dict_str[idx]

                if escape_next:
                    escape_next = False
                    continue

                if char == '\\':
                    escape_next = True
                    continue

                if char == '"' and not escape_next:
                    in_string = not in_string
                    continue

                if not in_string:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            n = idx + 1
                            break

            patient_data_str = dict_str[m:n]

            # Parse individual fields
            patient = {}

            # Extract each field from the patient data string
            field_pattern = r'"(\w+)":\s*([^,\n}]+)'
            for field_match in re.finditer(field_pattern, patient_data_str):
                field_name = field_match.group(1)
                field_value = field_match.group(2).strip()

                # Clean up the value
                if field_value.startswith('"') and field_value.endswith('"'):
                    # String value
                    patient[field_name] = field_value[1:-1]
                else:
                    # Try to parse as number
                    try:
                        if '.' in field_value:
                            patient[field_name] = float(field_value)
                        else:
                            patient[field_name] = float(int(field_value))
                    except ValueError:
                        patient[field_name] = field_value

            patients_dict[hospital_id] = patient
            i = n
        else:
            i += 1

    return patients_dict


def convert_to_json_format(patients_dict):
    """Convert dictionary format to JSON array format"""
    patients_list = []

    for hospital_id, patient_data in patients_dict.items():
        patient_obj = patient_data.copy()
        # Ensure hospital_id is included
        patient_obj["hospital_id"] = hospital_id
        patients_list.append(patient_obj)

    return patients_list


def main():
    # Paths
    tres_file = Path("patient_register.tres")
    json_file = Path("data.json")

    if not tres_file.exists():
        print(f"Error: {tres_file} not found")
        return

    try:
        print(f"Reading {tres_file}...")
        patients_dict = parse_tres_file(tres_file)
        print(f"Found {len(patients_dict)} patients")

        print(f"Converting to JSON format...")
        patients_list = convert_to_json_format(patients_dict)

        print(f"Writing to {json_file}...")
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(patients_list, f, indent=2)

        print(f"Success! Converted {len(patients_list)} patients to {json_file}")

        # Print sample
        print(f"\nSample entry:")
        print(json.dumps(patients_list[0], indent=2))

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
