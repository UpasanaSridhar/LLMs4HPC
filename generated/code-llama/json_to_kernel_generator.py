import json

def flatten_json(json_data):
    """
    Flattens a hierarchical JSON object into a single-level dictionary.
    
    :param json_data: JSON object (dictionary)
    :return: Flattened dictionary
    """
    def recursive_flatten(d, parent_key='', sep='_'):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(recursive_flatten(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    return recursive_flatten(json_data)

def read_and_flatten_json(json_file):
    """
    Reads a JSON file, flattens its content, and returns it as a dictionary.

    :param json_file: Path to the JSON file.
    :return: Flattened JSON dictionary.
    """
    with open(json_file, 'r') as file:
        json_data = json.load(file)
    return flatten_json(json_data)

def main():
    json_file = "avx2_double_ISA.json"  # Replace with your JSON file name

    try:
        flattened_json = read_and_flatten_json(json_file)
        print("Flattened JSON:")
        print(json.dumps(flattened_json, indent=4))
    except Exception as e:
        print(f"Error reading or flattening JSON: {e}")

if __name__ == "__main__":
    main()
