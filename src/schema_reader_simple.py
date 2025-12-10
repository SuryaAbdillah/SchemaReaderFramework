import json
from pathlib import Path


def infer_value_type(value):
    if value is None: return "null"
    if isinstance(value, bool): return "bool"
    if isinstance(value, int): return "int"
    if isinstance(value, float): return "float"
    if isinstance(value, str): return "string"
    if isinstance(value, list): return "array"
    if isinstance(value, dict): return "object"
    return "unknown"


def merge_type(a, b):
    if a is None:
        return b
    if a == b:
        return a
    if a == "null":
        return b
    if b == "null":
        return a
    return "mixed"


def analyze_jsonl_schema_fast(path, sample_lines=10):
    """
    FAST schema: only read first `sample_lines` lines.
    Does NOT scan entire file. Suitable for 500M+ rows.
    """
    path = Path(path)
    schema = {}
    read_lines = 0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if read_lines >= sample_lines:
                break

            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except:
                continue

            # update schema
            for key, value in obj.items():
                vtype = infer_value_type(value)

                if key not in schema:
                    schema[key] = {
                        "type": vtype,
                        "examples": []
                    }
                else:
                    schema[key]["type"] = merge_type(schema[key]["type"], vtype)

                # keep 1–2 examples
                if len(schema[key]["examples"]) < 2:
                    schema[key]["examples"].append(value)

            read_lines += 1

    return {
        "file": str(path),
        "num_columns": len(schema),
        "columns": schema,
        "sampled_lines": read_lines,
    }


def print_schema_table_fast(schema):
    print("=" * 100)
    print(f"FILE: {schema['file']}")
    print(f"Sampled lines : {schema['sampled_lines']}")
    print(f"Number of columns : {schema['num_columns']}")
    print("=" * 100)

    print(f"{'Column Name':25} | {'Type':10} | Examples")
    print("-" * 100)

    for col, meta in schema["columns"].items():
        print(f"{col[:25]:25} | {meta['type']:10} | {meta['examples']}")

    print("=" * 100)

