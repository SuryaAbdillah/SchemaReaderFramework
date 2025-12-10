import json
import multiprocessing as mp
from pathlib import Path
from typing import Dict, Any, List, Tuple


# ---------- Type helpers ----------

def infer_value_type(value):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
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


# ---------- Worker: process a file chunk ----------

def process_chunk(args):
    path, start, end = args
    schema = {}
    valid = invalid = total = 0

    with open(path, "r", encoding="utf-8") as f:
        f.seek(start)

        while f.tell() < end:
            line = f.readline()
            if not line:
                break

            total += 1
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    invalid += 1
                    continue
                valid += 1
            except json.JSONDecodeError:
                invalid += 1
                continue

            for key, value in obj.items():
                vtype = infer_value_type(value)

                if key not in schema:
                    schema[key] = {
                        "type": None,
                        "non_null": 0,
                        "null": 0,
                        "examples": []
                    }

                schema[key]["type"] = merge_type(schema[key]["type"], vtype)

                if value is None:
                    schema[key]["null"] += 1
                else:
                    schema[key]["non_null"] += 1
                    if len(schema[key]["examples"]) < 2:
                        schema[key]["examples"].append(value)

    return schema, total, valid, invalid


# ---------- Merge worker results ----------

def merge_schemas(schema_list: List[Dict[str, Any]]):
    final = {}

    for schema in schema_list:
        for col, meta in schema.items():
            if col not in final:
                final[col] = {
                    "type": meta["type"],
                    "non_null": meta["non_null"],
                    "null": meta["null"],
                    "examples": meta["examples"][:],
                }
            else:
                final[col]["type"] = merge_type(final[col]["type"], meta["type"])
                final[col]["non_null"] += meta["non_null"]
                final[col]["null"] += meta["null"]
                # keep only 2 examples max
                for ex in meta["examples"]:
                    if len(final[col]["examples"]) < 2:
                        final[col]["examples"].append(ex)

    return final


# ---------- Main function for multiprocessing schema ----------

def analyze_jsonl_schema_mp(path, num_workers=8):
    path = Path(path)
    file_size = path.stat().st_size
    chunk_size = file_size // num_workers

    # Prepare chunk boundaries
    offsets = []
    start = 0
    for i in range(num_workers):
        end = start + chunk_size
        if i == num_workers - 1:
            end = file_size
        offsets.append((str(path), start, end))
        start = end

    # Run multiprocessing
    with mp.Pool(num_workers) as pool:
        results = pool.map(process_chunk, offsets)

    # Merge results
    schemas = [r[0] for r in results]
    totals = [r[1] for r in results]
    valids = [r[2] for r in results]
    invalids = [r[3] for r in results]

    final_schema = merge_schemas(schemas)

    return {
        "file": str(path),
        "total_lines": sum(totals),
        "valid_lines": sum(valids),
        "invalid_lines": sum(invalids),
        "columns": final_schema,
        "num_columns": len(final_schema)
    }

def print_schema_table(schema):
    print("=" * 100)
    print(f"FILE: {schema['file']}")
    print(f"Total lines : {schema['total_lines']}")
    print(f"Valid lines : {schema['valid_lines']}")
    print(f"Invalid lines : {schema['invalid_lines']}")
    print(f"Number of columns : {schema['num_columns']}")
    print("=" * 100)

    print(f"{'Column Name':25} | {'Type':10} | {'Non-null':10} | {'Null':10} | Examples")
    print("-" * 100)

    for col, meta in schema["columns"].items():
        cname = col[:25]
        ctype = meta["type"]
        nn = meta["non_null"]
        nl = meta["null"]
        ex = str(meta["examples"])
        print(f"{cname:25} | {ctype:10} | {nn:<10} | {nl:<10} | {ex}")

    print("=" * 100)

