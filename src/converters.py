import json
import os
from pathlib import Path
import multiprocessing as mp
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.feather as feather
from fastavro import writer, parse_schema
from tqdm import tqdm


# ------------------------------------------------------------
# Normalize nested values → stringify
# ------------------------------------------------------------
def normalize_record(obj):
    new = {}
    for k, v in obj.items():
        if v is None:
            new[k] = None
        else:
            if isinstance(v, (dict, list)):
                new[k] = json.dumps(v)
            else:
                new[k] = str(v)   # 🔥 convert everything to string
    return new

# ------------------------------------------------------------
# Read chunk of JSONL into DataFrame
# ------------------------------------------------------------
def read_chunk(path, start_line, end_line):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < start_line:
                continue
            if i >= end_line:
                break

            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
                rows.append(normalize_record(obj))
            except:
                continue

    if len(rows) == 0:
        return pd.DataFrame()

    return pd.DataFrame(rows)


# ------------------------------------------------------------
# Worker for conversion
# ------------------------------------------------------------
def convert_worker(args):
    import csv
    (path, start, end, fmt, temp_file) = args

    df = read_chunk(path, start, end)
    if df.empty:
        return temp_file

    if fmt == "csv":
        df.to_csv(
        temp_file,
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
        doublequote=True
    )

    elif fmt == "parquet":
        table = pa.Table.from_pandas(df)
        pq.write_table(table, temp_file)

    elif fmt == "feather":
        table = pa.Table.from_pandas(df)
        feather.write_feather(table, temp_file)

    elif fmt == "avro":
        records = df.to_dict(orient="records")
        avro_schema = {
            "type": "record",
            "name": "Row",
            "fields": [
                {"name": col, "type": ["null", "string"]}  # simplified for all columns
                for col in df.columns
            ]
        }
        parsed = parse_schema(avro_schema)
        with open(temp_file, "wb") as out:
            writer(out, parsed, records)
    elif fmt == "jsonl":
        with open(temp_file, "w", encoding="utf-8") as fout:
            for _, row in df.iterrows():
                fout.write(json.dumps(row.to_dict()) + "\n")

    return temp_file


# ------------------------------------------------------------
# Main converter with tqdm
# ------------------------------------------------------------
def convert_jsonl(
    scan_info,
    output_dir,
    fmt="csv",
    num_workers=4
):
    path = scan_info["path"]
    chunk_ranges = scan_info["chunk_ranges"]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare worker tasks
    tasks = []
    temp_files = []
    for i, (s, e) in enumerate(chunk_ranges):
        temp_path = output_dir / f"temp_chunk_{i}.{fmt}"
        temp_files.append(temp_path)
        tasks.append((str(path), s, e, fmt, temp_path))

    # Run multiprocessing with tqdm
    print("\nProcessing chunks...")
    with mp.Pool(num_workers) as pool:
        list(tqdm(
            pool.imap(convert_worker, tasks),
            total=len(tasks),
            desc="Converting",
            ncols=100
        ))

    # Merge phase
    print("\nMerging output files...")
    input_stem = Path(path).stem
    final_file = output_dir / f"{input_stem}.{fmt}"

    # -------------------- CSV MERGING ------------------------
    if fmt == "csv":
        first = True
        with open(final_file, "w", encoding="utf-8") as out:
            for temp in tqdm(temp_files, desc="Merging CSV", ncols=100):
                if not temp.exists():
                    continue
                with open(temp, "r", encoding="utf-8") as f:
                    if first:
                        out.write(f.read())
                        first = False
                    else:
                        f.readline()  # skip header
                        out.write(f.read())

    # ------------------- PARQUET / FEATHER MERGING ----------
    elif fmt in ("parquet", "feather"):
        frames = []
        for temp in tqdm(temp_files, desc="Loading chunks", ncols=100):
            if fmt == "parquet":
                frames.append(pd.read_parquet(temp))
            else:
                frames.append(pd.read_feather(temp))

        df_final = pd.concat(frames, ignore_index=True)
        table = pa.Table.from_pandas(df_final)

        if fmt == "parquet":
            pq.write_table(table, final_file)
        else:
            feather.write_feather(table, final_file)

    # ---------------------- AVRO MERGING ----------------------
    elif fmt == "avro":
        all_records = []
        import fastavro

        for temp in tqdm(temp_files, desc="Reading AVRO", ncols=100):
            try:
                with open(temp, "rb") as f:
                    for record in fastavro.reader(f):
                        all_records.append(record)
            except:
                pass

        if len(all_records) > 0:
            avro_schema = {
                "type": "record",
                "name": "Row",
                "fields": [
                    {"name": k, "type": ["null", "string"]}
                    for k in all_records[0].keys()
                ]
            }
            parsed = parse_schema(avro_schema)
            with open(final_file, "wb") as out:
                writer(out, parsed, all_records)

    # ---------------------- JSONL MERGING ------------------------
    elif fmt == "jsonl":
        with open(final_file, "w", encoding="utf-8") as out:
            for temp in tqdm(temp_files, desc="Merging JSONL", ncols=100):
                if temp.exists():
                    with open(temp, "r", encoding="utf-8") as f:
                        out.write(f.read())

    # Cleanup
    print("\nCleaning up temporary files...")
    for t in temp_files:
        if t.exists():
            t.unlink()

    print(f"\n[OK] Conversion complete → {final_file}")
    return final_file

