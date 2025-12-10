import time
import psutil
import os
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
import pyarrow.feather as feather
from fastavro import reader as avro_reader
import json

def measure_cpu_during(func, *args, **kwargs):
    """
    Run any function and measure average CPU usage of THIS process
    during the execution.
    """
    proc = psutil.Process(os.getpid())

    # Prime internal counters
    proc.cpu_percent(interval=None)

    start = time.time()
    result = func(*args, **kwargs)
    elapsed = time.time() - start

    # CPU % used by this process during the elapsed time
    cpu_avg = proc.cpu_percent(interval=None)

    return result, elapsed, cpu_avg

def format_ms(seconds):
    """Convert seconds → milliseconds with 4-decimal precision."""
    return float(f"{seconds * 1000:.4f}")

def read_jsonl(filepath):
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                records.append(obj)
            except:
                continue
    return records

# ------------------------------------------------------------
# Utility – Measure CPU & Memory
# ------------------------------------------------------------
def get_metrics():
    process = psutil.Process(os.getpid())
    return {
        "cpu_percent": psutil.cpu_percent(interval=None),
        "rss_memory_mb": process.memory_info().rss / (1024 * 1024)
    }


# ------------------------------------------------------------
# Benchmark Write (already done in converters)
# ------------------------------------------------------------
def benchmark_write(format_name, convert_func, scan_info, output_dir, fmt, num_workers):
    print(f"\n[Write Benchmark] {format_name}")

    proc = psutil.Process(os.getpid())
    mem_before = proc.memory_info().rss / (1024 * 1024)

    # use the new wrapper
    final_file, elapsed, cpu_avg = measure_cpu_during(
            convert_func,
            scan_info,
            output_dir,
            fmt=fmt,
            num_workers=num_workers
        )

    mem_after = proc.memory_info().rss / (1024 * 1024)
    file_size = os.path.getsize(final_file) / (1024 * 1024)

    return {
        "format": format_name,
        "output_file": final_file,
        "file_size_mb": file_size,
        "write_time_ms": float(f"{elapsed * 1000:.4f}"),
        "cpu_avg_proc_percent": cpu_avg,
        "mem_before_mb": mem_before,
        "mem_after_mb": mem_after,
    }

# ------------------------------------------------------------
# Benchmark Read
# ------------------------------------------------------------
def benchmark_read(filepath, fmt):
    print(f"\n[Read Benchmark] {fmt}")

    proc = psutil.Process(os.getpid())
    mem_before = proc.memory_info().rss / (1024 * 1024)

    # --- CPU + time wrapper ---
    def load_file():
        if fmt == "csv":
            return pd.read_csv(filepath)
        elif fmt == "parquet":
            return pq.read_table(filepath).to_pandas()
        elif fmt == "feather":
            return feather.read_feather(filepath)
        elif fmt == "avro":
            records = []
            with open(filepath, "rb") as f:
                for r in avro_reader(f):
                    records.append(r)
            return pd.DataFrame(records)
        elif fmt == "jsonl":
            records = read_jsonl(filepath)
            return pd.DataFrame(records)
        else:
            raise ValueError("Unknown format")

    df, elapsed, cpu_avg = measure_cpu_during(load_file)

    mem_after = proc.memory_info().rss / (1024 * 1024)

    return {
        "format": fmt,
        "read_time_ms": float(f"{elapsed * 1000:.4f}"),
        "rows_loaded": len(df),
        "cpu_avg_proc_percent": cpu_avg,
        "mem_before_mb": mem_before,
        "mem_after_mb": mem_after,
    }

# ------------------------------------------------------------
# Benchmark Query (simple)
# ------------------------------------------------------------
def is_float_series(series: pd.Series):
    """Check if series contains numeric values represented as strings."""
    try:
        pd.to_numeric(series, errors="raise")
        return True
    except:
        return False

def benchmark_query(df, fmt):
    print(f"\n[Query Benchmark] {fmt}")

    proc = psutil.Process(os.getpid())
    mem_before = proc.memory_info().rss / (1024 * 1024)

    # --- Define workload as a single function for CPU measurement ---
    def run_queries():
        results = {}

        # 1. Projection
        start = time.time()
        col0 = df.columns[0]
        _ = df[[col0]]
        results["projection_ms"] = float(f"{(time.time() - start) * 1000:.4f}")

        # 2. Filtering
        numeric_cols = [c for c in df.columns if is_float_series(df[c])]

        if "temperature" in numeric_cols:
            start = time.time()
            temp_float = pd.to_numeric(df["temperature"])
            _ = temp_float[temp_float > 25]
            results["filter_ms"] = float(f"{(time.time() - start) * 1000:.4f}")
        elif len(numeric_cols) > 0:
            col = numeric_cols[0]
            start = time.time()
            col_float = pd.to_numeric(df[col])
            _ = col_float[col_float > col_float.mean()]
            results["filter_ms"] = float(f"{(time.time() - start) * 1000:.4f}")
        else:
            results["filter_ms"] = None

        # 3. Aggregation
        if len(numeric_cols) > 0:
            start = time.time()
            converted = df[numeric_cols].apply(pd.to_numeric)
            _ = converted.mean()
            results["aggregation_ms"] = float(f"{(time.time() - start) * 1000:.4f}")
        else:
            results["aggregation_ms"] = None

        return results

    # --- Measure CPU + Time during all queries ---
    results, elapsed, cpu_avg = measure_cpu_during(run_queries)

    mem_after = proc.memory_info().rss / (1024 * 1024)

    # Add system metrics
    results.update({
        "cpu_avg_proc_percent": cpu_avg,
        "mem_before_mb": mem_before,
        "mem_after_mb": mem_after,
        "total_query_ms": float(f"{elapsed * 1000:.4f}")
    })

    return results

#def benchmark_query(df, fmt):
#    print(f"\n[Query Benchmark] {fmt}")
#
#    results = {}
#
#    # 1. Projection (always safe)
#    start = time.time()
#    if len(df.columns) > 0:
#        col0 = df.columns[0]
#        _ = df[[col0]]
#    results["projection_ms"] = format_ms(time.time() - start)
#
#    # 2. Filtering (only if numeric column exists)
#    numeric_cols = []
#    for col in df.columns:
#        if is_float_series(df[col]):
#            numeric_cols.append(col)
#
#    if "temperature" in numeric_cols:
#        start = time.time()
#        temp_float = pd.to_numeric(df["temperature"])
#        _ = temp_float[temp_float > 25]
#        results["filter_ms"] = format_ms(time.time() - start)
#    elif len(numeric_cols) > 0:
#        # pick the first numeric column
#        col = numeric_cols[0]
#        start = time.time()
#        col_float = pd.to_numeric(df[col])
#        _ = col_float[col_float > col_float.mean()]  # generic filter
#        results["filter_ms"] = format_ms(time.time() - start)
#    else:
#        results["filter_ms"] = None  # no numeric columns
#
#    # 3. Aggregation (mean of numeric columns)
#    if len(numeric_cols) > 0:
#        start = time.time()
#        converted = df[numeric_cols].apply(pd.to_numeric)
#        _ = converted.mean()
#        results["aggregation_ms"] = format_ms(time.time() - start)
#    else:
#        results["aggregation_ms"] = None
#
#    return results
