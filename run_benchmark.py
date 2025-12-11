import re
import subprocess
from pathlib import Path
import sys

import pandas as pd

########################################
# EXPERIMENT CONFIGURATION
########################################

# List of JSONL files to benchmark
JSONL_FILES = [
    Path(r"<path_to_dataset>.jsonl"),
]

# Combinations of chunk_size & num_workers
CHUNK_SIZES = [10_000, 50_000, 10_000, 50_000]
NUM_WORKERS_LIST = [1, 2, 3, 4, 5]

# Number of repeated experiments per combination
N_EXPERIMENTS = 5

# Formats supported by main.py
FORMATS = ["jsonl", "csv", "parquet", "avro", "feather"]

# Python executable & main script
PYTHON_EXE = sys.executable
MAIN_SCRIPT = Path("main.py")

# Conversion folder (can be overwritten many times)
CONVERT_OUTPUT_DIR = Path("output")

# Root directory for all automated benchmark outputs
BASE_OUTPUT = Path("automate_benchmark_outputs")


########################################
# HELPER FUNCTIONS
########################################

def infer_rows_from_filename(path: Path) -> int | None:
    """
    Try to infer the number of rows from the filename.
      *_10k_valid.jsonl -> 10_000
      *_1m_valid.jsonl  -> 1_000_000
    """
    name = path.name.lower()
    m = re.search(r"_(\d+)([km])?_valid\.jsonl$", name)
    if not m:
        return None

    n = int(m.group(1))
    suffix = m.group(2)
    if suffix == "k":
        return n * 1_000
    if suffix == "m":
        return n * 1_000_000
    return n


def run_benchmark_for_combo(
    jsonl_path: Path,
    chunk_size: int,
    num_workers: int,
    combo_dir: Path,
    exp_id: int,
):
    """
    Run:
      - main.py --mode benchmark
      - main.py --mode ml_benchmark
    for a single combination (file, chunk_size, num_workers, experiment id).

    Generated CSVs:
      combo_dir / benchmark_results_experiment_{exp_id} / ...
      combo_dir / ml_benchmark_results_experiment_{exp_id} / ...
    """
    bench_exp_dir = combo_dir / f"benchmark_results_experiment_{exp_id}"
    ml_exp_dir = combo_dir / f"ml_benchmark_results_experiment_{exp_id}"

    bench_exp_dir.mkdir(parents=True, exist_ok=True)
    ml_exp_dir.mkdir(parents=True, exist_ok=True)
    CONVERT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) IO benchmark
    cmd_bench = [
        PYTHON_EXE,
        str(MAIN_SCRIPT),
        "--mode", "benchmark",
        "--chunk-size", str(chunk_size),
        "--input-data", str(jsonl_path),
        "--num-workers", str(num_workers),
        "--formats", *FORMATS,
        "--output-dir", str(CONVERT_OUTPUT_DIR),
        "--benchmark-csv-dir", str(bench_exp_dir),
        "--ml-benchmark-csv-dir", str(ml_exp_dir),  # ignored by mode=benchmark, safe to pass
    ]
    print("\n[RUN IO BENCHMARK]", " ".join(cmd_bench))
    subprocess.run(cmd_bench, check=True)

    # 2) ML benchmark
    cmd_ml = [
        PYTHON_EXE,
        str(MAIN_SCRIPT),
        "--mode", "ml_benchmark",
        "--input-data", str(jsonl_path),
        "--formats", *FORMATS,
        "--output-dir", str(CONVERT_OUTPUT_DIR),
        "--ml-benchmark-csv-dir", str(ml_exp_dir),
    ]
    print("\n[RUN ML BENCHMARK]", " ".join(cmd_ml))
    subprocess.run(cmd_ml, check=True)


def collect_benchmark_csvs_for_combo(
    combo_dir: Path,
    stem: str,
    kind: str,
) -> pd.DataFrame:
    """
    Collect all:
      combo_dir / benchmark_results_experiment_*/benchmark_{kind}_{stem}.csv
    and concatenate them.

    kind: "write", "read", "query"
    """
    pattern = f"benchmark_results_experiment_*/benchmark_{kind}_{stem}.csv"
    paths = list(combo_dir.glob(pattern))

    dfs = []
    for p in paths:
        # extract experiment ID from path
        m_exp = re.search(r"benchmark_results_experiment_(\d+)", str(p))
        exp_id = int(m_exp.group(1)) if m_exp else None

        df = pd.read_csv(p)
        df["experiment"] = exp_id

        # if memory columns exist, compute delta
        if "mem_before_mb" in df.columns and "mem_after_mb" in df.columns:
            df["mem_delta_mb"] = df["mem_after_mb"] - df["mem_before_mb"]

        dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


def summarize_benchmark(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize IO benchmark:
      - group by 'format' (and any other grouping columns if needed),
      - average all numeric columns,
      - for memory, mem_delta_mb is the main indicator.
    """
    if df.empty:
        return df

    df = df.copy()

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    # 'experiment' is an ID, do not average it
    if "experiment" in numeric_cols:
        numeric_cols.remove("experiment")

    group_cols = [c for c in ["format"] if c in df.columns]

    if not group_cols:
        # fallback: just mean of all numeric columns
        summary = df[numeric_cols].mean().to_frame().T
        summary["n_runs"] = len(df)
        return summary

    grouped = df.groupby(group_cols)
    summary = grouped[numeric_cols].mean().reset_index()
    summary["n_runs"] = grouped.size().values

    return summary


def collect_ml_csvs_for_combo(
    combo_dir: Path,
    stem: str,
    model: str,
) -> pd.DataFrame:
    """
    Collect all:
      combo_dir / ml_benchmark_results_experiment_*/ml_benchmark_{model}_{stem}.csv

    model: "lr", "svm", "mlp"
    """
    pattern = f"ml_benchmark_results_experiment_*/ml_benchmark_{model}_{stem}.csv"
    paths = list(combo_dir.glob(pattern))

    dfs = []
    for p in paths:
        m_exp = re.search(r"ml_benchmark_results_experiment_(\d+)", str(p))
        exp_id = int(m_exp.group(1)) if m_exp else None

        df = pd.read_csv(p)
        df["experiment"] = exp_id
        dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


def summarize_ml_benchmark(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize ML benchmark:
      - group by 'format' (if available),
      - average all numeric columns.
    """
    if df.empty:
        return df

    df = df.copy()
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if "experiment" in numeric_cols:
        numeric_cols.remove("experiment")

    group_cols = [c for c in ["format"] if c in df.columns]

    if not group_cols:
        summary = df[numeric_cols].mean().to_frame().T
        summary["n_runs"] = len(df)
        return summary

    grouped = df.groupby(group_cols)
    summary = grouped[numeric_cols].mean().reset_index()
    summary["n_runs"] = grouped.size().values

    return summary


########################################
# MAIN LOOP
########################################

def main():
    BASE_OUTPUT.mkdir(parents=True, exist_ok=True)

    for jsonl_path in JSONL_FILES:
        stem = jsonl_path.stem
        n_rows = infer_rows_from_filename(jsonl_path)

        dataset_dir = BASE_OUTPUT / stem
        dataset_dir.mkdir(parents=True, exist_ok=True)

        print("\n==============================")
        print(f"DATASET : {stem}")
        print(f"ROWS    : {n_rows if n_rows is not None else 'unknown'}")
        print(f"OUT DIR : {dataset_dir}")
        print("==============================")

        for chunk_size in CHUNK_SIZES:
            # skip if chunk_size > n_rows (if n_rows can be inferred)
            if n_rows is not None and chunk_size > n_rows:
                print(f"[SKIP] chunk_size={chunk_size} > n_rows={n_rows}")
                continue

            for num_workers in NUM_WORKERS_LIST:
                combo_dir = dataset_dir / f"cs{chunk_size}_nw{num_workers}"
                combo_dir.mkdir(parents=True, exist_ok=True)

                print(
                    f"\n>>> Combination: file={stem}, chunk_size={chunk_size}, num_workers={num_workers}"
                )

                # Run multiple experiments for this combination
                for exp_id in range(1, N_EXPERIMENTS + 1):
                    print(f"\n=== EXPERIMENT {exp_id}/{N_EXPERIMENTS} ===")
                    run_benchmark_for_combo(
                        jsonl_path=jsonl_path,
                        chunk_size=chunk_size,
                        num_workers=num_workers,
                        combo_dir=combo_dir,
                        exp_id=exp_id,
                    )

                # =====================
                # SUMMARY FOR THIS COMBINATION
                # =====================
                print(f"\n=== SUMMARY for {stem} | cs={chunk_size} | nw={num_workers} ===")

                # 1) IO benchmark
                for kind in ["write", "read", "query"]:
                    df_bench = collect_benchmark_csvs_for_combo(combo_dir, stem, kind)
                    if df_bench.empty:
                        print(f"[WARN] No IO benchmark data for kind={kind} in this combination")
                        continue

                    summary_bench = summarize_benchmark(df_bench)
                    out_path = combo_dir / f"summary_benchmark_{kind}_{stem}.csv"
                    summary_bench.to_csv(out_path, index=False)
                    print(f"[SAVE] IO summary ({kind}) -> {out_path}")

                # 2) ML benchmark
                for model in ["lr", "svm", "mlp"]:
                    df_ml = collect_ml_csvs_for_combo(combo_dir, stem, model)
                    if df_ml.empty:
                        print(f"[WARN] No ML benchmark data for model={model} in this combination")
                        continue

                    summary_ml = summarize_ml_benchmark(df_ml)
                    out_path = combo_dir / f"summary_ml_benchmark_{model}_{stem}.csv"
                    summary_ml.to_csv(out_path, index=False)
                    print(f"[SAVE] ML summary ({model}) -> {out_path}")


if __name__ == "__main__":
    main()
