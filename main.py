import argparse
from src.schema_reader_simple import analyze_jsonl_schema_fast, print_schema_table_fast
from src.converters import convert_jsonl
from src.benchmark import benchmark_write, benchmark_read, benchmark_query, read_jsonl
from src.file_scanner import scan_jsonl_file
import pandas as pd
import os
from pathlib import Path
from src.utils import print_table
from src.utils import print_ml_table
from src.benchmark_ml import *
from tabulate import tabulate

def flatten_columns(df):
    df = df.copy()
    df.columns = [
        f"{col[0]}_{col[1]}" if isinstance(col, tuple) else col
        for col in df.columns
    ]
    return df

def flatten_stats_df(df):
    """
    Flatten MultiIndex columns like:
    write_time_ms -> write_time_ms_mean, write_time_ms_std
    """
    df_flat = df.copy()
    df_flat.columns = [
        f"{col[0]}_{col[1]}" for col in df_flat.columns
    ]
    return df_flat.reset_index()

def main_schema(input_path, sample_lines):
    schema = analyze_jsonl_schema_fast(input_path, sample_lines=sample_lines)
    print_schema_table_fast(schema)


def main_converter(input_data, chunk_size, num_workers, formats, output_dir):
    scan_info = scan_jsonl_file(input_data, chunk_size=chunk_size)

    for fmt in formats:
        out_dir = f"{output_dir}/{fmt}"
        convert_jsonl(scan_info, output_dir=out_dir, fmt=fmt, num_workers=num_workers)

def main_benchmark(
    input_data,
    chunk_size,
    num_workers,
    formats,
    output_dir,
    benchmark_csv_dir=None,
    repeat=5
):
    scan_info = scan_jsonl_file(input_data, chunk_size=chunk_size)

    # Define folder to save benchmark results
    if benchmark_csv_dir is None:
        csv_dir = Path.cwd() / "benchmark_results"
    else:
        csv_dir = Path(benchmark_csv_dir)

    csv_dir.mkdir(parents=True, exist_ok=True)

    all_write = []
    all_read = []
    all_query = []

    input_stem = Path(input_data).stem

    print(f"\nRunning benchmark {repeat}× for robust evaluation...\n")

    for run_id in range(repeat):
        print(f"\n=== Iteration {run_id + 1}/{repeat} ===")

        results_write = []
        results_read = []
        results_query = []

        for fmt in formats:
            fmt_output_dir = f"{output_dir}/{fmt}"

            # ---------------- WRITE BENCHMARK ----------------
            write_res = benchmark_write(
                format_name=fmt,
                convert_func=convert_jsonl,
                scan_info=scan_info,
                output_dir=fmt_output_dir,
                fmt=fmt,
                num_workers=num_workers
            )

            write_res["run"] = run_id + 1
            results_write.append(write_res)

            # ---------------- READ BENCHMARK ----------------
            read_res = benchmark_read(write_res["output_file"], fmt)
            read_res["format"] = fmt
            read_res["run"] = run_id + 1
            results_read.append(read_res)

            # ---------------- QUERY BENCHMARK ----------------
            if fmt == "jsonl":
                df = pd.DataFrame(read_jsonl(input_data))
            elif fmt == "csv":
                df = pd.read_csv(write_res["output_file"])
            elif fmt == "parquet":
                df = pd.read_parquet(write_res["output_file"])
            elif fmt == "feather":
                df = pd.read_feather(write_res["output_file"])
            else:
                import fastavro
                rec = []
                with open(write_res["output_file"], "rb") as f:
                    for r in fastavro.reader(f):
                        rec.append(r)
                df = pd.DataFrame(rec)

            query_res = benchmark_query(df, fmt)
            query_res = {"run": run_id + 1, "format": fmt, **query_res}
            results_query.append(query_res)

        # Collect this run
        all_write.append(
            pd.DataFrame(results_write).drop(columns=["output_file"], errors="ignore")
        )
        all_read.append(pd.DataFrame(results_read))
        all_query.append(pd.DataFrame(results_query))

    # ================= AGGREGATE ALL RUNS =================
    df_write_all = pd.concat(all_write, ignore_index=True)
    df_read_all = pd.concat(all_read, ignore_index=True)
    df_query_all = pd.concat(all_query, ignore_index=True)

    # ================= MEAN & STD =================
    df_write_all_clean = df_write_all.drop(columns=["run"], errors="ignore")
    df_read_all_clean  = df_read_all.drop(columns=["run"], errors="ignore")
    df_query_all_clean = df_query_all.drop(columns=["run"], errors="ignore")

    df_write_stats = df_write_all_clean.groupby("format").agg(["mean", "std"])
    df_read_stats = df_read_all_clean.groupby("format").agg(["mean", "std"])
    df_query_stats = df_query_all_clean.groupby("format").agg(["mean", "std"])

    # -------- Flatten for console printing --------
    df_write_stats_flat = flatten_stats_df(df_write_stats).reset_index(drop=True)
    df_read_stats_flat  = flatten_stats_df(df_read_stats).reset_index(drop=True)
    df_query_stats_flat = flatten_stats_df(df_query_stats).reset_index(drop=True)
    
    # -------- Console output --------
    print_table(
        "WRITE BENCHMARK RESULTS — MEAN ± STD",
        df_write_stats_flat
    )
    
    print_table(
        "READ BENCHMARK RESULTS — MEAN ± STD",
        df_read_stats_flat
    )
    
    print_table(
        "QUERY BENCHMARK RESULTS — MEAN ± STD",
        df_query_stats_flat
    )

    # ================= SAVE CSV =================
    write_order = ["run", "format", "file_size_mb", "write_time_ms", "cpu_avg_proc_percent", "mem_before_mb", "mem_after_mb"]
    df_write_all = df_write_all[write_order]
    df_write_all.to_csv(
        csv_dir / f"benchmark_write_{input_stem}_all_runs.csv",
        float_format="%.4f",
        index=False
    )
    read_order = ["run", "format", "read_time_ms", "cpu_avg_proc_percent", "mem_before_mb", "mem_after_mb"]
    df_read_all = df_read_all[read_order]
    df_read_all.to_csv(
        csv_dir / f"benchmark_read_{input_stem}_all_runs.csv",
        float_format="%.4f",
        index=False
    )

    df_query_all.to_csv(
        csv_dir / f"benchmark_query_{input_stem}_all_runs.csv",
        float_format="%.4f",
        index=False
    )

    #df_write_stats_flat.insert(0, "run", range(1, len(df_write_stats_flat) + 1))
    df_write_stats_flat.to_csv(
        csv_dir / f"benchmark_write_{input_stem}_mean_std.csv",
        float_format="%.4f",
        index=False
    )
    
    #df_read_stats_flat.insert(0, "run", range(1, len(df_read_stats_flat) + 1))
    df_read_stats_flat.to_csv(
        csv_dir / f"benchmark_read_{input_stem}_mean_std.csv",
        float_format="%.4f",
        index=False
    )

    #df_query_stats_flat.insert(0, "run", range(1, len(df_query_stats_flat) + 1))
    df_query_stats_flat.to_csv(
        csv_dir / f"benchmark_query_{input_stem}_mean_std.csv",
        float_format="%.4f",
        index=False
    )

    print("\nBenchmark completed successfully.")
    print(f"Raw results saved   → *_all_runs.csv")
    print(f"Mean ± Std saved    → *_mean_std.csv")

def main_ml_benchmark(
    input_data,
    formats,
    output_dir,
    ml_benchmark_csv_dir=None,
    repeat=5
):
    output_dir = Path(output_dir)

    if ml_benchmark_csv_dir is None:
        ml_csv_dir = Path.cwd() / "ml_benchmark_results"
    else:
        ml_csv_dir = Path(ml_benchmark_csv_dir)

    ml_csv_dir.mkdir(parents=True, exist_ok=True)

    input_stem = Path(input_data).stem

    all_runs = []   # store RAW results from all repetitions

    print(f"\nRunning ML benchmark {repeat}× for robustness...\n")

    for run_id in range(repeat):
        print(f"\n========== ML BENCHMARK RUN {run_id + 1}/{repeat} ==========")

        for fmt in formats:
            filepath = output_dir / fmt / f"{input_stem}.{fmt}"
            filepath = str(filepath)

            print(f"\n--- Format: {fmt.upper()} ---")

            df, read_res = load_data(filepath, fmt)
            (X_train, X_test, y_train, y_test), prep_res = preprocess(df)

            # Logistic Regression
            log_res = train_model(
                LogisticRegression(max_iter=300),
                X_train, y_train
            )

            # SVM
            svm_res = train_model(
                LinearSVC(),
                X_train, y_train
            )

            # MLP
            mlp_res = train_model(
                MLPClassifier(hidden_layer_sizes=(64,), max_iter=100),
                X_train, y_train
            )

            all_runs.append({
                "run": run_id + 1,
                "format": fmt,

                "read_ms": read_res["read_time_ms"],
                "prep_ms": prep_res["prep_time_ms"],

                # Logistic Regression
                "LR_fit_ms": log_res["fit_time_ms"],
                "LR_peak_mb": log_res["peak_memory_mb"],
                "LR_mem_inc_mb": log_res["memory_increase_mb"],
                "LR_mem_after_mb": log_res["mem_after_mb"],

                # SVM
                "SVM_fit_ms": svm_res["fit_time_ms"],
                "SVM_peak_mb": svm_res["peak_memory_mb"],
                "SVM_mem_inc_mb": svm_res["memory_increase_mb"],
                "SVM_mem_after_mb": svm_res["mem_after_mb"],

                # MLP
                "MLP_fit_ms": mlp_res["fit_time_ms"],
                "MLP_peak_mb": mlp_res["peak_memory_mb"],
                "MLP_mem_inc_mb": mlp_res["memory_increase_mb"],
                "MLP_mem_after_mb": mlp_res["mem_after_mb"],
            })

    # ================= AGGREGATION =================
    df_raw = pd.DataFrame(all_runs)

    df_raw["run"] = df_raw["run"].astype(int)
    df_metrics = df_raw.drop(columns=["run"])

    df_mean = df_metrics.groupby("format").mean(numeric_only=True)
    df_std  = df_metrics.groupby("format").std(numeric_only=True)

    # ================= PRINT TABLES =================
    lr_columns = [
        "read_ms", "prep_ms",
        "LR_fit_ms", "LR_peak_mb", "LR_mem_inc_mb", "LR_mem_after_mb"
    ]

    svm_columns = [
        "read_ms", "prep_ms",
        "SVM_fit_ms", "SVM_peak_mb", "SVM_mem_inc_mb", "SVM_mem_after_mb"
    ]

    mlp_columns = [
        "read_ms", "prep_ms",
        "MLP_fit_ms", "MLP_peak_mb", "MLP_mem_inc_mb", "MLP_mem_after_mb"
    ]

    def print_mean_std(title, cols):
        table = []
        for fmt in df_mean.index:
            row = {"format": fmt}
            for c in cols:
                row[f"{c}_mean"] = round(df_mean.loc[fmt, c], 4)
                row[f"{c}_std"]  = round(df_std.loc[fmt, c], 4)
            table.append(row)

        print("\n" + "=" * 100)
        print(title)
        print("=" * 100)
        print(tabulate(table, headers="keys", tablefmt="fancy_grid"))

    print_mean_std("LOGISTIC REGRESSION — MEAN ± STD", lr_columns)
    print_mean_std("SVM — MEAN ± STD", svm_columns)
    print_mean_std("MLP — MEAN ± STD", mlp_columns)

    # ================= SAVE CSV =================
    df_raw.to_csv(
        ml_csv_dir / f"ml_benchmark_raw_{input_stem}.csv",
        float_format="%.4f",
        index=False
    )

    #df_mean.insert(0, "run", range(1, len(df_mean) + 1))
    df_mean.to_csv(
        ml_csv_dir / f"ml_benchmark_mean_{input_stem}.csv",
        float_format="%.4f",
        index=False
    )

    #df_std.insert(0, "run", range(1, len(df_std) + 1))
    df_std.to_csv(
        ml_csv_dir / f"ml_benchmark_std_{input_stem}.csv",
        float_format="%.4f",
        index=False
    )

    print(f"\n[OK] ML benchmark completed.")
    print(f"Raw results  → ml_benchmark_raw_{input_stem}.csv")
    print(f"Mean results → ml_benchmark_mean_{input_stem}.csv")
    print(f"Std results  → ml_benchmark_std_{input_stem}.csv")

def parse_args():
    parser = argparse.ArgumentParser(description="JSONL Schema Reader & Converter")

    parser.add_argument("--mode", choices=["schema", "convert", "benchmark", "ml_benchmark"], required=True, help="Choose whether to analyze schema or convert data")
    parser.add_argument("--input-data", type=str, required=True, help="Path to input JSONL file")
    parser.add_argument("--sample-lines", type=int, default=10, help="Number of sample lines for schema analysis")
    parser.add_argument("--repeat", type=int, default=5, help="Number of repeated experiments")
    parser.add_argument("--chunk-size", type=int, default=100_000, help="Chunk size for scanning JSONL during conversion")
    parser.add_argument("--num-workers", type=int, default=20, help="Number of workers for conversion")
    parser.add_argument("--formats", nargs="+", default=["csv"], help="Output formats: csv parquet avro feather" )
    parser.add_argument("--output-dir", type=str, default="output", help="Base output directory")
    parser.add_argument("--benchmark-csv-dir", type=str, default=None, help="Directory to save IO benchmark CSV results (default: ./benchmark_results)")
    parser.add_argument("--ml-benchmark-csv-dir", type=str, default=None, help="Directory to save ML benchmark CSV results (default: ./ml_benchmark_results)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "schema":
        main_schema(args.input_data, args.sample_lines)

    elif args.mode == "convert":
        main_converter(
            input_data=args.input_data,
            chunk_size=args.chunk_size,
            num_workers=args.num_workers,
            formats=args.formats,
            output_dir=args.output_dir,
        )
    elif args.mode == "benchmark":
        main_benchmark(
            input_data=args.input_data,
            chunk_size=args.chunk_size,
            num_workers=args.num_workers,
            formats=args.formats,
            output_dir=args.output_dir,
            benchmark_csv_dir=args.benchmark_csv_dir,
            repeat=args.repeat
        )
    elif args.mode == "ml_benchmark":
        main_ml_benchmark(
            input_data=args.input_data,
            formats=args.formats,
            output_dir=args.output_dir,
            ml_benchmark_csv_dir=args.ml_benchmark_csv_dir,
            repeat=args.repeat
        )
