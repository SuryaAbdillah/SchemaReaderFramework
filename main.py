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

def main_schema(input_path, sample_lines):
    schema = analyze_jsonl_schema_fast(input_path, sample_lines=sample_lines)
    print_schema_table_fast(schema)


def main_converter(input_data, chunk_size, num_workers, formats, output_dir):
    scan_info = scan_jsonl_file(input_data, chunk_size=chunk_size)

    for fmt in formats:
        out_dir = f"{output_dir}/{fmt}"
        convert_jsonl(scan_info, output_dir=out_dir, fmt=fmt, num_workers=num_workers)

def main_benchmark(input_data, chunk_size, num_workers, formats, output_dir):
    #jsonl_path = input_data
    scan_info = scan_jsonl_file(input_data, chunk_size=chunk_size)

    results_write = []
    results_read = []
    results_query = []

    for fmt in formats:
        fmt_output_dir = f"{output_dir}/{fmt}"

        # do conversion as before

        write_res = benchmark_write(
            format_name=fmt,
            convert_func=convert_jsonl,
            scan_info=scan_info,
            output_dir=fmt_output_dir,
            fmt=fmt,
            num_workers=num_workers
        )
        results_write.append(write_res)

        # ---- READ BENCHMARK ----
        read_res = benchmark_read(write_res["output_file"], fmt)
        results_read.append(read_res)

        # ---- QUERY BENCHMARK ----
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
        results_query.append({"format": fmt, **query_res})

    # Save results
    print_table("WRITE BENCHMARK RESULTS", results_write)
    print_table("READ BENCHMARK RESULTS", results_read)
    print_table("QUERY BENCHMARK RESULTS", results_query) 

    input_stem = Path(input_data).stem
    pd.DataFrame(results_write).to_csv(f"benchmark_write_{input_stem}.csv", index=False)
    pd.DataFrame(results_read).to_csv(f"benchmark_read_{input_stem}.csv", index=False)
    pd.DataFrame(results_query).to_csv(f"benchmark_query_{input_stem}.csv", index=False)

    print("Benchmark completed. Results saved to CSV.")

def main_ml_benchmark(input_data, formats, output_dir):
    #formats = ["jsonl", "csv", "parquet", "feather", "avro"]
    input_stem = Path(input_data).stem

    results = []

    for fmt in formats:
        filepath = f"output/{fmt}/{input_stem}.{fmt}"

        print(f"\n=== ML BENCHMARK: {fmt.upper()} ===")

        df, read_res = load_data(filepath, fmt)
        (X_train, X_test, y_train, y_test), prep_res = preprocess(df)

        # Logistic regression
        log_res = train_model(LogisticRegression(max_iter=300), X_train, y_train)

        # Linear SVM
        svm_res = train_model(LinearSVC(), X_train, y_train)

        # MLP neural network
        mlp_res = train_model(MLPClassifier(hidden_layer_sizes=(64,), max_iter=100), X_train, y_train)

        results.append({
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

    print("\n\n=== FINAL ML BENCHMARK RESULTS ===")
    # Columns for each model
    lr_columns = [
        "format", "read_ms", "prep_ms",
        "LR_fit_ms", "LR_peak_mb", "LR_mem_inc_mb", "LR_mem_after_mb"
    ]
    
    svm_columns = [
        "format", "read_ms", "prep_ms",
        "SVM_fit_ms", "SVM_peak_mb", "SVM_mem_inc_mb", "SVM_mem_after_mb"
    ]
    
    mlp_columns = [
        "format", "read_ms", "prep_ms",
        "MLP_fit_ms", "MLP_peak_mb", "MLP_mem_inc_mb", "MLP_mem_after_mb"
    ]
    
    # Print tables
    print_ml_table("LOGISTIC REGRESSION ML BENCHMARK", results, lr_columns)
    print_ml_table("SVM ML BENCHMARK", results, svm_columns)
    print_ml_table("MLP NEURAL NETWORK ML BENCHMARK", results, mlp_columns) 

def parse_args():
    parser = argparse.ArgumentParser(description="JSONL Schema Reader & Converter")

    parser.add_argument("--mode", choices=["schema", "convert", "benchmark", "ml_benchmark"], required=True, help="Choose whether to analyze schema or convert data")
    parser.add_argument("--input-data", type=str, required=True, help="Path to input JSONL file")
    parser.add_argument("--sample-lines", type=int, default=10, help="Number of sample lines for schema analysis")
    parser.add_argument("--chunk-size", type=int, default=100_000, help="Chunk size for scanning JSONL during conversion")
    parser.add_argument("--num-workers", type=int, default=20, help="Number of workers for conversion")
    parser.add_argument("--formats", nargs="+", default=["csv"], help="Output formats: csv parquet avro feather" )
    parser.add_argument("--output-dir", type=str, default="output", help="Base output directory")
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
        )
    elif args.mode == "ml_benchmark":
        main_ml_benchmark(
            input_data=args.input_data,
            formats=args.formats,
            output_dir=args.output_dir,
        )
