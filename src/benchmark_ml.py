import time
import psutil
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow.feather as feather
from fastavro import reader as avro_reader
from sklearn.preprocessing import LabelEncoder
import json
import random
import threading

# ---------- Utility ----------
class MemorySampler:
    def __init__(self, sample_interval=0.01):
        self.sample_interval = sample_interval
        self.process = psutil.Process()
        self.running = False
        self.peak_memory = 0
        self.thread = None

    def _sample(self):
        while self.running:
            mem = self.process.memory_info().rss / (1024 * 1024)  # MB
            if mem > self.peak_memory:
                self.peak_memory = mem
            time.sleep(self.sample_interval)

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._sample)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()

def format_ms(sec):
    return float(f"{sec * 1000:.4f}")

def get_process_mem():
    return psutil.Process().memory_info().rss / (1024 * 1024)


# ---------- Load Functions ----------
def load_data(filepath, fmt):
    t0 = time.time()
    mem_before = get_process_mem()

    if fmt == "csv":
        df = pd.read_csv(filepath)
    elif fmt == "jsonl":
        rows = []
        with open(filepath, "r") as f:
            for line in f:
                rows.append(json.loads(line))
        df = pd.DataFrame(rows)
    elif fmt == "parquet":
        df = pq.read_table(filepath).to_pandas()
    elif fmt == "feather":
        df = feather.read_feather(filepath)
    elif fmt == "avro":
        rows = []
        with open(filepath, "rb") as f:
            for r in avro_reader(f):
                rows.append(r)
        df = pd.DataFrame(rows)
    else:
        raise ValueError("Unknown format")

    mem_after = get_process_mem()
    return df, {
        "read_time_ms": format_ms(time.time() - t0),
        "mem_before_mb": mem_before,
        "mem_after_mb": mem_after,
    }


# ---------- Preprocessing ----------
def preprocess(df):
    t0 = time.time()
    mem_before = get_process_mem()

    # --- Step 1: Convert numeric-like columns ---
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except:
            pass

    # --- Step 2: Replace NaN values ---
    df = df.fillna("MISSING")

    # --- Step 3: Label-encode all non-numerical features ---
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # Now ALL columns are numeric
    numeric_cols = df.columns.tolist()

    # --- Step 4: Select a label column with >= 2 unique classes ---
    candidate_labels = [
        col for col in numeric_cols if df[col].nunique() >= 2
    ]

    if len(candidate_labels) == 0:
        print("⚠️ No valid label column found. Generating synthetic label...")
        df["__synthetic_label__"] = np.random.randint(0, 2, size=len(df))
        label_col = "__synthetic_label__"
    else:
        # Prefer low-cardinality columns (classification-friendly)
        candidate_labels_sorted = sorted(candidate_labels, key=lambda c: df[c].nunique())
        label_col = candidate_labels_sorted[0]
        print(f"Using '{label_col}' as label column (classes={df[label_col].nunique()})")

    y = df[label_col].values
    X = df.drop(columns=[label_col]).values

    # --- Step 5: Discretize label if too many classes ---
    if df[label_col].nunique() > 20:
        print(f"⚠️ Label '{label_col}' has many classes. Binning into 4 categories...")
        y = pd.qcut(y, q=4, labels=False, duplicates="drop")

    # --- Step 6: Ensure X contains no NaN ---
    X = np.nan_to_num(X, nan=0.0)

    # --- Step 7: Scale features ---
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # --- Step 8: Train/test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    mem_after = get_process_mem()

    return (X_train, X_test, y_train, y_test), {
        "prep_time_ms": format_ms(time.time() - t0),
        "mem_before_mb": mem_before,
        "mem_after_mb": mem_after,
        "label_column": label_col,
        "num_features": X_train.shape[1],
        "unique_classes": len(np.unique(y))
    }

# ---------- Train Models ----------
def train_model(model, X_train, y_train):
    mem_before = get_process_mem()
    sampler = MemorySampler(sample_interval=0.01)

    start = time.time()
    sampler.start()

    model.fit(X_train, y_train)

    sampler.stop()
    mem_after = get_process_mem()

    return {
        "fit_time_ms": format_ms(time.time() - start),
        "mem_before_mb": mem_before,
        "mem_after_mb": mem_after,
        "peak_memory_mb": sampler.peak_memory,
        "memory_increase_mb": sampler.peak_memory - mem_before
    }

#def train_model(model, X_train, y_train):
#    t0 = time.time()
#    mem_before = get_process_mem()
#
#    model.fit(X_train, y_train)
#
#    mem_after = get_process_mem()
#    return {
#        "fit_time_ms": format_ms(time.time() - t0),
#        "mem_before_mb": mem_before,
#        "mem_after_mb": mem_after,
#    }
#
