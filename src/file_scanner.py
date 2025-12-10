from pathlib import Path

def scan_jsonl_file(path, chunk_size=100_000):
    """
    Scan the file ONCE:
    - Count total lines
    - Prepare chunk (start_line, end_line) pairs
    """
    path = Path(path)

    print(f"Scanning total lines: {path}")

    # Count total lines fast and memory-safe
    with path.open("r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    # Build chunk ranges
    chunk_ranges = []
    start = 0
    while start < total_lines:
        end = min(start + chunk_size, total_lines)
        chunk_ranges.append((start, end))
        start = end

    print(f"✔ Total lines: {total_lines}")
    print(f"✔ Total chunks: {len(chunk_ranges)}")

    return {
        "path": str(path),
        "total_lines": total_lines,
        "chunk_ranges": chunk_ranges,
    }

