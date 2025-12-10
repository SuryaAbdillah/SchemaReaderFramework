from tabulate import tabulate

def print_table(title, results):
    if len(results) == 0:
        print(f"\n{title}: No data")
        return

    print("\n" + "=" * 80)
    print(f"{title}")
    print("=" * 80)

    # Remove "output_file" column BEFORE printing
    filtered_results = [
        {k: v for k, v in row.items() if k != "output_file"}
        for row in results
    ]

    headers = list(filtered_results[0].keys())
    rows = [list(r.values()) for r in filtered_results]

    print(tabulate(rows, headers=headers, tablefmt="fancy_grid"))

def print_ml_table(title, results, columns):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    filtered = [
        {k: r[k] for k in columns if k in r}
        for r in results
    ]

    print(tabulate(filtered, headers="keys", tablefmt="fancy_grid"))
