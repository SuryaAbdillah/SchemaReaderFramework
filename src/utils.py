from tabulate import tabulate
import pandas as pd

def print_table(title, results):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    # ----------------------------------------------------------------------
    # CASE 1 — results is a Pandas DataFrame
    # ----------------------------------------------------------------------
    if isinstance(results, pd.DataFrame):

        df = results.copy()

        # Remove unneeded columns
        df = df.drop(columns=["output_file"], errors="ignore")

        # Flatten MultiIndex columns if needed
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join([str(c) for c in col]).strip() for col in df.columns]

        # Reset index so MultiIndex rows don't break tabulate
        df = df.reset_index(drop=False)

        print(tabulate(df, headers=df.columns, tablefmt="fancy_grid"))
        return

    # ----------------------------------------------------------------------
    # CASE 2 — results is a LIST of dicts
    # ----------------------------------------------------------------------
    if isinstance(results, list) and len(results) > 0 and isinstance(results[0], dict):
        filtered = [
            {k: v for k, v in row.items() if k != "output_file"}
            for row in results
        ]
        print(tabulate(filtered, headers="keys", tablefmt="fancy_grid"))
        return

    # ----------------------------------------------------------------------
    # CASE 3 — results is a LIST of tuples/other objects
    # ----------------------------------------------------------------------
    if isinstance(results, list):
        print(tabulate(results, tablefmt="fancy_grid"))
        return

    # ----------------------------------------------------------------------
    # FALLBACK
    # ----------------------------------------------------------------------
    print(results)


#def print_table(title, results):
#    if len(results) == 0:
#        print(f"\n{title}: No data")
#        return
#
#    print("\n" + "=" * 80)
#    print(f"{title}")
#    print("=" * 80)
#
#    # Remove "output_file" column BEFORE printing
#    filtered_results = [
#        {k: v for k, v in row.items() if k != "output_file"}
#        for row in results
#    ]
#
#    headers = list(filtered_results[0].keys())
#    rows = [list(r.values()) for r in filtered_results]
#
#    print(tabulate(rows, headers=headers, tablefmt="fancy_grid"))

def print_ml_table(title, results, columns):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    filtered = [
        {k: r[k] for k in columns if k in r}
        for r in results
    ]

    print(tabulate(filtered, headers="keys", tablefmt="fancy_grid"))
