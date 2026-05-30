import pandas as pd
import json

def inspect():
    file_path = 'data/test.csv'
    print("--- Reading first 2 rows of test.csv ---")
    df_preview = pd.read_csv(file_path, nrows=2)
    print("Columns:", list(df_preview.columns))
    
    for idx, row in df_preview.iterrows():
        print(f"\n=== Row {idx} ===")
        for col in df_preview.columns:
            val = row[col]
            # Truncate long code blocks for display
            if isinstance(val, str) and len(val) > 200:
                display_val = val[:200] + "... [TRUNCATED]"
            else:
                display_val = val
            print(f"[{col}]: {display_val}")
            
    print("\n--- Counting rows and analyzing target distribution in chunks ---")
    chunk_size = 10000
    total_rows = 0
    target_counts = {0: 0, 1: 0}
    null_counts = {}
    
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        total_rows += len(chunk)
        if 'target' in chunk.columns:
            for val, count in chunk['target'].value_counts().items():
                target_counts[val] = target_counts.get(val, 0) + count
        for col in chunk.columns:
            null_counts[col] = null_counts.get(col, 0) + chunk[col].isnull().sum()
            
    print(f"Total Rows: {total_rows}")
    print("Target distribution (0: Safe, 1: Vulnerable):", target_counts)
    print("Null value counts per column:", null_counts)

if __name__ == '__main__':
    inspect()
