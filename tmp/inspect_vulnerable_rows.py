import pandas as pd

def inspect_vuln():
    file_path = 'data/test.csv'
    # Read entire test set (it is only 1.03 GB, we can read specific columns to save memory)
    cols = ['index', 'target', 'processed_func', 'flaw_line', 'flaw_line_index']
    df = pd.read_csv(file_path, usecols=cols)
    
    vuln_df = df[df['target'] == 1]
    print(f"Total vulnerable rows (target == 1): {len(vuln_df)}")
    
    # Check how many have non-null flaw_line and flaw_line_index
    non_null_flaw = vuln_df[vuln_df['flaw_line'].notnull()]
    print(f"Vulnerable rows with non-null flaw_line: {len(non_null_flaw)}")
    
    # Display the first 3 rows of non-null flaw_line
    print("\n--- Examples of Vulnerable functions with flaw lines ---")
    for idx, row in non_null_flaw.head(3).iterrows():
        print(f"\n=== Vulnerable Row Index {row['index']} ===")
        print(f"[flaw_line]: {row['flaw_line']}")
        print(f"[flaw_line_index]: {row['flaw_line_index']}")
        print("[processed_func]:")
        lines = row['processed_func'].split('\n')
        # Print first 15 lines of the function
        for i, line in enumerate(lines[:15]):
            print(f"{i:3d}: {line}")
        if len(lines) > 15:
            print("...")

if __name__ == '__main__':
    inspect_vuln()
