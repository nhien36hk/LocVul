import pandas as pd

def main():
    print("Reading data/test.csv...")
    df = pd.read_csv("data/test.csv")
    
    # Filter safe samples (target == 0)
    safe_samples = df[df['target'] == 0].sample(n=100, random_state=42)
    
    # Filter vulnerable samples (target == 1) with non-null flaw_line
    vuln_samples = df[(df['target'] == 1) & (df['flaw_line'].notnull())].sample(n=50, random_state=42)
    
    # Combine and shuffle
    smoke_df = pd.concat([safe_samples, vuln_samples]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save as data/dataset.csv
    smoke_df.to_csv("data/dataset.csv", index=False)
    print(f"Created data/dataset.csv with {len(smoke_df)} samples!")
    print(f"Safe samples: {len(smoke_df[smoke_df['target'] == 0])}")
    print(f"Vulnerable samples: {len(smoke_df[smoke_df['target'] == 1])}")

if __name__ == '__main__':
    main()
