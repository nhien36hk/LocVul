import json
import os
import pandas as pd
from collections import defaultdict

def normalize_code(code):
    if not code:
        return ""
    return "".join(code.split())

def main():
    # Paths
    gt_path = '/home/nhien36hk/Documents/Intern-InsectLab/Project-XAI/LocVul/data/ground_truth/full_gt.json'
    split_dir = '/home/nhien36hk/Documents/Intern-InsectLab/Project-XAI/LocVul/data/devign_split'
    out_dir = '/home/nhien36hk/Documents/Intern-InsectLab/Project-XAI/LocVul/data'

    splits = {
        'train': os.path.join(split_dir, 'train.json'),
        'val': os.path.join(split_dir, 'val.json'),
        'test': os.path.join(split_dir, 'test.json')
    }

    # Verify input paths exist
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"Ground truth file not found at: {gt_path}")
    for name, path in splits.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Split file for {name} not found at: {path}")

    # 1. Load Ground Truth Database
    print("Loading ground truth database...")
    with open(gt_path, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)
    print(f"Loaded {len(gt_data)} ground truth functions.")

    # 2. Index ground truth functions by normalized prefix and exact matches
    print("Indexing ground truth functions for fast lookup...")
    prefix_len = 150
    gt_prefix_index = defaultdict(list)
    gt_full_normalized_index = defaultdict(list)

    for idx, item in enumerate(gt_data):
        normalized_func = normalize_code(item['func'])
        prefix = normalized_func[:prefix_len]
        gt_prefix_index[prefix].append(idx)
        gt_full_normalized_index[normalized_func].append(idx)

    # 3. Map splits and save CSV files
    mapped_dfs = {}

    for split_name, split_path in splits.items():
        print(f"\nProcessing {split_name} split ({os.path.basename(split_path)})...")
        with open(split_path, 'r', encoding='utf-8') as f:
            split_data = json.load(f)
            
        mapped_records = []
        
        for idx, item in enumerate(split_data):
            func_text = item['func']
            target_label = item['target']
            
            normalized_func = normalize_code(func_text)
            prefix = normalized_func[:prefix_len]
            
            match_idx = None
            
            # Exact Match Strategy
            if normalized_func in gt_full_normalized_index:
                candidates = gt_full_normalized_index[normalized_func]
                if len(candidates) == 1:
                    match_idx = candidates[0]
                else:
                    # Choose candidate matching target label
                    for cand in candidates:
                        gt_target = 1 if gt_data[cand].get('target') is True or gt_data[cand].get('target') == 1 else 0
                        if gt_target == target_label:
                            match_idx = cand
                            break
                    if match_idx is None:
                        match_idx = candidates[0]
                        
            # Prefix Match Strategy (Fallback)
            if match_idx is None and prefix in gt_prefix_index:
                candidates = gt_prefix_index[prefix]
                if len(candidates) == 1:
                    match_idx = candidates[0]
                else:
                    possible = []
                    for cand in candidates:
                        gt_norm = normalize_code(gt_data[cand]['func'])
                        if gt_norm.startswith(normalized_func) or normalized_func.startswith(gt_norm):
                            possible.append(cand)
                    if len(possible) == 1:
                        match_idx = possible[0]
                    elif len(possible) > 1:
                        for cand in possible:
                            gt_target = 1 if gt_data[cand].get('target') is True or gt_data[cand].get('target') == 1 else 0
                            if gt_target == target_label:
                                match_idx = cand
                                break
                        if match_idx is None:
                            match_idx = possible[0]
                    else:
                        match_idx = candidates[0]
                        
            # Extract ground truth properties
            if match_idx is not None:
                gt_item = gt_data[match_idx]
                flaw_line = "/~/".join(gt_item.get('vul_lines', {}).get('code', []))
                flaw_line_index = ",".join(map(str, gt_item.get('vul_lines', {}).get('line_no', [])))
                project = gt_item.get('project', 'unknown')
                metadata = json.dumps(gt_item.get('metadata')) if gt_item.get('metadata') is not None else ""
            else:
                print("No matched ground truth")
                flaw_line = ""
                flaw_line_index = ""
                project = "unknown"
                metadata = ""
                
            mapped_records.append({
                'project': project,
                'target': target_label,
                'processed_func': func_text,
                'flaw_line': flaw_line,
                'flaw_line_index': flaw_line_index,
                'metadata': metadata
            })
            
        df = pd.DataFrame(mapped_records)
        split_csv_path = os.path.join(out_dir, f"{split_name}.csv")
        df.to_csv(split_csv_path, index=True, index_label='index')
        print(f"Successfully saved {split_name} split to: {split_csv_path} ({len(df)} rows)")
        mapped_dfs[split_name] = df

    # 4. Concatenate train, val, test to form dataset.csv
    print("\nRecreating dataset.csv by concatenating train, val, and test splits...")
    dataset_df = pd.concat([mapped_dfs['train'], mapped_dfs['val'], mapped_dfs['test']], ignore_index=True)
    dataset_csv_path = os.path.join(out_dir, 'dataset.csv')
    dataset_df.to_csv(dataset_csv_path, index=True, index_label='index')
    print(f"Successfully saved combined dataset.csv to: {dataset_csv_path} ({len(dataset_df)} rows)")

if __name__ == "__main__":
    main()
