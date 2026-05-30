import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple
import pandas as pd
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import configs
from src.utils.io import loads
from src.helper import check_pkl_exists
from src.data.load import load_split_datasets
from transformers import AutoTokenizer


def read_json_array(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def merge_raw_jsons(raw_dir: str) -> pd.DataFrame:
    # Strictly require these three files
    required_files = [
        os.path.join(raw_dir, "train.json"),
        os.path.join(raw_dir, "valid.json"),
        os.path.join(raw_dir, "test.json"),
    ]

    missing = [p for p in required_files if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(
            "Missing required raw JSON files: " + ", ".join(missing)
        )

    candidates: List[str] = required_files

    if not candidates:
        raise FileNotFoundError(
            f"No raw json files found in {raw_dir}. Expected train/validation/test .json files."
        )

    frames: List[pd.DataFrame] = []
    for path in sorted(candidates):
        data = read_json_array(path)
        frames.append(pd.DataFrame(data))

    merged = pd.concat(frames, ignore_index=True)
    # Ensure required columns exist
    if "func" not in merged.columns:
        raise KeyError("Merged raw JSONs must contain a 'func' field")
    return merged


def build_prefix_index(df: pd.DataFrame, prefix_len: int) -> Dict[str, List[int]]:
    index: Dict[str, List[int]] = {}
    funcs = df["func"].astype(str)
    for i, f in enumerate(funcs):
        key = f[:prefix_len]
        index.setdefault(key, []).append(i)
    return index


def map_test_to_ground_truth(
    test_funcs: List[str], raw_df: pd.DataFrame, prefix_len: int
) -> Tuple[List[Optional[dict]], List[Tuple[int, str]]]:
    prefix_index = build_prefix_index(raw_df, prefix_len)
    matches: List[Optional[dict]] = []
    issues: List[Tuple[int, str]] = []  # (idx, reason)

    raw_funcs = raw_df["func"].astype(str).tolist()

    for idx, f in enumerate(test_funcs):
        prefix = (f or "")[:prefix_len]
        cand_idx = prefix_index.get(prefix, [])

        match_obj: Optional[dict] = None
        if len(cand_idx) == 1:
            match_obj = raw_df.iloc[cand_idx[0]].to_dict()
        elif len(cand_idx) > 1:
            # Disambiguate by full startswith on func
            for ci in cand_idx:
                if raw_funcs[ci].startswith(f):
                    match_obj = raw_df.iloc[ci].to_dict()
                    break
            # If still ambiguous, take the first candidate
            if match_obj is None:
                match_obj = raw_df.iloc[cand_idx[0]].to_dict()
                issues.append((idx, "ambiguous_prefix_multiple_matches"))
        else:
            print("Can not match index")
            # No candidate by prefix; try slower scan with startswith
            found = None
            for ci, rfunc in enumerate(raw_funcs):
                if rfunc.startswith(prefix):
                    found = ci
                    break
            if found is not None:
                match_obj = raw_df.iloc[found].to_dict()
                issues.append((idx, "prefix_not_indexed_but_found_by_scan"))
            else:
                match_obj = None
                issues.append((idx, "no_match"))

        matches.append(match_obj)

    return matches, issues


def print_mapping_statistics(matches: List[Optional[dict]], issues: List[Tuple[int, str]], total_tests: int) -> None:
    """In thống kê về mapping: số matches, hàm lỗi/an toàn và tỷ lệ."""
    num_no_match = sum(1 for _, r in issues if r == "no_match")
    num_ambiguous = sum(1 for _, r in issues if r == "ambiguous_prefix_multiple_matches")
    num_matched = sum(1 for m in matches if m is not None)

    print("\n===== Mapping Statistics =====")
    print(f"Total test functions      : {total_tests}")
    print(f"Matched (found ground truth): {num_matched} ({num_matched / total_tests * 100:.2f}%)")
    print(f"No-match                 : {num_no_match}")
    print(f"Ambiguous prefix matches : {num_ambiguous}")

    # Thống kê vulnerable/safe dựa trên field 'target'
    matched_with_target = [m for m in matches if m is not None and "target" in m]
    if matched_with_target:
        vul_count = sum(1 for m in matched_with_target if m.get("target") == 1)
        safe_count = sum(1 for m in matched_with_target if m.get("target") == 0)
        total_labeled = len(matched_with_target)
        vul_ratio = vul_count / total_labeled if total_labeled > 0 else 0.0
        safe_ratio = safe_count / total_labeled if total_labeled > 0 else 0.0

        print("\nLabeled Ground-Truth Stats (by 'target'):")
        print(f"  Vulnerable (target=1): {vul_count:5d} ({vul_ratio * 100:5.2f}%)")
        print(f"  Safe       (target=0): {safe_count:5d} ({safe_ratio * 100:5.2f}%)")
    else:
        print("\n⚠️  Warning: No 'target' field found in matched ground-truth objects.")
    print("=" * 60)


def create_ground_truth_all(
    test_df: pd.DataFrame, raw_df: pd.DataFrame, prefix_len: int, out_path: str
) -> None:
    """Create ground truth for all test functions"""
    # Extract test functions
    test_funcs = test_df["func"].astype(str).tolist()
    print(f"Test size (all): {len(test_funcs)}")

    # Map by prefix
    matches, issues = map_test_to_ground_truth(test_funcs, raw_df, prefix_len)

    # Print mapping statistics
    print_mapping_statistics(matches, issues, len(test_funcs))

    # Save output
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(matches, f, ensure_ascii=False, indent=2)
    print("Saved:", out_path)

    # Also save a simple CSV report of issues (optional)
    if issues:
        report_path = os.path.splitext(out_path)[0] + "_issues.csv"
        pd.DataFrame(issues, columns=["test_index", "reason"]).to_csv(report_path, index=False)
        print("Saved issues report:", report_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Map current split test set to raw ground truth by function prefix")
    parser.add_argument("--input-name", dest="input_name", default="input", help="Name of folder inside PATHS.input containing input PKLs (same as run.py)")
    parser.add_argument("--raw-dir", dest="raw_dir", default="data/raw/devign_gt", help="Directory containing merged raw JSON files (train/validation/test .json)")
    parser.add_argument("--prefix-len", dest="prefix_len", type=int, default=100, help="Prefix length used for startswith matching")
    parser.add_argument("--out", dest="out_path", default="data/ground_truth/test_ground_truth.json", help="Output JSON file with ground-truth objects aligned to test set")

    args = parser.parse_args()

    PATHS = configs.Paths()
    split_dir = PATHS.split
    input_path = os.path.join(PATHS.data, args.input_name)

    # Load full dataset (DataFrame)
    print("Load input pkl file from:", input_path)
    input_dataset_df = loads(input_path)

    # # Load splits
    # if check_pkl_exists(split_dir, 'split_idx.pkl'):
    #     print("Loading existing split indices…")
    #     _, _, test_df = load_split_datasets(split_dir, input_dataset_df)

    # Merge raw JSONs
    raw_df = merge_raw_jsons(args.raw_dir)
    print(f"Merged raw size: {len(raw_df)}")

    create_ground_truth_all(input_dataset_df, raw_df, args.prefix_len, args.out_path)


if __name__ == "__main__":
    main()


