import os
import json
import ast
import math
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

def get_line_embeddings(lines, tokenizer, model):
    """
    Get the embeddings for a list of lines using a CodeT5 model.
    Exactly matching the logic in seq2seq_eval.py.
    """
    inputs = tokenizer(lines, padding=True, truncation=True, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model.encoder(**inputs)
    hidden_states = outputs.last_hidden_state  # Shape: (batch_size, seq_len, hidden_dim)
    embeddings = hidden_states.mean(dim=1)  # Mean-pooling across tokens
    return embeddings.cpu().numpy()

def get_most_similar_line(predicted_line, original_lines, tokenizer, model):
    """
    Find the most similar line from original lines based on cosine similarity.
    Exactly matching the logic in seq2seq_eval.py, with CUDA OOM fallback.
    """
    device = next(model.parameters()).device
    try:
        predicted_embedding = get_line_embeddings([predicted_line], tokenizer, model)[0]
        original_embeddings = get_line_embeddings(original_lines, tokenizer, model)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if isinstance(e, RuntimeError) and "out of memory" not in str(e).lower():
            raise e
        # Graceful fallback to CPU for this sample
        model.to("cpu")
        torch.cuda.empty_cache()
        try:
            predicted_embedding = get_line_embeddings([predicted_line], tokenizer, model)[0]
            original_embeddings = get_line_embeddings(original_lines, tokenizer, model)
        finally:
            model.to(device)
            
    cosine_similarities = cosine_similarity([predicted_embedding], original_embeddings).flatten()
    most_similar_idx = np.argmax(cosine_similarities)
    return original_lines[most_similar_idx]

def get_metrics(gt_vulnerable_lines: set, predicted_vulnerable_lines: set):
    """
    Calculate MSP, MSR, MIoU from ground truth and predicted lines sets.
    """
    intersection = gt_vulnerable_lines.intersection(predicted_vulnerable_lines)
    union = gt_vulnerable_lines.union(predicted_vulnerable_lines)

    msp = len(intersection) / len(predicted_vulnerable_lines) if len(predicted_vulnerable_lines) > 0 else 0
    msr = len(intersection) / len(gt_vulnerable_lines) if len(gt_vulnerable_lines) > 0 else 0
    miou = len(intersection) / len(union) if len(union) > 0 else 0

    return msp, msr, miou

def compute_group_metrics(gt_lens, sim_lens, msps, msrs, mious):
    """
    Compute Acc_vlc, MAE, RMSE, and mean MSP, MSR, MIoU for a group of samples.
    """
    if not gt_lens:
        return {
            "count": 0,
            "accuracy_vlc": 0.0,
            "mae": 0.0,
            "rmse": 0.0,
            "msp": 0.0,
            "msr": 0.0,
            "miou": 0.0
        }
    acc_vlc = np.mean([1 if s == g else 0 for s, g in zip(sim_lens, gt_lens)])
    mae = mean_absolute_error(gt_lens, sim_lens)
    rmse = np.sqrt(mean_squared_error(gt_lens, sim_lens))
    msp = np.mean(msps)
    msr = np.mean(msrs)
    miou = np.mean(mious)
    
    return {
        "count": len(gt_lens),
        "accuracy_vlc": float(acc_vlc),
        "mae": float(mae),
        "rmse": float(rmse),
        "msp": float(msp),
        "msr": float(msr),
        "miou": float(miou)
    }

def load_and_filter_data(file_path):
    print(f"Loading Excel from {file_path}...")
    df = pd.read_excel(file_path)
    if 'metadata' in df.columns:
        filtered_indices = []
        for idx, row in df.iterrows():
            meta = row['metadata']
            if pd.isna(meta):
                print(f"Sample {idx} has no metadata")
                continue
            try:
                if isinstance(meta, str):
                    try:
                        meta_dict = json.loads(meta)
                    except json.JSONDecodeError:
                        meta_dict = ast.literal_eval(meta)
                else:
                    meta_dict = meta
                
                if isinstance(meta_dict, dict):
                    codet5_meta = meta_dict.get("codet5", {})
                    is_within = codet5_meta.get("vul_line_within_128")
                    if is_within is True or str(is_within).lower() == 'true':
                        filtered_indices.append(idx)
            except Exception as e:
                pass
        
        df = df.loc[filtered_indices].reset_index(drop=True)
        print(f"Filtered down to {len(df)} samples with vul_line_within_128 == True")
    else:
        print("Warning: No 'metadata' column found, proceeding with all samples.")
    return df

def evaluate_excel(df, file_path, model_path_t5, checkpoint_t5, model_path_bert, checkpoint_bert, json_output_path="evaluation_metrics.json", load_base_codet5p=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Evaluating {len(df)} samples...")

    # Step 1: Load CodeBERT and predict Giai doan 1 labels (TP vs FN)
    print("Loading CodeBERT classifier and predicting (TP vs FN)...")
    tokenizer_bert = AutoTokenizer.from_pretrained(model_path_bert, do_lower_case=True)
    model_bert = AutoModelForSequenceClassification.from_pretrained(model_path_bert, num_labels=2)
    model_bert.resize_token_embeddings(len(tokenizer_bert))
    
    checkpoint = torch.load(checkpoint_bert, map_location=device)
    model_bert.load_state_dict(checkpoint['model'])
    model_bert.to(device)
    model_bert.eval()

    preds_g1 = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Classifier Prediction"):
        source_code = str(row['Source Code'])
        # Tokenize and run CodeBERT inference
        inputs = tokenizer_bert(source_code, truncation=True, padding=True, max_length=512, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model_bert(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
        pred = np.argmax(probs)
        preds_g1.append(pred)

    # Release CodeBERT from VRAM
    del model_bert
    torch.cuda.empty_cache()
    print("CodeBERT released from memory.")

    # Step 2: Load CodeT5 model and perform similarity replacement
    print("Loading CodeT5 model and tokenizer...")
    tokenizer_t5 = AutoTokenizer.from_pretrained(model_path_t5, do_lower_case=True)
    model_t5 = AutoModelForSeq2SeqLM.from_pretrained(model_path_t5)
    
    if load_base_codet5p:
        print("Running with original non-finetuned base model weights from Hugging Face (no checkpoint loaded).")
    elif checkpoint_t5 and os.path.exists(checkpoint_t5):
        print(f"Loading CodeT5 fine-tuned checkpoint from {checkpoint_t5}...")
        checkpoint = torch.load(checkpoint_t5, map_location=device)
        model_t5.load_state_dict(checkpoint['model'])
    else:
        print(f"Warning: Checkpoint {checkpoint_t5} not found. Running with base model weights.")
        
    model_t5.to(device)
    model_t5.eval()

    similar_predictions = []
    
    # Store individual metrics for grouping
    results_records = []

    print("Running similarity replacement and line metrics...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Similarity Replacement"):
        source_code = str(row['Source Code'])
        actual_lines_str = str(row['Actual Vulnerable Lines']) if not pd.isna(row['Actual Vulnerable Lines']) else ""
        pred_lines_str = str(row['Predicted Vulnerable Lines']) if not pd.isna(row['Predicted Vulnerable Lines']) else ""
        pred_g1 = preds_g1[idx]

        # Original lines in the function
        original_lines = source_code.split('\n')
        predicted_lines = [line for line in (pred_lines_str.split('\n') if pred_lines_str else []) if line.strip()]

        # 1. Similarity Replacement
        similar_lines = []
        for j, predicted_line in enumerate(predicted_lines):
            if predicted_line not in original_lines:
                similar_line = get_most_similar_line(predicted_line, original_lines, tokenizer_t5, model_t5)
            else:
                similar_line = predicted_line
            similar_lines.append(similar_line)

        sim_pred_str = '\n'.join(similar_lines)
        similar_predictions.append(sim_pred_str)

        # Convert to sets for metric calculation (ignoring empty lines)
        gt_set = set(line.strip() for line in actual_lines_str.split('\n') if line.strip())
        pred_sim_set = set(line.strip() for line in similar_lines if line.strip())

        # Compute MSP, MSR, MIoU
        msp, msr, miou = get_metrics(gt_set, pred_sim_set)

        results_records.append({
            "gt_len": len(gt_set),
            "sim_len": len(pred_sim_set),
            "msp": msp,
            "msr": msr,
            "miou": miou,
            "pred_g1": pred_g1  # 1 = True Positive, 0 = False Negative
        })

    # Save similarity-replaced predictions back to a new Excel file
    df['Predicted Vulnerable Lines (Similarity Replaced)'] = similar_predictions
    df['CodeBERT Prediction'] = preds_g1
    output_file = file_path.replace('.xlsx', '_evaluated.xlsx')
    df.to_excel(output_file, index=False)
    print(f"Saved evaluated results to: {output_file}")

    # Step 3: Compute split metrics
    splits_summary = {}

    for dataset_name, data_filter in [
        ("full_dataset", lambda r: True),
        ("reliable_set", lambda r: r["gt_len"] <= 20)
    ]:
        dataset_records = [r for r in results_records if data_filter(r)]
        
        # Overall
        gt_overall = [r["gt_len"] for r in dataset_records]
        sim_overall = [r["sim_len"] for r in dataset_records]
        msp_overall = [r["msp"] for r in dataset_records]
        msr_overall = [r["msr"] for r in dataset_records]
        miou_overall = [r["miou"] for r in dataset_records]
        
        # True Positive (pred_g1 == 1)
        tp_records = [r for r in dataset_records if r["pred_g1"] == 1]
        gt_tp = [r["gt_len"] for r in tp_records]
        sim_tp = [r["sim_len"] for r in tp_records]
        msp_tp = [r["msp"] for r in tp_records]
        msr_tp = [r["msr"] for r in tp_records]
        miou_tp = [r["miou"] for r in tp_records]

        # False Negative (pred_g1 == 0)
        fn_records = [r for r in dataset_records if r["pred_g1"] == 0]
        gt_fn = [r["gt_len"] for r in fn_records]
        sim_fn = [r["sim_len"] for r in fn_records]
        msp_fn = [r["msp"] for r in fn_records]
        msr_fn = [r["msr"] for r in fn_records]
        miou_fn = [r["miou"] for r in fn_records]

        splits_summary[dataset_name] = {
            "overall": compute_group_metrics(gt_overall, sim_overall, msp_overall, msr_overall, miou_overall),
            "true_positive": compute_group_metrics(gt_tp, sim_tp, msp_tp, msr_tp, miou_tp),
            "false_negative": compute_group_metrics(gt_fn, sim_fn, msp_fn, msr_fn, miou_fn)
        }

    # Save to JSON
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(splits_summary, f, indent=4, ensure_ascii=False)
    print(f"Saved evaluation metrics to: {json_output_path}")

    # Print Summary Tables
    for d_name in ["full_dataset", "reliable_set"]:
        title = "FULL DATASET EVALUATION" if d_name == "full_dataset" else "RELIABLE SET (GT <= 20) EVALUATION"
        d_summary = splits_summary[d_name]
        print("\n" + "="*70)
        print(f" {title}")
        print("="*70)
        print(f"{'Metric':<15} | {'Overall':<15} | {'True Positive (TP)':<18} | {'False Negative (FN)':<18}")
        print("-"*70)
        print(f"{'Count':<15} | {d_summary['overall']['count']:15d} | {d_summary['true_positive']['count']:18d} | {d_summary['false_negative']['count']:18d}")
        print(f"{'Accuracy VLC':<15} | {d_summary['overall']['accuracy_vlc']:15.4f} | {d_summary['true_positive']['accuracy_vlc']:18.4f} | {d_summary['false_negative']['accuracy_vlc']:18.4f}")
        print(f"{'MAE':<15} | {d_summary['overall']['mae']:15.4f} | {d_summary['true_positive']['mae']:18.4f} | {d_summary['false_negative']['mae']:18.4f}")
        print(f"{'RMSE':<15} | {d_summary['overall']['rmse']:15.4f} | {d_summary['true_positive']['rmse']:18.4f} | {d_summary['false_negative']['rmse']:18.4f}")
        print(f"{'MSP':<15} | {d_summary['overall']['msp']:15.4f} | {d_summary['true_positive']['msp']:18.4f} | {d_summary['false_negative']['msp']:18.4f}")
        print(f"{'MSR':<15} | {d_summary['overall']['msr']:15.4f} | {d_summary['true_positive']['msr']:18.4f} | {d_summary['false_negative']['msr']:18.4f}")
        print(f"{'MIoU':<15} | {d_summary['overall']['miou']:15.4f} | {d_summary['true_positive']['miou']:18.4f} | {d_summary['false_negative']['miou']:18.4f}")
        print("="*70)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", default="test_results.xlsx", type=str)
    parser.add_argument("--model_path_t5", default="Salesforce/codet5-base", type=str)
    parser.add_argument("--checkpoint_t5", default="./checkpoints_seq2seq/best_weights.pt", type=str)
    parser.add_argument("--model_path_bert", default="microsoft/codebert-base", type=str)
    parser.add_argument("--checkpoint_bert", default="checkpoints/best_weights.pt", type=str)
    parser.add_argument("--output_json", default="evaluation_metrics.json", type=str)
    parser.add_argument("--load_base_codet5p", action="store_true", help="Load base CodeT5/CodeT5+ model from Hugging Face without loading checkpoint")
    args = parser.parse_args()

    df_filtered = load_and_filter_data(args.file_path)

    evaluate_excel(
        df=df_filtered,
        file_path=args.file_path,
        model_path_t5=args.model_path_t5,
        checkpoint_t5=args.checkpoint_t5,
        model_path_bert=args.model_path_bert,
        checkpoint_bert=args.checkpoint_bert,
        json_output_path=args.output_json,
        load_base_codet5p=args.load_base_codet5p
    )
