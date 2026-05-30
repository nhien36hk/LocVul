import os
import math
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

def get_line_embeddings(lines, tokenizer, model):
    """
    Get the embeddings for a list of lines using a CodeT5 model.
    
    Args:
    lines (list of str): The lines of code to embed.
    tokenizer: The tokenizer for the CodeT5 model.
    model: The CodeT5 model.
    
    Returns:
    embeddings (torch.Tensor): A tensor containing the embeddings for each line.
    """
    # Tokenize the input lines
    inputs = tokenizer(lines, padding=True, truncation=True, return_tensors="pt")
    
    # Move inputs to the same device as the model
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    # Get the model output
    with torch.no_grad():
        outputs = model.encoder(**inputs)
    
    # Extract the last hidden state
    hidden_states = outputs.last_hidden_state  # Shape: (batch_size, seq_len, hidden_dim)
    
    # To get a single embedding per line, we can mean-pool the hidden states across the sequence dimension
    # or use just the first token's representation, depending on your task.
    # Here, we'll use mean-pooling:
    embeddings = hidden_states.mean(dim=1)  # Shape: (batch_size, hidden_dim)
    
    return embeddings.cpu().numpy()  # Return the embeddings as a NumPy array

def get_most_similar_line(predicted_line, original_lines, tokenizer, model):
    """
    Find the most similar line from original lines based on cosine similarity.
    Exactly matching the logic in seq2seq_eval.py.
    """
    predicted_embedding = get_line_embeddings([predicted_line], tokenizer, model)[0]
    original_embeddings = get_line_embeddings(original_lines, tokenizer, model)

    cosine_similarities = cosine_similarity([predicted_embedding], original_embeddings).flatten()
    most_similar_idx = np.argmax(cosine_similarities)  # Find the index of the most similar line
    
    return original_lines[most_similar_idx]

def get_metrics(gt_vulnerable_lines: set, predicted_vulnerable_lines: set):
    """
    Calculate MSP, MSR, MIoU from ground truth and predicted lines sets.
    Provided by the user.
    """
    intersection = gt_vulnerable_lines.intersection(predicted_vulnerable_lines)
    union = gt_vulnerable_lines.union(predicted_vulnerable_lines)

    msp = len(intersection) / len(predicted_vulnerable_lines) if len(predicted_vulnerable_lines) > 0 else 0
    msr = len(intersection) / len(gt_vulnerable_lines) if len(gt_vulnerable_lines) > 0 else 0
    miou = len(intersection) / len(union) if len(union) > 0 else 0

    return (msp, msr, miou, intersection, union)

def evaluate_excel(file_path, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Excel
    df = pd.read_excel(file_path)
    print(f"Loaded {len(df)} samples from {file_path}")

    # Load Model & Tokenizer for similarity replacement
    print("Loading CodeT5 model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, do_lower_case=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    model.eval()

    sim_msps, sim_msrs, sim_mious = [], [], []
    sim_lengths, gt_lengths = [], []
    similar_predictions = []

    print("Running evaluation and similarity replacement...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        source_code = str(row['Source Code'])
        actual_lines_str = str(row['Actual Vulnerable Lines']) if not pd.isna(row['Actual Vulnerable Lines']) else ""
        pred_lines_str = str(row['Predicted Vulnerable Lines']) if not pd.isna(row['Predicted Vulnerable Lines']) else ""

        # Original lines in the function
        original_lines = source_code.split('\n')
        predicted_lines = pred_lines_str.split('\n') if pred_lines_str else []

        # 1. Similarity Replacement (Exactly matching seq2seq_eval.py loop)
        similar_lines = []
        for j, predicted_line in enumerate(predicted_lines):
            if predicted_line not in original_lines and j < len(predicted_lines) - 1:
                similar_line = get_most_similar_line(predicted_line, original_lines, tokenizer, model)
            else:
                similar_line = predicted_line
            similar_lines.append(similar_line)

        sim_pred_str = '\n'.join(similar_lines)
        similar_predictions.append(sim_pred_str)

        # Convert to sets for metric calculation (ignoring empty lines)
        gt_set = set(line.strip() for line in actual_lines_str.split('\n') if line.strip())
        pred_sim_set = set(line.strip() for line in similar_lines if line.strip())

        # Collect lengths for VLC, MAE, RMSE
        gt_lengths.append(len(gt_set))
        sim_lengths.append(len(pred_sim_set))

        # Compute MSP, MSR, MIoU
        msp_sim, msr_sim, miou_sim, _, _ = get_metrics(gt_set, pred_sim_set)
        sim_msps.append(msp_sim)
        sim_msrs.append(msr_sim)
        sim_mious.append(miou_sim)

    # Save similarity-replaced predictions back to a new Excel file
    df['Predicted Vulnerable Lines (Similarity Replaced)'] = similar_predictions
    output_file = file_path.replace('.xlsx', '_evaluated.xlsx')
    df.to_excel(output_file, index=False)
    print(f"Saved evaluated results with similarity replacement to: {output_file}")

    # Compute overall metrics
    # A. Accuracy VLC: proportion of samples where |P_i| = |G_i|
    acc_vlc_sim = np.mean([1 if s == g else 0 for s, g in zip(sim_lengths, gt_lengths)])

    # B. MAE, RMSE
    mae_sim = mean_absolute_error(gt_lengths, sim_lengths)
    rmse_sim = np.sqrt(mean_squared_error(gt_lengths, sim_lengths))

    # C. Mean MSP, MSR, MIoU
    mean_msp_sim = np.mean(sim_msps)
    mean_msr_sim = np.mean(sim_msrs)
    mean_miou_sim = np.mean(sim_mious)

    # Summary
    print("\n" + "="*50)
    print("EVALUATION METRICS SUMMARY (ON VULNERABLE FUNCTIONS)")
    print("="*50)
    print(f"{'Metric':<25} | {'LocVul (Similarity Replaced)':<30}")
    print("-"*50)
    print(f"{'Accuracy VLC':<25} | {acc_vlc_sim:30.4f}")
    print(f"{'MAE':<25} | {mae_sim:30.4f}")
    print(f"{'RMSE':<25} | {rmse_sim:30.4f}")
    print(f"{'MSP':<25} | {mean_msp_sim:30.4f}")
    print(f"{'MSR':<25} | {mean_msr_sim:30.4f}")
    print(f"{'MIoU':<25} | {mean_miou_sim:30.4f}")
    print("="*50)

if __name__ == "__main__":
    file_path = "test_results.xlsx"
    model_path = "./codet5-base"
    evaluate_excel(file_path, model_path)
