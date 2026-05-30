#!/usr/bin/env python3
"""
Script để thêm metadata về tokens và vul_lines vào ground truth JSON.
Thay vì tạo nhiều file riêng biệt, script này thêm key 'metadata' vào mỗi record
với thông tin cho cả BERT family và CodeT5 family để có thể filter linh hoạt sau này.
"""

import json
import os
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer

def calculate_index_128(func_text, tokenizer, max_tokens=128):
    lines = func_text.split('\n')
    total_tokens = 0
    last_valid_line = 0
    num_special_tokens = 2
    
    # Tính tokens available sau khi trừ special tokens
    available_tokens = max_tokens - num_special_tokens
    
    for i, line in enumerate(lines):
        # Thêm newline vào cuối dòng (trừ dòng cuối cùng)
        # vì khi encode toàn bộ function, newline giữa các dòng được tokenize
        if i < len(lines) - 1:
            line_with_newline = line + '\n'
        else:
            line_with_newline = line
        
        # Đếm tokens cho dòng hiện tại (bao gồm newline nếu có, KHÔNG có special tokens)
        line_tokens = len(tokenizer.tokenize(line_with_newline))
        
        # Kiểm tra nếu thêm dòng này có vượt quá giới hạn không
        # (đã trừ đi special tokens)
        if total_tokens + line_tokens <= available_tokens:
            total_tokens += line_tokens
            last_valid_line = i + 1 
        else:
            # Vượt quá giới hạn, dừng lại
            break
    
    return last_valid_line


def count_total_tokens(func_text, tokenizer):
    tokens = tokenizer.encode(func_text, add_special_tokens=True)
    return len(tokens)


def add_metadata_to_ground_truth(input_file, output_file):
    """
    Thêm metadata về tokens và vul_lines vào ground truth JSON cho cả BERT và CodeT5
    
    Args:
        input_file: Đường dẫn file ground truth input
        output_file: Đường dẫn file ground truth output
    """
    print(f"Loading data from {input_file}...")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} records")
    
    # Load cả 2 tokenizers: BERT family và CodeT5 family
    print("Loading tokenizers...")
    print("  - Loading BERT tokenizer (GraphCodeBERT)...")
    tokenizer_bert = AutoTokenizer.from_pretrained('microsoft/graphcodebert-base')
    print("  - Loading CodeT5 tokenizer (CodeT5p-220m)...")
    tokenizer_codet5 = AutoTokenizer.from_pretrained('Salesforce/codet5p-220m')
    
    # Statistics cho cả 2 families
    records_with_vul = 0
    records_bert_under_128 = 0
    records_bert_vul_within_128 = 0
    records_bert_vul_beyond_128 = 0
    records_codet5_under_128 = 0
    records_codet5_vul_within_128 = 0
    records_codet5_vul_beyond_128 = 0
    
    print("Processing records...")
    for record in tqdm(data, desc="Adding metadata"):
        # Lấy function text và vul_lines
        func_text = record.get('func', '')
        vul_lines = record.get('vul_lines', {})
        vul_line_nos = vul_lines.get('line_no', [])
        
        # Chỉ xử lý những record có vul_lines không rỗng
        if vul_line_nos:
            records_with_vul += 1
            
            # === BERT Family ===
            # 1. Đếm tổng tokens của function với BERT tokenizer
            total_tokens_bert = count_total_tokens(func_text, tokenizer_bert)
            under_128_tokens_bert = total_tokens_bert <= 128
            
            if under_128_tokens_bert:
                records_bert_under_128 += 1
            
            # 2. Tính index_128 với BERT tokenizer
            index_128_bert = calculate_index_128(func_text, tokenizer_bert)
            
            # 3. Kiểm tra xem tất cả vul_lines có nằm trong 128 tokens không (BERT)
            vul_line_within_128_bert = all(line_no <= index_128_bert for line_no in vul_line_nos)
            
            if vul_line_within_128_bert:
                records_bert_vul_within_128 += 1
            else:
                records_bert_vul_beyond_128 += 1
            
            # === CodeT5 Family ===
            # 1. Đếm tổng tokens của function với CodeT5 tokenizer
            total_tokens_codet5 = count_total_tokens(func_text, tokenizer_codet5)
            under_128_tokens_codet5 = total_tokens_codet5 <= 128
            
            if under_128_tokens_codet5:
                records_codet5_under_128 += 1
            
            # 2. Tính index_128 với CodeT5 tokenizer
            index_128_codet5 = calculate_index_128(func_text, tokenizer_codet5)
            
            # 3. Kiểm tra xem tất cả vul_lines có nằm trong 128 tokens không (CodeT5)
            vul_line_within_128_codet5 = all(line_no <= index_128_codet5 for line_no in vul_line_nos)
            
            if vul_line_within_128_codet5:
                records_codet5_vul_within_128 += 1
            else:
                records_codet5_vul_beyond_128 += 1
            
            # 4. Thêm key 'metadata' với thông tin cho cả 2 families
            record['metadata'] = {
                'bert': {
                    'under_128_tokens': under_128_tokens_bert,
                    'vul_line_within_128': vul_line_within_128_bert,
                    'index_128': index_128_bert,  # Thông tin bổ sung để debug
                    'total_tokens': total_tokens_bert  # Thông tin bổ sung để debug
                },
                'codet5': {
                    'under_128_tokens': under_128_tokens_codet5,
                    'vul_line_within_128': vul_line_within_128_codet5,
                    'index_128': index_128_codet5,  # Thông tin bổ sung để debug
                    'total_tokens': total_tokens_codet5  # Thông tin bổ sung để debug
                }
            }
        else:
            # Nếu không có vul_lines, set metadata = None để tránh lỗi khi lưu columns
            record['metadata'] = None
    
    # Save output
    print(f"\nSaving results to {output_file}...")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY:")
    print(f"Total records: {len(data)}")
    print(f"Records with vul_lines (có metadata dict): {records_with_vul}")
    print(f"Records without vul_lines (metadata = None): {len(data) - records_with_vul}")
    print(f"")
    print("Trong số records có vul_lines - BERT Family:")
    print(f"  - under_128_tokens=True: {records_bert_under_128}")
    print(f"  - under_128_tokens=False: {records_with_vul - records_bert_under_128}")
    print(f"  - vul_line_within_128=True: {records_bert_vul_within_128}")
    print(f"  - vul_line_within_128=False: {records_bert_vul_beyond_128}")
    print(f"")
    print("Trong số records có vul_lines - CodeT5 Family:")
    print(f"  - under_128_tokens=True: {records_codet5_under_128}")
    print(f"  - under_128_tokens=False: {records_with_vul - records_codet5_under_128}")
    print(f"  - vul_line_within_128=True: {records_codet5_vul_within_128}")
    print(f"  - vul_line_within_128=False: {records_codet5_vul_beyond_128}")
    print(f"")
    print(f"Output saved to: {output_file}")
    print(f"{'='*60}")
    
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description='Thêm metadata về tokens và vul_lines vào ground truth JSON'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='workspace/train_ground_truth.json',
        help='Input ground truth JSON file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='workspace/train_ground_truth_with_metadata.json',
        help='Output ground truth JSON file với metadata'
    )
    
    args = parser.parse_args()
    
    # Kiểm tra file input có tồn tại không
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found!")
        return 1

    try:
        add_metadata_to_ground_truth(args.input, args.output)
        print(f"\n✓ Done!")
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

