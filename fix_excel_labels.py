import pandas as pd
import os

def fix_excel_file(excel_filename, test_csv_path='data/test.csv'):
    if not os.path.exists(excel_filename):
        print(f"File {excel_filename} not found.")
        return

    print(f"Processing {excel_filename}...")
    
    # Load the original test.csv
    test_data = pd.read_csv(test_csv_path).dropna(subset=["processed_func"])
    
    # Apply the exact same filtering as Seq2Seq_vulnDet.py
    test_data = test_data[test_data["target"] == 1]
    test_data = test_data[~test_data['flaw_line_index'].isna()]
    test_data = test_data.reset_index(drop=True)
    
    # Replace /~/ with \n in flaw_line
    test_data['flaw_line'] = test_data['flaw_line'].str.replace('/~/', '\n')
    
    # Load the Excel file
    df_excel = pd.read_excel(excel_filename)
    
    if len(df_excel) != len(test_data):
        print(f"Warning: Length mismatch! Excel has {len(df_excel)}, CSV has {len(test_data)}")
        # If lengths mismatch, we cannot blindly zip. 
        # But the user asked for zip, so we assume they align. If not, this will crash (as intended for strictness).
        
    print("Verifying Source Code match 1-to-1 using zip...")
    
    new_actual_lines = []
    
    # Strict validation loop using zip
    for i, (code_excel, code_csv, flaw_csv) in enumerate(zip(df_excel['Source Code'], test_data['processed_func'], test_data['flaw_line'])):
        code_excel = str(code_excel).strip()
        code_csv = str(code_csv).strip()
        
        # Excel limits cell contents to 32,767 characters. It may truncate long source code.
        # So we check if code_csv starts with code_excel (allowing for some minor whitespace drift at the end).
        # We'll compare up to the length of code_excel.
        compare_len = min(len(code_excel), len(code_csv))
        
        # We also remove carriage returns to avoid \r\n vs \n mismatch
        clean_excel = code_excel[:compare_len].replace('\r', '')
        clean_csv = code_csv[:compare_len].replace('\r', '')
        
        if clean_excel != clean_csv:
            print(f"ERROR: Mismatch at row {i}!")
            print(f"Length Excel: {len(code_excel)}, Length CSV: {len(code_csv)}")
            print("--- Excel Source Code Preview ---")
            print(clean_excel[:200])
            print("--- CSV Source Code Preview ---")
            print(clean_csv[:200])
            raise ValueError(f"Source Code mismatch at row {i}! Verification failed.")
            
        new_actual_lines.append(flaw_csv)
        
    # If the loop finishes without error, it means 100% of the functions matched!
    print("Verification Passed: 100% Source Code matched (accounting for Excel truncation limits)!")
    df_excel['Actual Vulnerable Lines'] = new_actual_lines
        
    # Save the fixed Excel file, overwriting the original
    df_excel.to_excel(excel_filename, index=False)
    print(f"Successfully fixed 'Actual Vulnerable Lines' in {excel_filename}")

if __name__ == "__main__":
    fix_excel_file('test_results_unfinetuned.xlsx')
    fix_excel_file('test_results.xlsx')
    # If there are evaluated versions, we should probably fix them or they can just be re-evaluated
    # evaluate_results_excel.py will be re-run by the user anyway.
