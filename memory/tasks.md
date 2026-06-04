# Active Tasks

（无）

# Completed (recent)
- **[task-3] Split evaluation in evaluate_results_excel.py into separate length and correctness metrics functions**
  - Status: ✅ 完成
  - Requested: 2026-06-04 13:40
  - Updated: 2026-06-04 13:41
  - Notes: Split calculation into `evaluate_length_metrics` (for Acc VLC, MAE, RMSE on full and reliable datasets) and `evaluate_correctness_metrics` (for MSP, MSR, MIoU on within_128 dataset only). Updated `load_excel_with_metadata` to load the full dataset and parse metadata.
  - Result: Code updated and verified.
- **[task-2] Analyze dataset filtering logic in evaluate_results_excel.py**
  - Status: ✅ 完成
- **[task-1] Add base CodeT5/CodeT5+ model loading support to evaluate_results_excel.py**
  - Status: ✅ 完成
