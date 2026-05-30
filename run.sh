#!/bin/bash

# Default parameters
EPOCHS=10
# Set SMOKE to empty for full dataset (omit the flag)
SMOKE=""
# Larger batch size suitable for a 25 GB GPU (adjust as needed)
BATCH_SIZE=16
SEED=9

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --epochs) EPOCHS="$2"; shift ;;
        --smoke) SMOKE="$2"; shift ;;
        --batch_size) BATCH_SIZE="$2"; shift ;;
        --seed) SEED="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

echo "================================================================="
echo "LocVul Smoke Test Pipeline"
echo "Epochs: $EPOCHS"
echo "Smoke samples: $SMOKE"
echo "Batch size: $BATCH_SIZE"
echo "Seed index: $SEED"
echo "================================================================="

# Set memory allocation configuration to prevent PyTorch NVML assert issues on MIG/restricted GPUs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Step 1: Run CodeBERT training and self-attention line localization
echo ""
echo ">>> [STAGE 1] Running CodeBERT Pipeline (vulnDet_pipeline.py)..."
python vulnDet_pipeline.py \
    --seed=$SEED \
    --FINE_TUNE="yes" \
    --model_variation="microsoft/codebert-base" \
    --checkpoint_dir="./checkpoints" \
    --sampling="no" \
    --REMOVE_MISSING_LINE_LABELS="yes" \
    --EXPLAINER="ATTENTION" \
    --EXPLAIN_ONLY_TP="no" \
    --sort_by_lines="yes" \
    --epochs=$EPOCHS \
    ${SMOKE:+--smoke=$SMOKE} \
    --batch_size=$BATCH_SIZE

# Step 2: Run CodeT5 line-level seq2seq training
echo ""
echo ">>> [STAGE 2] Running CodeT5 Seq2Seq Pipeline (Seq2Seq_vulnDet.py)..."
python Seq2Seq_vulnDet.py \
    --seed=$SEED \
    --FINE_TUNE="yes" \
    --model_variation="Salesforce/codet5-base" \
    --checkpoint_dir="./checkpoints_seq2seq" \
    --epochs=$EPOCHS \
    ${SMOKE:+--smoke=$SMOKE} \
    --batch_size=$BATCH_SIZE

echo ""
echo ">>> LocVul Smoke Test Pipeline execution completed!"
