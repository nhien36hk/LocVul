#!/bin/bash
#SBATCH --job-name=Vul_LocVul
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --gres=mps:a100:5
#SBATCH --mem=44G
#SBATCH --time=72:00:00

# Thông số VRAM cần thiết (Tăng lên 40GB cho model 7B)
REQUIRED_VRAM=25000  

# =========================================================
# CHUẨN BỊ MÔI TRƯỜNG
# =========================================================
mkdir -p logs
module clear -f
module load shared python312

# Đường dẫn đến Python trong môi trường bạn vừa cài
export PYTHON_BIN="/datastore/inseclab/HuuNhien/envs/Finetune/bin/python"
export HF_HOME=/datastore/inseclab/rapp/cache/huggingface
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# --- GỌI HELPER TÌM GPU THEO TÀI LIỆU LAB ---
unset CUDA_VISIBLE_DEVICES
echo "🔍 Đang tìm GPU trống ít nhất ${REQUIRED_VRAM}MB..."
CHECK_OUT=$(/usr/local/bin/gpu_check.sh $REQUIRED_VRAM $SLURM_JOB_ID)
EXIT_CODE=$?

if [ $EXIT_CODE -eq 10 ]; then
    echo "Hệ thống đang bận, Job sẽ tự động đợi trong hàng đợi..."
    exit 0 
elif [ $EXIT_CODE -eq 11 ]; then
    echo "Lỗi: $CHECK_OUT"
    exit 1 
fi

BEST_GPU=$CHECK_OUT
export CUDA_VISIBLE_DEVICES=$BEST_GPU
echo "✅ Đã chọn GPU: $BEST_GPU cho Job $SLURM_JOB_ID"

# Thiết lập MPS
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-job$SLURM_JOB_ID
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-job$SLURM_JOB_ID
rm -rf $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY


# Fixed Ground-truth
echo "Running Fixed Ground-truth"
$PYTHON_BIN fix_excel_labels.py

echo ""
echo "Running Evualation Base Weights."
$PYTHON_BIN evaluate_results_excel.py --load_base_codet5p

echo ""
echo "Running Evualation Finetune Weights."
$PYTHON_BIN evaluate_results_excel.py 

echo ""
echo "LocVul Pipeline execution completed!"

# Dọn dẹp MPS sau khi xong
rm -rf $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
echo "✅ [ALL COMPLETED] Hoàn tất toàn bộ pipeline lúc: $(date)"
