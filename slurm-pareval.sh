#!/bin/bash
#SBATCH --job-name=pareval
#SBATCH --output=pareval_%j.out
#SBATCH --error=pareval_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:b200:1
#SBATCH --container-image=pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel

# ============================================================================
# ParEval Benchmark - Full Evaluation Suite (Slurm + Enroot)
# 
# Runs BOTH generation and translation tasks with all execution models
# (excluding AMD HIP)
#
# Usage:
#   sbatch slurm-pareval.sh <model_name> <hf_token> [num_samples] [inference_config]
#
# Examples:
#   sbatch slurm-pareval.sh deepseek-ai/deepseek-coder-6.7b-instruct hf_xxxxx
#   sbatch slurm-pareval.sh Qwen/Qwen2.5-Coder-7B-Instruct hf_xxxxx 50 chatml
#   sbatch slurm-pareval.sh bigcode/starcoder2-7b hf_xxxxx 50 starcoder
#
# Based on: https://doi.org/10.1145/3625549.3658689
# ============================================================================

set -e

# ============================================================================
# Input Arguments
# ============================================================================
MODEL_NAME="${1:?Error: Please provide model name as first argument}"
HF_TOKEN="${2:?Error: Please provide HuggingFace token as second argument}"
NUM_SAMPLES="${3:-50}"
INFERENCE_CONFIG="${4:-instruct}"

# Execution models to test (excluding AMD HIP)
INCLUDE_MODELS="serial,omp,mpi,mpi+omp,kokkos,cuda"

# Derived variables
MODEL_SHORT=$(echo "${MODEL_NAME}" | sed 's/.*\///' | sed 's/[^a-zA-Z0-9]/-/g')
WORK_DIR="/workspace"
SCRATCH_DIR="/tmp/pareval_${SLURM_JOB_ID}"
OUTPUT_DIR="${WORK_DIR}/results/${MODEL_SHORT}_${SLURM_JOB_ID}"

echo "=========================================="
echo "ParEval Benchmark"
echo "=========================================="
echo "Model:       ${MODEL_NAME}"
echo "Config:      ${INFERENCE_CONFIG}"
echo "Samples:     ${NUM_SAMPLES}"
echo "Job ID:      ${SLURM_JOB_ID}"
echo "Node:        ${SLURMD_NODENAME}"
echo "Start:       $(date)"
echo "=========================================="

# ============================================================================
# Setup Environment
# ============================================================================
echo ""
echo "[Setup] Installing system dependencies..."

apt-get update -qq
apt-get install -y -qq git build-essential cmake libopenmpi-dev openmpi-bin > /dev/null

cd "${WORK_DIR}"

# ============================================================================
# Clone ParEval
# ============================================================================
echo "[Setup] Cloning ParEval repository..."

if [ -d "ParEval" ]; then
    cd ParEval && git pull && cd ..
else
    git clone --recurse-submodules https://github.com/parallelcodefoundry/ParEval.git
fi

cd ParEval
mkdir -p "${OUTPUT_DIR}" "${SCRATCH_DIR}"

# ============================================================================
# Install Python Dependencies
# ============================================================================
echo "[Setup] Installing Python dependencies..."

pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
pip install --quiet --upgrade transformers

# ============================================================================
# Build Kokkos (with CUDA support if available)
# ============================================================================
echo "[Setup] Building Kokkos..."

cd tpl/kokkos
if [ ! -f "build/lib/libkokkoscore.a" ]; then
    mkdir -p build && cd build
    
    # Check if CUDA is available for Kokkos GPU backend
    if command -v nvcc &>/dev/null; then
        echo "[Setup] Building Kokkos with CUDA backend..."
        cmake .. \
            -DCMAKE_INSTALL_PREFIX=. \
            -DKokkos_ENABLE_CUDA=ON \
            -DKokkos_ENABLE_CUDA_LAMBDA=ON \
            -DKokkos_ENABLE_THREADS=ON > /dev/null
    else
        echo "[Setup] Building Kokkos with threads backend only..."
        cmake .. \
            -DCMAKE_INSTALL_PREFIX=. \
            -DKokkos_ENABLE_THREADS=ON > /dev/null
    fi
    
    make install -j${SLURM_CPUS_PER_TASK:-8} > /dev/null 2>&1
    cd ..
fi
cd "${WORK_DIR}/ParEval"

# ============================================================================
# Build C++ Drivers
# ============================================================================
echo "[Setup] Building C++ drivers..."

cd drivers/cpp
make -j${SLURM_CPUS_PER_TASK:-8} > /dev/null 2>&1 || true
cd "${WORK_DIR}/ParEval"

echo "[Setup] Complete!"

# ============================================================================
# Step 1: Generate LLM Outputs (Generation Task)
# ============================================================================
echo ""
echo "=========================================="
echo "[1/5] Generating LLM outputs (generation task)"
echo "=========================================="

python generate/generate.py \
    --prompts prompts/generation-prompts.json \
    --model "${MODEL_NAME}" \
    --output "${OUTPUT_DIR}/generation-outputs.json" \
    --cache "${OUTPUT_DIR}/generation-cache.jsonl" \
    --inference-config "${INFERENCE_CONFIG}" \
    --hf_token "${HF_TOKEN}" \
    --prompted \
    --do_sample \
    --num_samples_per_prompt ${NUM_SAMPLES} \
    --temperature 0.2 \
    --top_p 0.95 \
    --max_new_tokens 1024 \
    --batch_size 16

echo "[1/5] Generation complete: ${OUTPUT_DIR}/generation-outputs.json"

# ============================================================================
# Step 2: Evaluate Generated Code (Generation Task)
# ============================================================================
echo ""
echo "=========================================="
echo "[2/5] Evaluating generated code (generation task)"
echo "=========================================="

cd drivers

echo "Testing execution models: ${INCLUDE_MODELS}"

python run-all.py "${OUTPUT_DIR}/generation-outputs.json" \
    --yes-to-all \
    -o "${OUTPUT_DIR}/generation-results.json" \
    --include-models ${INCLUDE_MODELS} \
    --scratch-dir "${SCRATCH_DIR}" \
    --build-timeout 60 \
    --run-timeout 120

cd "${WORK_DIR}/ParEval"

echo "[2/5] Generation evaluation complete: ${OUTPUT_DIR}/generation-results.json"

# ============================================================================
# Step 3: Generate LLM Outputs (Translation Task)
# ============================================================================
echo ""
echo "=========================================="
echo "[3/5] Generating LLM outputs (translation task)"
echo "=========================================="

python generate/translate.py \
    --prompts prompts/translation-prompts.json \
    --model "${MODEL_NAME}" \
    --output "${OUTPUT_DIR}/translation-outputs.json" \
    --cache "${OUTPUT_DIR}/translation-cache.jsonl" \
    --inference-config "${INFERENCE_CONFIG}" \
    --hf_token "${HF_TOKEN}" \
    --prompted \
    --do_sample \
    --num_samples_per_prompt ${NUM_SAMPLES} \
    --temperature 0.2 \
    --top_p 0.95 \
    --max_new_tokens 1024 \
    --batch_size 16

echo "[3/5] Translation generation complete: ${OUTPUT_DIR}/translation-outputs.json"

# ============================================================================
# Step 4: Evaluate Generated Code (Translation Task)
# ============================================================================
echo ""
echo "=========================================="
echo "[4/5] Evaluating generated code (translation task)"
echo "=========================================="

cd drivers

python run-all.py "${OUTPUT_DIR}/translation-outputs.json" \
    --yes-to-all \
    -o "${OUTPUT_DIR}/translation-results.json" \
    --include-models ${INCLUDE_MODELS} \
    --scratch-dir "${SCRATCH_DIR}" \
    --build-timeout 60 \
    --run-timeout 120

cd "${WORK_DIR}/ParEval"

echo "[4/5] Translation evaluation complete: ${OUTPUT_DIR}/translation-results.json"

# ============================================================================
# Step 5: Compute Metrics for Both Tasks
# ============================================================================
echo ""
echo "=========================================="
echo "[5/5] Computing metrics"
echo "=========================================="

cd analysis

# Generation task metrics
echo "Computing metrics for generation task..."
python create-dataframe.py "${OUTPUT_DIR}/generation-results.json" -o "${OUTPUT_DIR}/generation-results.csv"
python metrics.py "${OUTPUT_DIR}/generation-results.csv" \
    --problem-sizes ../drivers/problem-sizes.json \
    --model-name "${MODEL_SHORT}" \
    -o "${OUTPUT_DIR}/generation-metrics.csv" 2>&1 | tee "${OUTPUT_DIR}/generation-metrics.txt"

# Translation task metrics
echo "Computing metrics for translation task..."
python create-dataframe.py "${OUTPUT_DIR}/translation-results.json" -o "${OUTPUT_DIR}/translation-results.csv"
python metrics.py "${OUTPUT_DIR}/translation-results.csv" \
    --problem-sizes ../drivers/problem-sizes.json \
    --model-name "${MODEL_SHORT}" \
    -o "${OUTPUT_DIR}/translation-metrics.csv" 2>&1 | tee "${OUTPUT_DIR}/translation-metrics.txt"

cd "${WORK_DIR}/ParEval"

echo "[5/5] Metrics complete!"

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "=========================================="
echo "ParEval Benchmark Complete!"
echo "=========================================="
echo "Model:            ${MODEL_NAME}"
echo "Config:           ${INFERENCE_CONFIG}"
echo "Samples/prompt:   ${NUM_SAMPLES}"
echo "Execution models: ${INCLUDE_MODELS}"
echo "Job ID:           ${SLURM_JOB_ID}"
echo ""
echo "Output files in: ${OUTPUT_DIR}/"
ls -la "${OUTPUT_DIR}/"
echo ""
echo "============ GENERATION TASK METRICS ============"
cat "${OUTPUT_DIR}/generation-metrics.txt" | tail -60
echo ""
echo "============ TRANSLATION TASK METRICS ============"
cat "${OUTPUT_DIR}/translation-metrics.txt" | tail -60
echo ""
echo "End time: $(date)"
echo "=========================================="

# Cleanup scratch directory
rm -rf "${SCRATCH_DIR}"
