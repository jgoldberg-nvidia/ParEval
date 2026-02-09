#!/bin/bash
#SBATCH --job-name=pareval-full
#SBATCH --output=pareval-full_%j.out
#SBATCH --error=pareval-full_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

# ============================================================================
# ParEval Benchmark - FULL SCRIPT (Generation + Evaluation)
#
# Runs code generation with a HuggingFace model, then evaluates the outputs.
#
# Usage:
#   sbatch slurm-pareval-full.sh <model_name> [num_samples] [problem_type]
#
# Examples:
#   # CodeLlama 7B with 20 samples per prompt
#   sbatch slurm-pareval-full.sh codellama/CodeLlama-7b-hf 20
#
#   # DeepSeek Coder with default 50 samples
#   sbatch slurm-pareval-full.sh deepseek-ai/deepseek-coder-6.7b-base
#
#   # Only run geometry problems
#   sbatch slurm-pareval-full.sh codellama/CodeLlama-7b-hf 20 geometry
#
# Problem types: dense_la, fft, geometry, graph, histogram, reduce,
#                scan, search, sort, sparse_la, stencil, transform
#
# ============================================================================

set -e

# ============================================================================
# Configuration
# ============================================================================
CONTAINER_IMAGE="nvcr.io#nvidia/pytorch:25.12-py3"
INPUT_DIR="${HOME}/benchmarks/pareval"
MODEL_NAME="${1:-codellama/CodeLlama-7b-hf}"
NUM_SAMPLES="${2:-50}"  # Paper default: 50 samples for pass@1,5,10,20
PROBLEM_TYPE="${3:-}"

# Create a clean model name for file naming (replace / with -)
MODEL_SHORT=$(echo "${MODEL_NAME}" | sed 's|/|-|g' | sed 's|\.|-|g')

# Output directory
OUTPUT_DIR="${INPUT_DIR}/results/${MODEL_SHORT}"

echo "=========================================="
echo "ParEval Benchmark - FULL MODE"
echo "=========================================="
echo "Container:        ${CONTAINER_IMAGE}"
echo "Model:            ${MODEL_NAME}"
echo "Model short name: ${MODEL_SHORT}"
echo "Samples/prompt:   ${NUM_SAMPLES}"
echo "Problem type:     ${PROBLEM_TYPE:-ALL}"
echo "Output dir:       ${OUTPUT_DIR}"
echo "Job ID:           ${SLURM_JOB_ID}"
echo "Node:             ${SLURMD_NODENAME}"
echo "Start:            $(date)"
echo "=========================================="

# ============================================================================
# Run everything inside container via srun + enroot
# ============================================================================
srun --container-image="${CONTAINER_IMAGE}" \
     --container-mounts="${INPUT_DIR}:/workspace/inputs" \
     bash -c "
set -e

MODEL_NAME='${MODEL_NAME}'
MODEL_SHORT='${MODEL_SHORT}'
NUM_SAMPLES='${NUM_SAMPLES}'
PROBLEM_TYPE='${PROBLEM_TYPE}'
WORK_DIR='/workspace'
SCRATCH_DIR='/tmp/pareval_${SLURM_JOB_ID}'
OUTPUT_DIR='/workspace/inputs/results/${MODEL_SHORT}'
OUTPUTS_FILE=\"\${OUTPUT_DIR}/${MODEL_SHORT}-outputs.json\"

mkdir -p \"\${OUTPUT_DIR}\" \"\${SCRATCH_DIR}\"

# ============================================================================
# Set HuggingFace cache to scratch directory (more space than home)
# ============================================================================
export HF_HOME=\"\${SCRATCH_DIR}/hf_cache\"
export TRANSFORMERS_CACHE=\"\${SCRATCH_DIR}/hf_cache\"
export HF_DATASETS_CACHE=\"\${SCRATCH_DIR}/hf_cache/datasets\"
mkdir -p \"\${HF_HOME}\"
echo \"HuggingFace cache: \${HF_HOME}\"

# ============================================================================
# Setup Environment
# ============================================================================
echo ''
echo '[Setup] Verifying tools...'
which git g++ make python || { echo 'ERROR: Missing required tools'; exit 1; }
nvidia-smi || echo 'Warning: nvidia-smi not found'

cd \"\${WORK_DIR}\"

# ============================================================================
# Clone ParEval
# ============================================================================
echo '[Setup] Cloning ParEval repository...'
if [ -d 'ParEval' ]; then
    cd ParEval && git pull --quiet && cd ..
else
    git clone --recurse-submodules https://github.com/jgoldberg-nvidia/ParEval.git
fi
cd ParEval

# ============================================================================
# Install Python Dependencies
# ============================================================================
echo '[Setup] Installing Python dependencies...'
pip install --quiet tqdm transformers accelerate sentencepiece 2>/dev/null || true

# ============================================================================
# Build C++ Drivers
# ============================================================================
echo '[Setup] Building C++ drivers...'
cd drivers/cpp
make -j\${SLURM_CPUS_PER_TASK:-8} > /dev/null 2>&1 || true
cd \"\${WORK_DIR}/ParEval\"

# ============================================================================
# Create local launch config
# ============================================================================
cat > drivers/local-launch-configs.json << 'LAUNCH_EOF'
{
    \"serial\": {
        \"format\": \"{exec_path} {args}\",
        \"params\": [{}]
    },
    \"omp\": {
        \"format\": \"{exec_path} {args} {num_threads}\",
        \"params\": [
            {\"num_threads\": 1},
            {\"num_threads\": 2},
            {\"num_threads\": 4},
            {\"num_threads\": 8}
        ]
    },
    \"cuda\": {
        \"format\": \"{exec_path} {args}\",
        \"params\": [{}]
    }
}
LAUNCH_EOF

echo '[Setup] Complete!'

# ============================================================================
# Detect inference config based on model name
# ============================================================================
INFERENCE_CONFIG=''
if [[ \"\${MODEL_NAME}\" == *'codellama'* ]] || [[ \"\${MODEL_NAME}\" == *'CodeLlama'* ]]; then
    INFERENCE_CONFIG='codellama'
elif [[ \"\${MODEL_NAME}\" == *'deepseek'* ]]; then
    INFERENCE_CONFIG='deepseek'
elif [[ \"\${MODEL_NAME}\" == *'starcoder'* ]] || [[ \"\${MODEL_NAME}\" == *'StarCoder'* ]]; then
    INFERENCE_CONFIG='starcoder'
elif [[ \"\${MODEL_NAME}\" == *'phind'* ]] || [[ \"\${MODEL_NAME}\" == *'Phind'* ]]; then
    INFERENCE_CONFIG='phind'
elif [[ \"\${MODEL_NAME}\" == *'magicoder'* ]]; then
    INFERENCE_CONFIG='magicoder'
elif [[ \"\${MODEL_NAME}\" == *'qwen'* ]] || [[ \"\${MODEL_NAME}\" == *'Qwen'* ]]; then
    INFERENCE_CONFIG='qwen'
else
    INFERENCE_CONFIG='codellama'  # default fallback
fi
echo \"Using inference config: \${INFERENCE_CONFIG}\"

# ============================================================================
# Step 1: Generate Code
# ============================================================================
echo ''
echo '=========================================='
echo '[1/3] Generating code with model'
echo '=========================================='
echo \"Model: \${MODEL_NAME}\"
echo \"Samples per prompt: \${NUM_SAMPLES}\"

cd generate

python generate.py \\
    --prompts ../prompts/generation-prompts.json \\
    --model \"\${MODEL_NAME}\" \\
    --output \"\${OUTPUTS_FILE}\" \\
    --num_samples_per_prompt \${NUM_SAMPLES} \\
    --temperature 0.2 \\
    --top_p 0.95 \\
    --do_sample \\
    --batch_size 8 \\
    --inference-config \"\${INFERENCE_CONFIG}\" \\
    --cache \"\${SCRATCH_DIR}/cache.jsonl\"

cd \"\${WORK_DIR}/ParEval\"

echo '[1/3] Generation complete!'
echo \"Outputs saved to: \${OUTPUTS_FILE}\"
PROMPT_COUNT=\$(python -c \"import json; print(len(json.load(open('\${OUTPUTS_FILE}'))))\")
echo \"Generated \${PROMPT_COUNT} prompts\"

# ============================================================================
# Step 2: Evaluate Generated Code
# ============================================================================
echo ''
echo '=========================================='
echo '[2/3] Evaluating generated code'
echo '=========================================='

cd drivers

INCLUDE_MODELS='serial omp cuda'
PROBLEM_TYPE_ARG=''
RESULTS_SUFFIX=''

if [ -n \"\${PROBLEM_TYPE}\" ]; then
    PROBLEM_TYPE_ARG=\"--problem-type \${PROBLEM_TYPE}\"
    RESULTS_SUFFIX=\"_\${PROBLEM_TYPE}\"
    echo \"Filtering to problem type: \${PROBLEM_TYPE}\"
fi

python run-all.py \"\${OUTPUTS_FILE}\" \\
    --yes-to-all \\
    -o \"\${OUTPUT_DIR}/results\${RESULTS_SUFFIX}.json\" \\
    --include-models \${INCLUDE_MODELS} \\
    --launch-configs local-launch-configs.json \\
    --scratch-dir \"\${SCRATCH_DIR}\" \\
    --build-timeout 60 \\
    --run-timeout 120 \\
    \${PROBLEM_TYPE_ARG}

cd \"\${WORK_DIR}/ParEval\"
echo '[2/3] Evaluation complete!'

# ============================================================================
# Step 3: Compute Metrics
# ============================================================================
echo ''
echo '=========================================='
echo '[3/3] Computing metrics'
echo '=========================================='

cd analysis

python create-dataframe.py \"\${OUTPUT_DIR}/results\${RESULTS_SUFFIX}.json\" \\
    -o \"\${OUTPUT_DIR}/results\${RESULTS_SUFFIX}.csv\"

python metrics.py \"\${OUTPUT_DIR}/results\${RESULTS_SUFFIX}.csv\" \\
    --problem-sizes ../drivers/problem-sizes.json \\
    --model-name \"\${MODEL_SHORT}\" \\
    -o \"\${OUTPUT_DIR}/metrics\${RESULTS_SUFFIX}.csv\" 2>&1 | tee \"\${OUTPUT_DIR}/metrics\${RESULTS_SUFFIX}.txt\"

# Generate full report
if [ -f 'pareval_report.py' ]; then
    python pareval_report.py \"\${OUTPUT_DIR}\" -o \"\${OUTPUT_DIR}/report.txt\" || true
fi

cd \"\${WORK_DIR}/ParEval\"
echo '[3/3] Metrics complete!'

# ============================================================================
# Summary
# ============================================================================
echo ''
echo '=========================================='
echo 'ParEval FULL Benchmark Complete!'
echo '=========================================='
echo \"Model: \${MODEL_NAME}\"
echo \"Samples per prompt: \${NUM_SAMPLES}\"
echo \"Problem type: \${PROBLEM_TYPE:-ALL}\"
echo ''
echo 'Results saved to:' \"\${OUTPUT_DIR}/\"
ls -la \"\${OUTPUT_DIR}/\"
echo ''

# Show quick metrics summary
if [ -f \"\${OUTPUT_DIR}/metrics\${RESULTS_SUFFIX}.csv\" ]; then
    echo '============ QUICK METRICS ============'
    python -c \"
import pandas as pd
df = pd.read_csv('\${OUTPUT_DIR}/metrics\${RESULTS_SUFFIX}.csv')
print('pass@1 by execution model:')
for em in df['execution model'].unique():
    rate = df[df['execution model'] == em]['pass@1'].mean()
    print(f'  {em}: {rate:.1%}')
\"
fi

# Cleanup
rm -rf \"\${SCRATCH_DIR}\"
"

echo ""
echo "End time: $(date)"
echo "=========================================="
