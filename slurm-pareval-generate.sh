#!/bin/bash
#SBATCH --job-name=pareval-gen
#SBATCH --output=pareval-gen_%j.out
#SBATCH --error=pareval-gen_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:b200:1

# ============================================================================
# ParEval Benchmark - GENERATION ONLY
#
# Generates code outputs using a HuggingFace model.
# After this completes, run launch-parallel.sh to evaluate in parallel.
#
# Usage:
#   sbatch slurm-pareval-generate.sh <model_name> [num_samples]
#
# Examples:
#   sbatch slurm-pareval-generate.sh codellama/CodeLlama-7b-hf 20
#   sbatch slurm-pareval-generate.sh deepseek-ai/deepseek-coder-6.7b-base 50
#
# After completion, run:
#   ./launch-parallel.sh <model-outputs.json> "serial,omp,cuda"
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

# Create a clean model name for file naming
MODEL_SHORT=$(echo "${MODEL_NAME}" | sed 's|/|-|g' | sed 's|\.|-|g')
OUTPUTS_FILE="${MODEL_SHORT}-outputs.json"

echo "=========================================="
echo "ParEval Benchmark - GENERATION MODE"
echo "=========================================="
echo "Container:        ${CONTAINER_IMAGE}"
echo "Model:            ${MODEL_NAME}"
echo "Model short name: ${MODEL_SHORT}"
echo "Samples/prompt:   ${NUM_SAMPLES}"
echo "Output file:      ${INPUT_DIR}/${OUTPUTS_FILE}"
echo "Job ID:           ${SLURM_JOB_ID}"
echo "Node:             ${SLURMD_NODENAME}"
echo "Start:            $(date)"
echo "=========================================="

# ============================================================================
# Run generation inside container
# ============================================================================
srun --container-image="${CONTAINER_IMAGE}" \
     --container-mounts="${INPUT_DIR}:/workspace/inputs" \
     bash -c "
set -e

MODEL_NAME='${MODEL_NAME}'
MODEL_SHORT='${MODEL_SHORT}'
NUM_SAMPLES='${NUM_SAMPLES}'
OUTPUTS_FILE='${OUTPUTS_FILE}'
WORK_DIR='/workspace'
SCRATCH_DIR='/tmp/pareval_gen_${SLURM_JOB_ID}'

mkdir -p \"\${SCRATCH_DIR}\"

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
which git python || { echo 'ERROR: Missing required tools'; exit 1; }
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
# Generate Code
# ============================================================================
echo ''
echo '=========================================='
echo 'Generating code with model'
echo '=========================================='
echo \"Model: \${MODEL_NAME}\"
echo \"Samples per prompt: \${NUM_SAMPLES}\"
echo \"Output: /workspace/inputs/\${OUTPUTS_FILE}\"

cd generate

python generate.py \\
    --prompts ../prompts/generation-prompts.json \\
    --model \"\${MODEL_NAME}\" \\
    --output \"/workspace/inputs/\${OUTPUTS_FILE}\" \\
    --num_samples_per_prompt \${NUM_SAMPLES} \\
    --temperature 0.2 \\
    --top_p 0.95 \\
    --do_sample \\
    --batch_size 8 \\
    --inference-config \"\${INFERENCE_CONFIG}\" \\
    --cache \"\${SCRATCH_DIR}/cache.jsonl\"

cd \"\${WORK_DIR}/ParEval\"

# ============================================================================
# Summary
# ============================================================================
echo ''
echo '=========================================='
echo 'Generation Complete!'
echo '=========================================='
PROMPT_COUNT=\$(python -c \"import json; print(len(json.load(open('/workspace/inputs/\${OUTPUTS_FILE}'))))\")
echo \"Generated \${PROMPT_COUNT} prompts\"
echo \"Output file: /workspace/inputs/\${OUTPUTS_FILE}\"
echo ''
echo 'Next step - run parallel evaluation:'
echo \"  ./launch-parallel.sh \${OUTPUTS_FILE} 'serial,omp,cuda'\"
echo ''

# Cleanup
rm -rf \"\${SCRATCH_DIR}\"
"

echo ""
echo "=========================================="
echo "Generation Complete!"
echo "=========================================="
echo "Output file: ${INPUT_DIR}/${OUTPUTS_FILE}"
echo ""
echo "Next step - launch parallel evaluation:"
echo "  cd $(dirname "$0")"
echo "  ./launch-parallel.sh ${OUTPUTS_FILE} 'serial,omp,cuda'"
echo ""
echo "End time: $(date)"
echo "=========================================="
