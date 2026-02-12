#!/bin/bash
#SBATCH --job-name=pareval-gen-fast
#SBATCH --output=pareval-gen-fast_%j.out
#SBATCH --error=pareval-gen-fast_%j.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:b200:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G

# ============================================================================
# ParEval Benchmark - FAST GENERATION (vLLM + 8 GPUs)
#
# Uses vLLM with tensor parallelism across 8 GPUs for ~5-10x speedup.
#
# Usage:
#   sbatch slurm-pareval-generate-fast.sh <model_name> [num_samples]
#
# Examples:
#   sbatch slurm-pareval-generate-fast.sh codellama/CodeLlama-7b-hf
#   sbatch slurm-pareval-generate-fast.sh codellama/CodeLlama-34b-hf 50
#   sbatch slurm-pareval-generate-fast.sh deepseek-ai/deepseek-coder-6.7b-base
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
echo "ParEval Benchmark - FAST GENERATION (vLLM)"
echo "=========================================="
echo "Container:        ${CONTAINER_IMAGE}"
echo "Model:            ${MODEL_NAME}"
echo "Model short name: ${MODEL_SHORT}"
echo "Samples/prompt:   ${NUM_SAMPLES}"
echo "Output file:      ${INPUT_DIR}/${OUTPUTS_FILE}"
echo "GPUs requested:   8 (tensor parallel)"
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
# Set ALL cache directories to scratch (more space than home)
# ============================================================================
export HF_HOME=\"\${SCRATCH_DIR}/hf_cache\"
export TRANSFORMERS_CACHE=\"\${SCRATCH_DIR}/hf_cache\"
export HF_DATASETS_CACHE=\"\${SCRATCH_DIR}/hf_cache/datasets\"
export XDG_CACHE_HOME=\"\${SCRATCH_DIR}/cache\"
export VLLM_CACHE_ROOT=\"\${SCRATCH_DIR}/vllm_cache\"
export HOME_CACHE=\"\${SCRATCH_DIR}/cache\"
mkdir -p \"\${HF_HOME}\" \"\${XDG_CACHE_HOME}\" \"\${VLLM_CACHE_ROOT}\" \"\${XDG_CACHE_HOME}/vllm\"
# Create symlink to redirect ~/.cache to scratch
rm -rf ~/.cache 2>/dev/null || true
ln -sf \"\${XDG_CACHE_HOME}\" ~/.cache 2>/dev/null || true
echo \"All caches redirected to: \${SCRATCH_DIR}\"

# ============================================================================
# Setup Environment
# ============================================================================
echo ''
echo '[Setup] Verifying tools...'
which git python || { echo 'ERROR: Missing required tools'; exit 1; }

echo ''
echo '[Setup] GPU Configuration:'
nvidia-smi --query-gpu=index,name,memory.total --format=csv
NUM_GPUS=\$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo \"Total GPUs available: \${NUM_GPUS}\"

cd \"\${WORK_DIR}\"

# ============================================================================
# Clone ParEval (fresh clone every time)
# ============================================================================
echo ''
echo '[Setup] Cloning ParEval repository (fresh)...'
rm -rf ParEval
git clone --recurse-submodules https://jgoldberg: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@github.com/jgoldberg-nvidia/ParEval.git
cd ParEval

# ============================================================================
# Install Python Dependencies (including vLLM)
# ============================================================================
echo ''
echo '[Setup] Installing Python dependencies...'
pip install tqdm transformers accelerate sentencepiece

echo ''
echo '[Setup] Removing incompatible flash_attn from container...'
pip uninstall -y flash_attn 2>/dev/null || true

echo ''
echo '[Setup] Installing vLLM (this may take 5-10 minutes)...'
pip install vllm

echo '[Setup] vLLM installed successfully!'

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
elif [[ \"\${MODEL_NAME}\" == *'Nemotron'* ]]; then
    INFERENCE_CONFIG='chatml'
elif [[ \"\${MODEL_NAME}\" == *'CodeQwen'* ]] && [[ \"\${MODEL_NAME}\" == *'Chat'* ]]; then
    INFERENCE_CONFIG='chatml'
elif [[ \"\${MODEL_NAME}\" == *'qwen'* ]] || [[ \"\${MODEL_NAME}\" == *'Qwen'* ]]; then
    INFERENCE_CONFIG='qwen'
else
    INFERENCE_CONFIG='codellama'  # default fallback
fi
echo \"Using inference config: \${INFERENCE_CONFIG}\"

# ============================================================================
# Generate Code with vLLM (FAST!)
# ============================================================================
echo ''
echo '=========================================='
echo 'Generating code with vLLM (tensor parallel)'
echo '=========================================='
echo \"Model: \${MODEL_NAME}\"
echo \"Samples per prompt: \${NUM_SAMPLES}\"
echo \"Tensor parallel GPUs: \${NUM_GPUS}\"
echo \"Output: /workspace/inputs/\${OUTPUTS_FILE}\"

cd generate

# vLLM automatically uses all available GPUs via tensor_parallel_size=torch.cuda.device_count()
python generate-vllm.py \\
    --prompts ../prompts/generation-prompts.json \\
    --model \"\${MODEL_NAME}\" \\
    --output \"/workspace/inputs/\${OUTPUTS_FILE}\" \\
    --num_samples_per_prompt \${NUM_SAMPLES} \\
    --temperature 0.2 \\
    --top_p 0.95 \\
    --do_sample \\
    --cache \"\${SCRATCH_DIR}/cache.jsonl\"

cd \"\${WORK_DIR}/ParEval\"

# ============================================================================
# Summary
# ============================================================================
echo ''
echo '=========================================='
echo 'Fast Generation Complete!'
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
echo "Fast Generation Complete!"
echo "=========================================="
echo "Output file: ${INPUT_DIR}/${OUTPUTS_FILE}"
echo ""
echo "Next step - launch parallel evaluation:"
echo "  cd $(dirname "$0")"
echo "  ./launch-parallel.sh ${OUTPUTS_FILE} 'serial,omp,cuda'"
echo ""
echo "End time: $(date)"
echo "=========================================="