#!/bin/bash
#SBATCH --job-name=pareval-test
#SBATCH --output=pareval-test_%j.out
#SBATCH --error=pareval-test_%j.err
#SBATCH --time=18:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:b200:1

# ============================================================================
# ParEval Benchmark - TEST SCRIPT (Evaluation Only)
#
# Uses pyxis/enroot to run in container with mounted outputs file
# Skips generation phase, only runs evaluation and metrics
# Supports parallel execution by problem type
#
# Usage:
#   sbatch slurm-pareval.sh [outputs_file] [include_models] [problem_type] [cuda_arch]
#
# Examples:
#   # Run all problem types with default A100 architecture
#   sbatch slurm-pareval.sh deepseek-outputs.json "serial,omp,cuda"
#
#   # Run with specific CUDA architecture
#   sbatch slurm-pareval.sh deepseek-outputs.json "serial,omp,cuda" "" "sm_89"  # RTX 4090
#   sbatch slurm-pareval.sh deepseek-outputs.json "serial,omp,cuda" "" "sm_86"  # RTX 3090
#   sbatch slurm-pareval.sh deepseek-outputs.json "serial,omp,cuda" "" "sm_90"  # H100
#   sbatch slurm-pareval.sh deepseek-outputs.json "serial,omp,cuda" "" "sm_100" # B200
#
#   # Run single problem type (for parallel execution)
#   sbatch slurm-pareval.sh deepseek-outputs.json "serial,omp,cuda" geometry "sm_80"
#
# CUDA Architectures:
#   sm_70  - V100
#   sm_75  - T4, RTX 2000 series
#   sm_80  - A100 (default)
#   sm_86  - RTX 3000 series
#   sm_89  - RTX 4000 series, L40
#   sm_90  - H100
#   sm_100 - B200
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
OUTPUTS_FILE="${1:-deepseek-outputs.json}"
INCLUDE_MODELS="${2:-serial,omp,cuda}"
PROBLEM_TYPE="${3:-}"  # Optional: specific problem type to test
CUDA_ARCH="${4:-sm_80}"  # CUDA architecture (default: A100)

MODEL_SHORT=$(basename "${OUTPUTS_FILE}" .json)

# Shared output directory for all jobs (no job ID suffix)
OUTPUT_BASE="${INPUT_DIR}/results/${MODEL_SHORT}"

echo "=========================================="
echo "ParEval Benchmark - TEST MODE"
echo "=========================================="
echo "Container:        ${CONTAINER_IMAGE}"
echo "Input dir:        ${INPUT_DIR}"
echo "Outputs file:     ${OUTPUTS_FILE}"
echo "Include models:   ${INCLUDE_MODELS}"
echo "Problem type:     ${PROBLEM_TYPE:-ALL}"
echo "CUDA arch:        ${CUDA_ARCH}"
echo "Output dir:       ${OUTPUT_BASE}"
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

OUTPUTS_FILE='/workspace/inputs/${OUTPUTS_FILE}'
INCLUDE_MODELS='${INCLUDE_MODELS}'
PROBLEM_TYPE='${PROBLEM_TYPE}'
CUDA_ARCH='${CUDA_ARCH}'
MODEL_SHORT='${MODEL_SHORT}'
WORK_DIR='/workspace'
SCRATCH_DIR='/tmp/pareval_${SLURM_JOB_ID}'
OUTPUT_DIR='/workspace/inputs/results/${MODEL_SHORT}'

mkdir -p \"\${OUTPUT_DIR}\" \"\${SCRATCH_DIR}\"

# ============================================================================
# Setup Environment
# ============================================================================
echo ''
echo '[Setup] Verifying build tools...'

which git g++ make || { echo 'ERROR: Missing required build tools'; exit 1; }
echo '[Setup] Build tools available'

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
# Configure CUDA Architecture
# ============================================================================
echo '[Setup] Configuring CUDA architecture:' \"\${CUDA_ARCH}\"

# Derive compute capability from sm_XX
COMPUTE_CAP=\$(echo \"\${CUDA_ARCH}\" | sed 's/sm_/compute_/')

# Update build-configs.json with the specified architecture
python -c \"
import json
with open('drivers/build-configs.json', 'r') as f:
    config = json.load(f)
config['cuda']['CXXFLAGS'] = '-std=c++17 --generate-code arch=\${COMPUTE_CAP},code=\${CUDA_ARCH} -O3 -Xcompiler \\\"-std=c++17 -O3\\\"'
with open('drivers/build-configs.json', 'w') as f:
    json.dump(config, f, indent=4)
print('Updated build-configs.json with CUDA arch: \${CUDA_ARCH}')
\"

# Also update cpp_driver_wrapper.py
sed -i 's/arch=compute_[0-9]*,code=sm_[0-9]*/arch='\"\${COMPUTE_CAP}\"',code='\"\${CUDA_ARCH}\"'/g' drivers/cpp/cpp_driver_wrapper.py
echo 'Updated cpp_driver_wrapper.py'

# ============================================================================
# Install Python Dependencies
# ============================================================================
echo '[Setup] Installing Python dependencies...'
pip install --quiet tqdm 2>/dev/null || true

# ============================================================================
# Build C++ Drivers
# ============================================================================
echo '[Setup] Building C++ drivers...'

cd drivers/cpp
make -j\${SLURM_CPUS_PER_TASK:-8} > /dev/null 2>&1 || true
cd \"\${WORK_DIR}/ParEval\"

# ============================================================================
# Create local launch config (no srun - we're already inside container)
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
# Verify outputs file exists
# ============================================================================
if [ ! -f \"\${OUTPUTS_FILE}\" ]; then
    echo 'ERROR: Outputs file not found:' \"\${OUTPUTS_FILE}\"
    echo 'Available files in /workspace/inputs:'
    ls -la /workspace/inputs/
    exit 1
fi

echo ''
echo 'Using outputs file:' \"\${OUTPUTS_FILE}\"
PROMPT_COUNT=\$(python -c \"import json; print(len(json.load(open('\${OUTPUTS_FILE}'))))\")
echo \"Found \${PROMPT_COUNT} prompts in outputs file\"

# ============================================================================
# Step 1: Evaluate Generated Code
# ============================================================================
echo ''
echo '=========================================='
echo '[1/2] Evaluating generated code'
echo '=========================================='

cd drivers

echo 'Testing execution models:' \"\${INCLUDE_MODELS}\"
if [ -n \"\${PROBLEM_TYPE}\" ]; then
    echo 'Problem type:' \"\${PROBLEM_TYPE}\"
fi

# Convert comma-separated to space-separated for argparse
MODELS_SPACED=\$(echo \"\${INCLUDE_MODELS}\" | tr ',' ' ')

# Build the command with optional problem type
PROBLEM_TYPE_ARG=\"\"
RESULTS_SUFFIX=\"\"
if [ -n \"\${PROBLEM_TYPE}\" ]; then
    PROBLEM_TYPE_ARG=\"--problem-type \${PROBLEM_TYPE}\"
    RESULTS_SUFFIX=\"_\${PROBLEM_TYPE}\"
fi

python run-all.py \"\${OUTPUTS_FILE}\" \\
    --yes-to-all \\
    -o \"\${OUTPUT_DIR}/results\${RESULTS_SUFFIX}.json\" \\
    --include-models \${MODELS_SPACED} \\
    --launch-configs local-launch-configs.json \\
    --scratch-dir \"\${SCRATCH_DIR}\" \\
    --build-timeout 60 \\
    --run-timeout 120 \\
    \${PROBLEM_TYPE_ARG}

cd \"\${WORK_DIR}/ParEval\"

echo '[1/2] Evaluation complete:' \"\${OUTPUT_DIR}/results\${RESULTS_SUFFIX}.json\"

# ============================================================================
# Step 2: Compute Metrics
# ============================================================================
echo ''
echo '=========================================='
echo '[2/2] Computing metrics'
echo '=========================================='

cd analysis

python create-dataframe.py \"\${OUTPUT_DIR}/results\${RESULTS_SUFFIX}.json\" -o \"\${OUTPUT_DIR}/results\${RESULTS_SUFFIX}.csv\"

python metrics.py \"\${OUTPUT_DIR}/results\${RESULTS_SUFFIX}.csv\" \\
    --problem-sizes ../drivers/problem-sizes.json \\
    --model-name \"\${MODEL_SHORT}\" \\
    -o \"\${OUTPUT_DIR}/metrics\${RESULTS_SUFFIX}.csv\" 2>&1 | tee \"\${OUTPUT_DIR}/metrics\${RESULTS_SUFFIX}.txt\"

cd \"\${WORK_DIR}/ParEval\"

echo '[2/2] Metrics complete!'

# ============================================================================
# Summary
# ============================================================================
echo ''
echo '=========================================='
echo 'ParEval Test Complete!'
echo '=========================================='
echo 'Outputs file:' \"\${OUTPUTS_FILE}\"
echo 'Models tested:' \"\${INCLUDE_MODELS}\"
echo 'Problem type:' \"\${PROBLEM_TYPE:-ALL}\"
echo 'CUDA arch:' \"\${CUDA_ARCH}\"
echo ''
echo 'Results saved to:' \"\${OUTPUT_DIR}/\"
ls -la \"\${OUTPUT_DIR}/\"
echo ''
echo '============ METRICS ============'
cat \"\${OUTPUT_DIR}/metrics\${RESULTS_SUFFIX}.txt\" | tail -60

# Cleanup
rm -rf \"\${SCRATCH_DIR}\"
"

echo ""
echo "End time: $(date)"
echo "=========================================="