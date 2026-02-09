#!/bin/bash
# ============================================================================
# Launch ParEval tests in parallel - one job per problem type
#
# Usage:
#   ./launch-parallel.sh <outputs_file> [include_models]
#
# Example:
#   ./launch-parallel.sh codellama-CodeLlama-7b-hf-outputs.json "serial,omp,cuda"
#
# This will submit 12 jobs, one for each problem type.
# All results go to: ~/benchmarks/pareval/results/<model>/
#
# Full workflow:
#   1. sbatch slurm-pareval-generate.sh codellama/CodeLlama-7b-hf 20
#   2. (wait for generation to complete)
#   3. ./launch-parallel.sh codellama-CodeLlama-7b-hf-outputs.json "serial,omp,cuda"
#
# ============================================================================

set -e

OUTPUTS_FILE="${1:-}"
INCLUDE_MODELS="${2:-serial,omp,cuda}"

if [ -z "${OUTPUTS_FILE}" ]; then
    echo "Usage: ./launch-parallel.sh <outputs_file> [include_models]"
    echo ""
    echo "Example:"
    echo "  ./launch-parallel.sh codellama-CodeLlama-7b-hf-outputs.json 'serial,omp,cuda'"
    exit 1
fi

PROBLEM_TYPES=(
    "dense_la"
    "fft"
    "geometry"
    "graph"
    "histogram"
    "reduce"
    "scan"
    "search"
    "sort"
    "sparse_la"
    "stencil"
    "transform"
)

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "Launching ParEval parallel jobs"
echo "=========================================="
echo "Outputs file:   ${OUTPUTS_FILE}"
echo "Models:         ${INCLUDE_MODELS}"
echo "Problem types:  ${#PROBLEM_TYPES[@]}"
echo ""

MODEL_SHORT=$(basename "${OUTPUTS_FILE}" .json)
echo "Results will be in: ~/benchmarks/pareval/results/${MODEL_SHORT}/"
echo ""

JOB_IDS=()
for PTYPE in "${PROBLEM_TYPES[@]}"; do
    JOB_ID=$(sbatch --parsable "${SCRIPT_DIR}/slurm-pareval.sh" "${OUTPUTS_FILE}" "${INCLUDE_MODELS}" "${PTYPE}")
    JOB_IDS+=("${JOB_ID}")
    echo "Submitted ${PTYPE}: Job ${JOB_ID}"
done

echo ""
echo "=========================================="
echo "All ${#PROBLEM_TYPES[@]} jobs submitted!"
echo "=========================================="
echo ""
echo "Monitor with: squeue -u \$USER"
echo ""
echo "After all jobs complete, combine results with:"
echo "  cd ~/benchmarks/pareval/results/${MODEL_SHORT}"
echo "  python ${SCRIPT_DIR}/analysis/pareval_report.py . -o report.txt"
echo ""
echo "Job IDs: ${JOB_IDS[*]}"
echo ""
echo "To cancel all jobs:"
echo "  scancel ${JOB_IDS[*]}"