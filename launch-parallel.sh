#!/bin/bash
# ============================================================================
# Launch ParEval tests in parallel - one job per problem type
#
# Usage:
#   ./launch-parallel.sh [outputs_file] [include_models]
#
# Example:
#   ./launch-parallel.sh deepseek-outputs.json "serial,omp,cuda"
#
# This will submit 12 jobs, one for each problem type.
# All results go to: ~/benchmarks/pareval/results/<model>/
# ============================================================================

OUTPUTS_FILE="${1:-deepseek-outputs.json}"
INCLUDE_MODELS="${2:-serial,omp,cuda}"

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
    JOB_ID=$(sbatch --parsable slurm-pareval-test.sh "${OUTPUTS_FILE}" "${INCLUDE_MODELS}" "${PTYPE}")
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
echo "After all jobs complete, merge results with:"
echo "  cd ~/benchmarks/pareval/results/${MODEL_SHORT}"
echo "  # Results are already split by problem type"
echo "  # View individual metrics: cat metrics_geometry.txt"
echo ""
echo "Job IDs: ${JOB_IDS[*]}"
