#!/bin/bash
# ------------------------------------------------------------
# run_search.sh
#
# runs structural similarity search on MARS HPC.
# finds lysines in target proteins structurally similar to
# a reference ubiquitination site using gower distance.
#
# requires structure files to already be downloaded.
# if you need to download first, use submit_search.sh instead.
#
# submit this job directly:
#   REFERENCE=Q8IXI2,572 TARGETS=data/raw/test_targets.csv sbatch run_search.sh
#
# or use the helper script (recommended):
#   bash submit_search.sh Q8IXI2,572 data/raw/test_targets.csv
# ------------------------------------------------------------

#SBATCH --job-name=ub_search
#SBATCH --partition=nodes
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-12:00:00
#SBATCH --output=outputs/logs/search_%j.log
#SBATCH --error=outputs/logs/search_%j.err
#SBATCH --account=None

echo "job started  : $(date)"
echo "job id       : $SLURM_JOB_ID"
echo "node         : $SLURMD_NODENAME"
echo "reference    : ${REFERENCE}"
echo "targets      : ${TARGETS}"
echo ""

module load apps/python3
source .venv/bin/activate

echo "python       : $(which python3)"
echo "working dir  : $(pwd)"
echo ""

mkdir -p outputs/logs

# validate required variables were passed
if [ -z "$REFERENCE" ]; then
    echo "error: REFERENCE not set (expected format: Q8IXI2,572)"
    exit 1
fi

if [ -z "$TARGETS" ]; then
    echo "error: TARGETS not set (expected: path to targets csv)"
    exit 1
fi

python predict.py \
    --mode search \
    --reference "$REFERENCE" \
    --targets "$TARGETS" \
    --structures data/structures \
    --verbose

echo ""
echo "job finished : $(date)"