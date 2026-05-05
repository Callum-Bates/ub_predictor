#!/bin/bash
# ------------------------------------------------------------
# run_pipeline.sh
#
# stage 2 of the ub_predictor pipeline on MARS HPC.
# runs feature generation and model training or prediction.
# downloads must be complete before running this script.
#
# this is a compute-heavy job - needs multiple cpus and
# substantial memory for parsing cif files and training.
#
# submit this job:
#   sbatch run_pipeline.sh
#
# or submit both jobs with automatic chaining:
#   bash submit_job.sh data/raw/sites.csv train
# ------------------------------------------------------------

#SBATCH --job-name=ub_pipeline
#SBATCH --partition=nodes
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-12:00:00
#SBATCH --output=outputs/logs/pipeline_%j.log
#SBATCH --error=outputs/logs/pipeline_%j.err
#SBATCH --account=None

echo "job started  : $(date)"
echo "job id       : $SLURM_JOB_ID"
echo "node         : $SLURMD_NODENAME"
echo "mode         : ${MODE:-train}"
echo "input file   : ${INPUT_FILE:-data/raw/sites.csv}"
echo ""

module load apps/python3
source .venv/bin/activate

echo "python       : $(which python3)"
echo "working dir  : $(pwd)"
echo ""

mkdir -p outputs/logs

python predict.py \
    --input "${INPUT_FILE:-data/raw/sites.csv}" \
    --mode "${MODE:-train}" \
    --structures data/structures \
    --verbose

echo ""
echo "job finished : $(date)"
