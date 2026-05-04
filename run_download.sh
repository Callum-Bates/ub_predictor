#!/bin/bash
# ------------------------------------------------------------
# run_download.sh
#
# stage 1 of the ub_predictor pipeline on MARS HPC.
# downloads alphafold cif and pae structure files for all
# proteins in the input dataset.
#
# this is a lightweight job - mostly waiting on network
# responses. minimal cpu and memory needed.
#
# submit this job:
#   sbatch run_download.sh
#
# or submit both jobs with automatic chaining:
#   bash submit_job.sh data/raw/sites.csv train
# ------------------------------------------------------------

#SBATCH --job-name=ub_download
#SBATCH --partition=nodes
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0-06:00:00
#SBATCH --output=outputs/logs/download_%j.log
#SBATCH --error=outputs/logs/download_%j.err

# print job information - useful for debugging later
echo "job started  : $(date)"
echo "job id       : $SLURM_JOB_ID"
echo "node         : $SLURMD_NODENAME"
echo "input file   : ${INPUT_FILE:-data/raw/sites.csv}"
echo ""

# load python - required on MARS before using python3
module load apps/python3

# activate the virtual environment
source .venv/bin/activate

# confirm what python we are using - good sanity check
echo "python       : $(which python3)"
echo "working dir  : $(pwd)"
echo ""

# create log directory if it does not exist yet
mkdir -p outputs/logs

# run download step only
# INPUT_FILE can be set as an environment variable when submitting:
#   INPUT_FILE=data/raw/my_sites.csv sbatch run_download.sh
python predict.py \
    --input "${INPUT_FILE:-data/raw/sites.csv}" \
    --download-only \
    --structures data/structures \
    --verbose

echo ""
echo "job finished : $(date)"