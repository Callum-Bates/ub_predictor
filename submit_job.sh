#!/bin/bash
# ------------------------------------------------------------
# submit_job.sh
#
# submits the full ub_predictor pipeline to MARS as two
# chained slurm jobs.
#
# job 1 downloads structure files (small allocation).
# job 2 runs feature generation and training/prediction
#        and starts automatically when job 1 succeeds.
#
# usage:
#   bash submit_job.sh <input_file> <mode>
#
# examples:
#   bash submit_job.sh data/raw/sites.csv train
#   bash submit_job.sh data/raw/sites.csv predict
# ------------------------------------------------------------

# read arguments - use defaults if not provided
INPUT_FILE=${1:-data/raw/sites.csv}
MODE=${2:-train}

# check input file exists before submitting anything
if [ ! -f "$INPUT_FILE" ]; then
    echo "error: input file not found - $INPUT_FILE"
    exit 1
fi

echo "submitting ub_predictor pipeline to MARS"
echo "  input : $INPUT_FILE"
echo "  mode  : $MODE"
echo ""

# submit download job
# --parsable means sbatch only prints the job ID, nothing else
# this lets us capture the ID into a variable
DOWNLOAD_JOB=$(INPUT_FILE=$INPUT_FILE sbatch --parsable run_download.sh)

if [ -z "$DOWNLOAD_JOB" ]; then
    echo "error: download job failed to submit"
    exit 1
fi

echo "  download job submitted : $DOWNLOAD_JOB"

# submit pipeline job with dependency on download job
# afterok means: only start if the previous job exited successfully
# if the download fails, the pipeline job will never start
PIPELINE_JOB=$(INPUT_FILE=$INPUT_FILE MODE=$MODE \
    sbatch --parsable \
    --dependency=afterok:$DOWNLOAD_JOB \
    run_pipeline.sh)

if [ -z "$PIPELINE_JOB" ]; then
    echo "error: pipeline job failed to submit"
    exit 1
fi

echo "  pipeline job submitted : $PIPELINE_JOB"
echo ""
echo "both jobs submitted successfully"
echo ""
echo "monitor progress:"
echo "  squeue -u \$USER"
echo "  tail -f outputs/logs/pipeline_${PIPELINE_JOB}.log"
echo ""
echo "cancel both jobs if needed:"
echo "  scancel $DOWNLOAD_JOB $PIPELINE_JOB"