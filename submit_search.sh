#!/bin/bash
# ------------------------------------------------------------
# submit_search.sh
#
# submits a structural similarity search to MARS as two
# chained slurm jobs.
#
# job 1 downloads structure files for target proteins.
# job 2 runs the search and starts when job 1 succeeds.
#
# usage:
#   bash submit_search.sh <reference> <targets_file>
#
# example:
#   bash submit_search.sh Q8IXI2,572 data/raw/test_targets.csv
#
# if your structures are already downloaded, you can skip
# the download step and submit run_search.sh directly:
#   REFERENCE=Q8IXI2,572 TARGETS=data/raw/test_targets.csv sbatch run_search.sh
# ------------------------------------------------------------

REFERENCE=${1}
TARGETS=${2:-data/raw/test_targets.csv}

# validate arguments
if [ -z "$REFERENCE" ]; then
    echo "error: reference site is required"
    echo "usage: bash submit_search.sh <reference> <targets_file>"
    echo "example: bash submit_search.sh Q8IXI2,572 data/raw/test_targets.csv"
    exit 1
fi

if [ ! -f "$TARGETS" ]; then
    echo "error: targets file not found - $TARGETS"
    exit 1
fi

echo "submitting ub_predictor search to MARS"
echo "  reference : $REFERENCE"
echo "  targets   : $TARGETS"
echo ""

# submit download job - reuses the existing download script
# passes the targets file as the input so structures are fetched
# for all target proteins before the search begins
DOWNLOAD_JOB=$(INPUT_FILE=$TARGETS sbatch --parsable run_download.sh)

if [ -z "$DOWNLOAD_JOB" ]; then
    echo "error: download job failed to submit"
    exit 1
fi

echo "  download job submitted : $DOWNLOAD_JOB"

# submit search job with dependency on download completing
SEARCH_JOB=$(REFERENCE=$REFERENCE TARGETS=$TARGETS \
    sbatch --parsable \
    --dependency=afterok:$DOWNLOAD_JOB \
    run_search.sh)

if [ -z "$SEARCH_JOB" ]; then
    echo "error: search job failed to submit"
    exit 1
fi

echo "  search job submitted   : $SEARCH_JOB"
echo ""
echo "both jobs submitted successfully"
echo ""
echo "monitor progress:"
echo "  squeue -u \$USER"
echo "  tail -f outputs/logs/search_${SEARCH_JOB}.log"
echo ""
echo "cancel both jobs if needed:"
echo "  scancel $DOWNLOAD_JOB $SEARCH_JOB"