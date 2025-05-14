#!/bin/bash
#SBATCH --job-name=nextflow_pipeline   # Job name
#SBATCH --ntasks=1                     # Number of tasks (we're running a single task, Nextflow will handle the rest)
#SBATCH --cpus-per-task=1             # Number of CPU cores per task
#SBATCH --mem=2G                     # Memory allocation per task (adjust as needed)
#SBATCH --time=10-00:00:00
#SBATCH --partition=LocalQ           # Partition to submit to (adjust if needed)

OUTPUT_DIR=/mnt/PROCESSED/klettl/nextflow/nextflow-reports/${SLURM_JOB_ID}/
mkdir -p $OUTPUT_DIR  # Create the output directory if it doesn't exist

CONFIG_ID=$1

nextflow run ./pipeline.nf \
    -c ./configs/$CONFIG_ID.config \
    -profile local \
    -with-report ${OUTPUT_DIR}/report.html \
    -with-timeline ${OUTPUT_DIR}/timeline.html \
    -with-trace ${OUTPUT_DIR}/trace.txt \
    "${@:2}"
