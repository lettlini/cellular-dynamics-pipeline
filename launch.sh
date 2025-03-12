#!/bin/bash
#SBATCH --job-name=nextflow_pipeline   # Job name
#SBATCH --output=/home/sc.uni-leipzig.de/kl63sahy/nextflow/nextflow-logs/nextflow_%j.out       # Standard output log (%j will be replaced with the job ID)
#SBATCH --error=/home/sc.uni-leipzig.de/kl63sahy/nextflow/nextflow-logs/nextflow_%j.err        # Standard error log (%j will be replaced with the job ID)
#SBATCH --ntasks=1                     # Number of tasks (we're running a single task, Nextflow will handle the rest)
#SBATCH --cpus-per-task=1             # Number of CPU cores per task
#SBATCH --mem=2G                     # Memory allocation per task (adjust as needed)
#SBATCH --time=2-00:00:00
#SBATCH --partition=paul           # Partition to submit to (adjust if needed)

OUTPUT_DIR=/home/sc.uni-leipzig.de/kl63sahy/nextflow/nextflow-reports/${SLURM_JOB_ID}/
mkdir -p $OUTPUT_DIR  # Create the output directory if it doesn't exist

# Load Nextflow module
module load Nextflow

# Load Mamba module & make sure conda and mamba are on the path
module load Mamba
source /software/easybuild/el8/amd_zen1/all/Mamba/23.1.0-4/etc/profile.d/conda.sh

CONFIG_ID=$1

nextflow run ./pipeline.nf \
    -c ./dataset_configs/$CONFIG_ID.config \
    -profile cluster \
    -with-report ${OUTPUT_DIR}/report.html \
    -with-timeline ${OUTPUT_DIR}/timeline.html \
    -with-trace ${OUTPUT_DIR}/trace.txt \
    "${@:2}"
