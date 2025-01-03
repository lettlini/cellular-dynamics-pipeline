#!/bin/bash
#SBATCH --job-name=nextflow_pipeline   # Job name
#SBATCH --output=/work/kl63sahy-monolayer/nextflow-logs/nextflow_%j.out       # Standard output log (%j will be replaced with the job ID)
#SBATCH --error=/work/kl63sahy-monolayer/nextflow-logs/nextflow_%j.err        # Standard error log (%j will be replaced with the job ID)
#SBATCH --ntasks=1                     # Number of tasks (we're running a single task, Nextflow will handle the rest)
#SBATCH --cpus-per-task=2             # Number of CPU cores per task
#SBATCH --mem=5G                     # Memory allocation per task (adjust as needed)
#SBATCH --time=2:00:00               # Maximum run time (in HH:MM:SS)
#SBATCH --partition=polaris-long           # Partition to submit to (adjust if needed)

# Load Nextflow module (if it's available as a module)
module load Nextflow
module load Graphviz
module load Mamba

nextflow run ~/cellular-dynamics-pipeline/pipeline.nf \
    -c ~/cellular-dynamics-pipeline/dataset_configs/test.config \
    -profile cluster
