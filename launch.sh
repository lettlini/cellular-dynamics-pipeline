#!/bin/bash
#SBATCH --partition sirius-long
#SBATCH --mem 5G
#SBATCH -c 1
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1

WORKFLOW=$1
CONFIG=$2

# Use a conda environment where you have installed Nextflow
# (may not be needed if you have installed it in a different way)
conda activate nextflow

nextflow -C ${CONFIG} run ${WORKFLOW}
