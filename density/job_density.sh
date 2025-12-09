#!/bin/bash

#SBATCH --job-name="density"
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=10000
#SBATCH --output=density.out
#SBATCH --error=density.err
#SBATCH --account=spitvent

if [ $# -ne 1 ]; then
    echo "Usage: $0 EXPERIMENT_NAME"
    exit 1
fi

source /home/${USER}/.bashrc
source activate ece4

python -u diag_density.py "$1"

