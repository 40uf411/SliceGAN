#!/bin/bash
#SBATCH --partition=gpu              # Specify the partition to run the job
#SBATCH --nodes=1                   # Number of nodes to allocate (1 node)
#SBATCH --mem-per-cpu=8096
#SBATCH --gres="gpu:TeslaA100_80:1"
#SBATCH --time=2-00:00:00           # Maximum runtime (2 days)
#SBATCH --job-name=SliceGAN       # Job name
#SBATCH --output=output.log         # Output file for job logs
#SBATCH --error=error.log           # Error file for job errors
#SBATCH --mail-user=ali.aouf@uclouvain.be
#SBATCH --mail-type=ALL


# Load necessary modules
module load releases/2024a
module load Python/3.12.3-GCCcore-13.3.0 

# Change to your code directory
cd /CECI/home/ucl/elen/aaouf/SliceGAN

# load env
source /CECI/home/ucl/elen/aaouf/SliceGAN/slicegan_env/bin/activate

# Run your Python script
python run_slicegan.py

deactivate
