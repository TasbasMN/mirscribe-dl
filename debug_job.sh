#!/bin/bash

#--- SLURM Job Settings ---#
#SBATCH -p debug
#SBATCH -A mtasbas
#SBATCH -J debug
#SBATCH --output=/arf/scratch/mtasbas/debugJ.out
#SBATCH --error=/arf/scratch/mtasbas/debugJ.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -C weka
#SBATCH --time=00:01:00



# actual code that does stuff-------------------------------------------------------------------------------

# Load the module that makes conda available
module load miniconda3

# Source the conda.sh file from the system installation
source /arf/sw/comp/python/miniconda3/etc/profile.d/conda.sh

#  Activate environment
conda activate mir

# Change to  working directory
cd /arf/scratch/mtasbas/mirscribe-dl/



# Use the full path to Python instead of activation
python main.py 

# actual code that does stuff-------------------------------------------------------------------------------


