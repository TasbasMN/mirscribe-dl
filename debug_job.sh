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
# module purge
# # Load the module that makes conda available
# module load miniconda3

# Source the conda.sh file from the system installation
source /arf/sw/comp/python/miniconda3/etc/profile.d/conda.sh

# #  Activate environment
# conda activate mir


# Change to  working directory
cd /arf/scratch/mtasbas/mirscribe-dl/

# Use EXPLICIT path to Python in your conda environment
/arf/home/mtasbas/miniconda3/envs/mir/bin/python main.py


# actual code that does stuff-------------------------------------------------------------------------------


# crontab line, works at 2.30am
# 30 2 * * * cd /arf/scratch/mtasbas && sbatch /arf/scratch/mtasbas/mirscribe-dl/debug_job.sh >> /arf/scratch/mtasbas/cron_debug.log 2>&1

