#!/bin/bash

#--- SLURM Job Settings ---#
#SBATCH -p debug
#SBATCH -A mtasbas
#SBATCH -J ${SUBFOLDER_NAME}_${VCF_BASENAME}
#SBATCH --error=${LOGS_DIR}/%J.err
#SBATCH --output=${LOGS_DIR}/%J.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${NUM_CPUS}
#SBATCH -C weka
#SBATCH --time=${ALLOCATED_TIME}
#SBATCH --mail-user=nazifts@gmail.com
#SBATCH --mail-type=ALL

#--- Job Specific Settings ---#
VCF_FILE="${VCF_FILE}"
JOB_NAME="\$(date +%Y%m%d_%H%M)_${SUBFOLDER_NAME}_${VCF_BASENAME}"
LINE_COUNT=${LINE_COUNT}
CHUNK_SIZE=${CHUNK_SIZE}
OUTPUT_DIR="${OUTPUT_DIR}"
LOGS_DIR="${LOGS_DIR}"

#--- Function Definitions ---#
seconds_to_hhmmss() {
    local seconds=\$1
    printf '%02dh:%02dm:%02ds\n' \$((seconds/3600)) \$((seconds%3600/60)) \$((seconds%60))
}

#--- Job Execution ---#
mkdir -p "\${LOGS_DIR}"
mkdir -p "\${OUTPUT_DIR}"
START_TIME=\$(date +%s)

echo "Started at: \$(date)"
echo "Start time variable: \${START_TIME}"

echo "VCF file: \${VCF_FILE}"
echo "Allocated time: ${ALLOCATED_TIME}"
echo "Chunk size: \${CHUNK_SIZE}"
echo "Number of CPUs: ${NUM_CPUS}"
echo "Output directory: \${OUTPUT_DIR}"
echo "Logs directory: \${LOGS_DIR}"

# actual code that does stuff-------------------------------------------------------------------------------
module load miniconda3
cd ${SCRATCH_DIR}/mirscribe-dl/

# Add these before running the Python command
echo "Current directory: \$(pwd)"
echo "Full VCF path: \${VCF_FILE}"
echo "Output directory: \${OUTPUT_DIR}"
echo "File exists check: \$(if [ -f "\${VCF_FILE}" ]; then echo "YES"; else echo "NO"; fi)"

# Use the full path to Python instead of activation
/arf/home/mtasbas/miniconda3/envs/mir/bin/python main.py -f "\${VCF_FILE}" -w ${NUM_CPUS} -c \${CHUNK_SIZE} -o "\${OUTPUT_DIR}"

# actual code that does stuff-------------------------------------------------------------------------------

echo "Completed at: \$(date)"


#--- Job Statistics ---#

# Rename the log files
mv "${LOGS_DIR}/\${SLURM_JOB_ID}.out" "${LOGS_DIR}/\${JOB_NAME}.out"

# Check if error file is empty and handle accordingly
if [ -s "${LOGS_DIR}/\${SLURM_JOB_ID}.err" ]; then
    # Error file has content, so rename it
    mv "${LOGS_DIR}/\${SLURM_JOB_ID}.err" "${LOGS_DIR}/\${JOB_NAME}.err"
else
    # Error file is empty, so delete it
    rm "${LOGS_DIR}/\${SLURM_JOB_ID}.err"
    echo "No errors detected, error file removed"
fi

#--- Final Report ---#
END_TIME=\$(date +%s)
RUNTIME=\$((END_TIME - START_TIME))
RUNTIME_HHMMSS=\$(seconds_to_hhmmss \$RUNTIME)
AVG_TIME_PER_LINE=\$(awk "BEGIN {printf \\"%.6f\\", \${RUNTIME} / \${LINE_COUNT}}")

echo "Job statistics:"
echo "Runtime: \${RUNTIME} seconds (\${RUNTIME_HHMMSS})"
echo "Input file line count: \${LINE_COUNT}"
echo "Average time per input line: \${AVG_TIME_PER_LINE} seconds"
echo "Runtime: \${RUNTIME}"
echo "line count: \${LINE_COUNT}"
echo "Results saved to: \${OUTPUT_DIR}"
echo "Logs saved to: \${LOGS_DIR}"