#!/bin/bash
#
# VCF Job Generator
# Generates SLURM job scripts for processing VCF files
#

#------------------------------------------------------------------------------
# Configuration
#------------------------------------------------------------------------------
SCRATCH_DIR="/arf/scratch/mtasbas"
TIME_PER_LINE=0.2
NUM_CPUS=54

#------------------------------------------------------------------------------
# Input Validation & Setup
#------------------------------------------------------------------------------
if [ $# -ne 1 ]; then
    echo "Usage: $0 <target_directory>"
    exit 1
fi

TARGET_DIR="${1%/}"

if [ ! -d "${TARGET_DIR}" ]; then
    echo "Error: Directory '${TARGET_DIR}' not found."
    exit 1
fi

# Get subfolder from path and setup directories
SUBFOLDER=$(echo "$TARGET_DIR" | cut -d'/' -f2)
BASE_NAME=$(basename ${TARGET_DIR})
SCRIPT_DIR="${SCRATCH_DIR}/scripts/${SUBFOLDER}/${BASE_NAME}"
LOGS_DIR="${SCRATCH_DIR}/logs/${SUBFOLDER}/${BASE_NAME}"
OUTPUT_DIR="${SCRATCH_DIR}/mirscribe-dl/results/${SUBFOLDER}"
BATCH_FILE="sbatch_commands_${BASE_NAME}_${SUBFOLDER}.txt"

mkdir -p "${SCRIPT_DIR}" "${LOGS_DIR}"
> "${BATCH_FILE}"

echo "Processing VCF files in: ${TARGET_DIR}"

#------------------------------------------------------------------------------
# Helper Functions
#------------------------------------------------------------------------------
calculate_time() {
    local lines=$1
    local seconds=$(echo "$lines * $TIME_PER_LINE" | bc)
    local minutes=$(echo "($seconds + 59) / 60" | bc)
    local rounded=$(( (minutes + 9) / 10 * 10 ))
    printf "%02d:%02d:00" $((rounded / 60)) $((rounded % 60))
}


calculate_chunk_size() {
    local lines=$1
    # Simply divide the number of lines by the number of cores
    echo $(( (lines + NUM_CPUS - 1) / NUM_CPUS ))
}

#------------------------------------------------------------------------------
# Process VCF Files
#------------------------------------------------------------------------------
VCF_FILES=("${TARGET_DIR}"/*.vcf)
if [ ! -e "${VCF_FILES[0]}" ]; then
    echo "Error: No VCF files found."
    exit 1
fi

for VCF_FILE in "${VCF_FILES[@]}"; do
    [ ! -f "${VCF_FILE}" ] && continue
    
    VCF_NAME=$(basename "${VCF_FILE}" .vcf)
    SCRIPT="${SCRIPT_DIR}/${VCF_NAME}_${SUBFOLDER}_job.sh"
    
    LINES=$(wc -l < "${VCF_FILE}")
    RUNTIME=$(calculate_time $LINES)
    CHUNKS=$(calculate_chunk_size $LINES)

    # Generate the job script with streamlined logging
    cat > "${SCRIPT}" << EOF
#!/bin/bash

#--- SLURM Job Settings ---#
#SBATCH -p hamsi
#SBATCH -A mtasbas
#SBATCH -J ${SUBFOLDER}_${VCF_NAME}
#SBATCH --error=${LOGS_DIR}/%J.err
#SBATCH --output=${LOGS_DIR}/%J.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${NUM_CPUS}
#SBATCH -C weka
#SBATCH --time=${RUNTIME}
#SBATCH --mail-user=nazifts@gmail.com
#SBATCH --mail-type=ALL

#--- Job Configuration ---#
VCF_FILE="${VCF_FILE}"
JOB_NAME="\$(date +%Y%m%d_%H%M)_${SUBFOLDER}_${VCF_NAME}"
OUTPUT_DIR="${OUTPUT_DIR}"
LOGS_DIR="${LOGS_DIR}"

# Create required directories
mkdir -p "\${OUTPUT_DIR}" "\${LOGS_DIR}"

# Log basic job info
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Job started: \${VCF_FILE} (${LINES} lines)"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Configuration: ${NUM_CPUS} CPUs, chunk size ${CHUNKS}"
START_TIME=\$(date +%s)

#--- Execute Processing ---#
# Setup environment
source /arf/sw/comp/python/miniconda3/etc/profile.d/conda.sh
cd ${SCRATCH_DIR}/mirscribe-dl/

# Run the main processing script
/arf/home/mtasbas/miniconda3/envs/mir/bin/python main.py \\
    -f "\${VCF_FILE}" \\
    -w ${NUM_CPUS} \\
    -c ${CHUNKS} \\
    -o "\${OUTPUT_DIR}"

#--- Post-processing ---#
# Rename log files with timestamp for better organization
if [ -f "${LOGS_DIR}/\${SLURM_JOB_ID}.out" ]; then
    mv "${LOGS_DIR}/\${SLURM_JOB_ID}.out" "${LOGS_DIR}/\${JOB_NAME}.out"
fi

# Handle error file
if [ -s "${LOGS_DIR}/\${SLURM_JOB_ID}.err" ]; then
    mv "${LOGS_DIR}/\${SLURM_JOB_ID}.err" "${LOGS_DIR}/\${JOB_NAME}.err"
else
    rm -f "${LOGS_DIR}/\${SLURM_JOB_ID}.err"
fi

# Calculate runtime statistics
END_TIME=\$(date +%s)
RUNTIME=\$((END_TIME - START_TIME))
HOURS=\$((RUNTIME / 3600))
MINUTES=\$(( (RUNTIME % 3600) / 60 ))
SECONDS=\$((RUNTIME % 60))

# Print simple completion summary
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Job completed in \${HOURS}h:\${MINUTES}m:\${SECONDS}s"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Results saved to: \${OUTPUT_DIR}"
EOF

    chmod +x "${SCRIPT}"
    echo "sbatch ${SCRIPT}" >> "${BATCH_FILE}"
    echo "Generated job for: ${VCF_NAME}"
done

echo "Generated scripts for ${#VCF_FILES[@]} files"
echo "Batch commands in: ${BATCH_FILE}"
