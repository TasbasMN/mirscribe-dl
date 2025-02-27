#!/bin/bash

# Adjust this value based on your average processing time per line (in seconds)
TIME_PER_LINE=0.2

# Check if an argument is provided
if [ $# -eq 0 ]; then
    echo "Error: Please provide a target directory containing VCF files as an argument."
    exit 1
fi

# Updated home and scratch directories
HOME_DIR="/arf/home/mtasbas"
SCRATCH_DIR="/arf/scratch/mtasbas"

TARGET_DIR="$1"
# Remove trailing slash from TARGET_DIR if present
TARGET_DIR="${TARGET_DIR%/}"

SCRIPT_DIR="${SCRATCH_DIR}/scripts/$(basename ${TARGET_DIR})"
LOGS_DIR="${SCRATCH_DIR}/logs/$(basename ${TARGET_DIR})"
OUTPUT_FILE="sbatch_commands_$(basename ${TARGET_DIR}).txt"

mkdir -p "${SCRIPT_DIR}"
mkdir -p "${LOGS_DIR}"

calculate_time() {
    local lines=$1
    local total_seconds=$(echo "$lines * $TIME_PER_LINE" | bc)
    local total_minutes=$(echo "($total_seconds + 59) / 60" | bc)
    local rounded_minutes=$(( (total_minutes + 29) / 30 * 30 ))
    local hours=$(( rounded_minutes / 60 ))
    local minutes=$(( rounded_minutes % 60 ))
    printf "%02d:%02d:00" $hours $minutes
}

calculate_conservative_chunk_size() {
    local total_lines=$1
    local TOTAL_MEMORY_MB=$((192 * 1024))  # 192 GB in MB
    local NUM_CORES=54
    local MEMORY_PER_LINE=3  # MB
    local BASE_MEMORY=200  # MB
    local SAFETY_FACTOR=0.8  # Use only 80% of available memory to be conservative

    local usable_memory=$(echo "($TOTAL_MEMORY_MB - $BASE_MEMORY) * $SAFETY_FACTOR" | bc)
    local max_lines_per_core=$(echo "$usable_memory / ($MEMORY_PER_LINE * $NUM_CORES)" | bc)
    local total_chunks=$(echo "($total_lines + $max_lines_per_core - 1) / $max_lines_per_core" | bc)
    local passes=$(echo "($total_chunks + $NUM_CORES - 1) / $NUM_CORES" | bc)

    echo $(( total_lines / (passes * NUM_CORES) ))
}

# Check if target directory exists
if [ ! -d "${TARGET_DIR}" ]; then
    echo "Error: Target directory '${TARGET_DIR}' does not exist."
    exit 1
fi

# Check if there are any VCF files
VCF_FILES=("${TARGET_DIR}"/*.vcf)
if [ ${#VCF_FILES[@]} -eq 0 ] || [ ! -e "${VCF_FILES[0]}" ]; then
    echo "Error: No VCF files found in '${TARGET_DIR}'."
    exit 1
fi

# Clear the output file before writing
> "$OUTPUT_FILE"

# Extract the folder name for output organization
FOLDER_NAME=$(basename "${TARGET_DIR}")

for VCF_FILE in "${VCF_FILES[@]}"; do
    VCF_BASENAME=$(basename "${VCF_FILE}" .vcf)
    SCRIPT_NAME="${SCRIPT_DIR}/${VCF_BASENAME}_job.sh"
    
    if [ ! -f "${VCF_FILE}" ]; then
        continue
    fi
    
    LINE_COUNT=$(wc -l < "${VCF_FILE}")
    ALLOCATED_TIME=$(calculate_time $LINE_COUNT)
    CHUNK_SIZE=$(calculate_conservative_chunk_size $LINE_COUNT)
    NUM_CPUS=54

    cat << EOF > "${SCRIPT_NAME}"
#!/bin/bash

#--- SLURM Job Settings ---#
#SBATCH -p hamsi
#SBATCH -A mtasbas
#SBATCH -J ${VCF_BASENAME}
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
JOB_NAME="\$(date +%Y%m%d_%H%M)_${VCF_BASENAME}"
LINE_COUNT=${LINE_COUNT}
CHUNK_SIZE=${CHUNK_SIZE}

#--- Function Definitions ---#
seconds_to_hhmmss() {
    local seconds=\$1
    printf '%02dh:%02dm:%02ds\n' \$((seconds/3600)) \$((seconds%3600/60)) \$((seconds%60))
}

#--- Job Execution ---#
mkdir -p "${LOGS_DIR}"
START_TIME=\$(date +%s)

echo "Started at: \$(date)"
echo "Start time variable: \${START_TIME}"

echo "VCF file: \${VCF_FILE}"
echo "Allocated time: ${ALLOCATED_TIME}"
echo "Chunk size: \${CHUNK_SIZE}"
echo "Number of CPUs: ${NUM_CPUS}"


# actual code that does stuff-------------------------------------------------------------------------------
module load miniconda3
conda activate mir
cd ${SCRATCH_DIR}/mirscribe-dl/

# Add these before running the Python command
echo "Current directory: $(pwd)"
echo "Full VCF path: ${VCF_FILE}"
echo "Processed path: ${RELATIVE_PATH}"
echo "File exists check: $(if [ -f "${RELATIVE_PATH}" ]; then echo "YES"; else echo "NO"; fi)"


# Use the absolute path directly
python main.py -f "\${VCF_FILE}" -w ${NUM_CPUS} -c \${CHUNK_SIZE}

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
EOF

    chmod +x "${SCRIPT_NAME}"
    
    # Add sbatch command to the output file
    echo "sbatch $SCRIPT_NAME" >> "$OUTPUT_FILE"
done

echo "All SLURM scripts have been generated in ${SCRIPT_DIR}"
echo "sbatch commands have been written to $OUTPUT_FILE"
echo "Found and processed ${#VCF_FILES[@]} VCF files from ${TARGET_DIR}"