#!/bin/bash

# crontab -e command
# */10 * * * * /bin/bash /arf/scratch/mtasbas/mirscribe-dl/cron.sh >> /arf/home/mtasbas/batch_submission.log 2>&1

# Define the path to your sbatch command files and working directory
SBATCH_FILES_DIR="/arf/scratch/mtasbas/mirscribe-dl"
HOME_DIR="/arf/home/mtasbas"  # Your home directory

# Define the path for logs and index tracking in home directory
LOG_FILE="$HOME_DIR/batch_submission.log"
INDEX_FILE="$HOME_DIR/.sbatch_current_index"

# Function to echo with timestamp
log_message() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

SBATCH_FILES=(
  "sbatch_commands_group_1_sim3.txt"
  "sbatch_commands_group_2_sim3.txt"
  "sbatch_commands_group_3_sim3.txt"
  "sbatch_commands_group_4_sim3.txt"
  "sbatch_commands_group_5_sim3.txt"
  "sbatch_commands_group_6_sim3.txt"
  # Add more files as needed
)

# Initialize the index file if it doesn't exist
if [ ! -f "$INDEX_FILE" ]; then
  echo "0" > "$INDEX_FILE"
fi

# Get current index
CURRENT_INDEX=$(cat "$INDEX_FILE")

# Check if we've processed all files
if [ "$CURRENT_INDEX" -ge "${#SBATCH_FILES[@]}" ]; then
  log_message "All sbatch files have been processed. Removing cron job."
  
  # Remove this script from crontab
  SCRIPT_PATH="$0"
  ESCAPED_SCRIPT_PATH=$(echo "$SCRIPT_PATH" | sed 's/\//\\\//g')
  crontab -l | grep -v "$ESCAPED_SCRIPT_PATH" | crontab -
  
  log_message "Cron job removed successfully. Automation complete."
  exit 0
fi

# Count jobs in the queue for your user
MY_QUEUED_JOBS=$(squeue -u $(whoami) -h | wc -l)

# If fewer than 20 jobs are running (meaning 80 or fewer in queue)
if [ "$MY_QUEUED_JOBS" -le 20 ]; then
  log_message "Queue has only $MY_QUEUED_JOBS jobs. Submitting next batch..."
  
  # Submit the next batch
  NEXT_BATCH="${SBATCH_FILES[$CURRENT_INDEX]}"
  cd "$SBATCH_FILES_DIR"  # Make sure we're in the right directory when submitting
  log_message "Executing commands from $NEXT_BATCH"
  cat "$NEXT_BATCH" | bash
  
  # Increment the index
  CURRENT_INDEX=$((CURRENT_INDEX + 1))
  echo "$CURRENT_INDEX" > "$INDEX_FILE"
  
  log_message "Submitted batch $NEXT_BATCH. Updated index to $CURRENT_INDEX"
  
  # If this was the last batch, remove the cron job on the next run
  if [ "$CURRENT_INDEX" -ge "${#SBATCH_FILES[@]}" ]; then
    log_message "This was the final batch. Cron job will self-terminate on next execution."
  fi
else
  log_message "Queue still has $MY_QUEUED_JOBS jobs. Waiting for queue to decrease."
fi
