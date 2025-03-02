#!/bin/bash

# ==============================================================================
#  MIRSCRIBE BATCH JOB SCHEDULER
# ==============================================================================
#
#  Purpose:    Automated HPC batch job submission with queue monitoring
#  Author:     mtasbas
#  Created:    2024
#  Usage:      Run from crontab every 10 minutes:
#                    */10 * * * * /bin/bash -c "cd /arf/scratch/mtasbas && /arf/scratch/mtasbas/mirscribe-dl/cron.sh >> /arf/home/mtasbas/batch_submission.log 2>&1"
# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Directory paths
SBATCH_FILES_DIR="/arf/scratch/mtasbas/mirscribe-dl"
HOME_DIR="/arf/home/mtasbas"

# Tracking files
LOG_FILE="$HOME_DIR/batch_submission.log"
INDEX_FILE="$HOME_DIR/.sbatch_current_index"
CREDENTIALS_FILE="$HOME_DIR/.pushover_credentials"  # Store API keys here

# Job control settings
MAX_QUEUED_JOBS=20

# Define the simulation variable
SIM_TYPE="sim4"

# Ordered list of batch files to process sequentially
SBATCH_FILES=(
  "sbatch_commands_group_0_${SIM_TYPE}.txt"
  "sbatch_commands_group_1_${SIM_TYPE}.txt"
  "sbatch_commands_group_2_${SIM_TYPE}.txt"
  "sbatch_commands_group_3_${SIM_TYPE}.txt"
  "sbatch_commands_group_4_${SIM_TYPE}.txt"
  "sbatch_commands_group_5_${SIM_TYPE}.txt"
  "sbatch_commands_group_6_${SIM_TYPE}.txt"
  # Add more files as needed
)

# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------

# Load Pushover credentials from external file
load_pushover_credentials() {
  if [ -f "$CREDENTIALS_FILE" ]; then
    source "$CREDENTIALS_FILE"
  else
    log_message "Credentials file not found. Notifications will not be sent."
    PUSHOVER_ENABLED="false"
  fi
}

# Log a message with timestamp
log_message() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Send notification via Pushover API
send_notification() {
  # Skip if notifications aren't enabled
  if [ "$PUSHOVER_ENABLED" != "true" ]; then
    log_message "Skipping notification (not enabled): $1 - $2"
    return
  fi
  
  local title="$1"
  local message="$2"
  local priority="${3:-0}"  # Default to normal priority if not specified
  
  log_message "Sending notification: $title - $message"
  
  curl -s \
    --form-string "token=$PUSHOVER_API_TOKEN" \
    --form-string "user=$PUSHOVER_USER_KEY" \
    --form-string "title=$title" \
    --form-string "message=$message" \
    --form-string "priority=$priority" \
    https://api.pushover.net/1/messages.json > /dev/null 2>&1
    
  if [ $? -eq 0 ]; then
    log_message "Notification sent successfully"
  else
    log_message "Failed to send notification"
  fi
}

# Initialize the index tracking file if it doesn't exist
initialize_index_file() {
  if [ ! -f "$INDEX_FILE" ]; then
    echo "0" > "$INDEX_FILE"
    log_message "Initialized index tracking at $INDEX_FILE"
  fi
}

# Get the current batch processing index
get_current_index() {
  cat "$INDEX_FILE"
}

# Update the index after processing a batch
update_index() {
  local new_index=$1
  echo "$new_index" > "$INDEX_FILE"
}

# Count the number of jobs in the queue for the current user
count_queued_jobs() {
  squeue -u $(whoami) -h | wc -l
}

# Submit the next batch of jobs
submit_next_batch() {
  local batch_index=$1
  local batch_file="${SBATCH_FILES[$batch_index]}"
  
  cd "$SBATCH_FILES_DIR"
  log_message "Executing commands from $batch_file"
  # cat "$batch_file" | bash is commented out because it catches "submitted batch job" messages, polluting the logfile
  cat "$batch_file" | bash > /dev/null

  local new_index=$((batch_index + 1))
  update_index "$new_index"
  log_message "Submitted batch $batch_file. Updated index to $new_index"
  
  # Send notification about new batch submission
  local batch_number=$((batch_index + 1))
  local total_batches=${#SBATCH_FILES[@]}
  send_notification "Batch Job Submitted" "Batch $batch_number of $total_batches submitted to HPC queue (File: $batch_file)"
  
  # Check if that was the last batch
  if [ "$new_index" -ge "${#SBATCH_FILES[@]}" ]; then
    log_message "This was the final batch. Cron job will self-terminate on next execution."
    send_notification "Final Batch Submitted" "All $total_batches batches have been submitted. Cron job will self-terminate on next execution." 1
  fi
}

# Remove this script from crontab when all batches are processed
remove_cron_job() {
  log_message "All sbatch files have been processed. Removing cron job."
  
  # Remove the tracking file
  if [ -f "$INDEX_FILE" ]; then
    rm "$INDEX_FILE"
    log_message "Removed index tracking file $INDEX_FILE"
  fi
  
  # Remove this script from crontab
  SCRIPT_PATH="$0"
  ESCAPED_SCRIPT_PATH=$(echo "$SCRIPT_PATH" | sed 's/\//\\\//g')
  crontab -l | grep -v "$ESCAPED_SCRIPT_PATH" | crontab -
  
  log_message "Cron job removed successfully. Automation complete."
  
  # Send notification about completion
  send_notification "Batch Processing Complete" "All batches have been processed and the cron job has been removed." 1
}

# -----------------------------------------------------------------------------
# Main Script Logic
# -----------------------------------------------------------------------------

# Load Pushover credentials
load_pushover_credentials

# Initialize tracking file if needed
initialize_index_file

# Get current processing index
CURRENT_INDEX=$(get_current_index)

# Check if we've processed all files
if [ "$CURRENT_INDEX" -ge "${#SBATCH_FILES[@]}" ]; then
  remove_cron_job
  exit 0
fi

# Count jobs in the queue for current user
MY_QUEUED_JOBS=$(count_queued_jobs)

# If queue has space for more jobs, submit the next batch
if [ "$MY_QUEUED_JOBS" -le "$MAX_QUEUED_JOBS" ]; then
  log_message "Queue has only $MY_QUEUED_JOBS jobs. Submitting next batch..."
  submit_next_batch "$CURRENT_INDEX"
else
  log_message "Queue still has $MY_QUEUED_JOBS jobs. Waiting for queue to decrease."
fi
