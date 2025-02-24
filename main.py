import os
from scripts.pipeline import *
from scripts.config import *
import torch
import time
import subprocess
import multiprocessing


# Use multiple threads for PyTorch operations
# This allows PyTorch to use multiple CPU cores for tensor operations
num_physical_cores = max(1, multiprocessing.cpu_count() // 2)
torch.set_num_threads(num_physical_cores)



            



def main():

    # Create the output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    run_pipeline(VCF_FULL_PATH, CHUNKSIZE, OUTPUT_DIR, VCF_ID)
    print("run_pipeline         ✓")

    results_filename = f"results_{VCF_ID}.csv"
    
    # stitch_and_cleanup_csv_files(OUTPUT_DIR, results_filename)
    # print("stitch_and_cleanup   ✓")
    
    # delete_fasta_files(OUTPUT_DIR)
    # print("delete_fasta_files   ✓")

    # delete_files(OUTPUT_DIR, "rnad", ".csv")

    
if __name__ == '__main__':
    # Get line count
    result = subprocess.run(['wc', '-l', VCF_FULL_PATH], capture_output=True, text=True)
    total_lines = int(result.stdout.split()[0])
    print(f"Total lines in VCF: {total_lines}")

    # get starting time
    start_time = time.time()
    main()
    # get ending time
    end_time = time.time()
    # get total time
    total_time = end_time - start_time
    
    # Calculate seconds per line
    seconds_per_line = total_time / total_lines
    
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Seconds per line: {seconds_per_line:.4f}")
    print(f"Lines per second: {1/seconds_per_line:.2f}")