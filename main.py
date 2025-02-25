import os
from scripts.pipeline import *
from scripts.config import *
import torch
import time
import subprocess
import multiprocessing


# Set start method for multiprocessing to 'spawn' for proper process isolation
# This is particularly important for CUDA operations across multiple processes
if __name__ == '__main__':
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        # Method already set
        pass


def main():
    # Create the output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Print CUDA availability and device count
    if torch.cuda.is_available():
        cuda_device_count = torch.cuda.device_count()
        print(f"CUDA is available with {cuda_device_count} device(s)")
        for i in range(cuda_device_count):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA is not available, using CPU")
    
    print(f"Multiprocessing with {WORKERS} workers")
    
    if SINGLE_CHROMOSOME:
        run_pipeline_single(VCF_FULL_PATH, CHUNKSIZE, OUTPUT_DIR, VCF_ID)
        print("run_pipeline_single   ✓")
    
    else:
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

    # Get starting time
    start_time = time.time()
    main()
    # Get ending time
    end_time = time.time()
    # Get total time
    total_time = end_time - start_time
    
    # Calculate seconds per line
    seconds_per_line = total_time / total_lines
    
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Seconds per line: {seconds_per_line:.4f}")
    print(f"Lines per second: {1/seconds_per_line:.2f}")