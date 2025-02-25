import os
import time
import subprocess
import contextlib
import torch
import multiprocessing
from scripts.pipeline import *
from scripts.config import *


def main():
    # Create the output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Check and print CUDA availability
    if torch.cuda.is_available():
        cuda_device_count = torch.cuda.device_count()
        print(f"CUDA is available with {cuda_device_count} device(s)")
        for i in range(cuda_device_count):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA is not available, using CPU")
    
    print(f"Multiprocessing with {WORKERS} workers")
    
    # Run appropriate pipeline based on configuration
    if SINGLE_CHROMOSOME:
        run_pipeline_single(VCF_FULL_PATH, CHUNKSIZE, OUTPUT_DIR, VCF_ID)
        print("run_pipeline_single   ✓")
    else:
        run_pipeline(VCF_FULL_PATH, CHUNKSIZE, OUTPUT_DIR, VCF_ID)
        print("run_pipeline         ✓")

    # Generate results filename and process results
    results_filename = f"results_{VCF_ID}.csv"
    stitch_and_cleanup_csv_files(OUTPUT_DIR, results_filename)
    print("stitch_and_cleanup   ✓")


if __name__ == '__main__':
    # Set start method for multiprocessing to 'spawn' for proper process isolation
    # This is particularly important for CUDA operations across multiple processes
    with contextlib.suppress(RuntimeError):
        multiprocessing.set_start_method('spawn')
    
    # Get line count
    try:
        result = subprocess.run(['wc', '-l', VCF_FULL_PATH], capture_output=True, text=True, check=True)
        total_lines = int(result.stdout.split()[0])
        print(f"Total lines in VCF: {total_lines}")
    except (subprocess.SubprocessError, ValueError) as e:
        print(f"Error counting lines in VCF file: {e}")
        total_lines = 0

    # Get starting time and run main process
    start_time = time.time()
    main()
    end_time = time.time()
    total_time = end_time - start_time
    
    # Calculate and print performance metrics
    if total_lines > 0:
        seconds_per_line = total_time / total_lines
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Seconds per line: {seconds_per_line:.4f}")
        print(f"Lines per second: {1/seconds_per_line:.2f}")
    else:
        print(f"Total time: {total_time:.2f} seconds")
        print("Could not calculate per-line metrics")
