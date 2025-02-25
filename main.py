import os, time, subprocess, contextlib, torch, multiprocessing
from scripts.pipeline import *
from scripts.config import *

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if torch.cuda.is_available():
        devices = torch.cuda.device_count()
        print(f"CUDA: {devices} device(s)")
        for i in range(devices):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CPU mode (no CUDA)")
    
    print(f"Workers: {WORKERS}")
    
    if SINGLE_CHROMOSOME:
        run_pipeline_single(VCF_FULL_PATH, CHUNKSIZE, OUTPUT_DIR, VCF_ID)
        print("✓ Single pipeline")
    else:
        run_pipeline(VCF_FULL_PATH, CHUNKSIZE, OUTPUT_DIR, VCF_ID)
        print("✓ Multi pipeline")

    compile_results(OUTPUT_DIR, f"results_{VCF_ID}.csv")
    print("✓ Results compiled")

if __name__ == '__main__':
    with contextlib.suppress(RuntimeError):
        multiprocessing.set_start_method('spawn')
    
    try:
        lines = int(subprocess.run(['wc', '-l', VCF_FULL_PATH], 
                   capture_output=True, text=True).stdout.split()[0])
        print(f"VCF: {lines} lines")
    except:
        lines = 0
        print("VCF: line count failed")

    start = time.time()
    main()
    elapsed = time.time() - start
    
    print(f"Time: {elapsed:.2f}s")
    if lines:
        spl = elapsed / lines
        print(f"Speed: {spl:.4f}s/line ({1/spl:.2f} lines/s)")
