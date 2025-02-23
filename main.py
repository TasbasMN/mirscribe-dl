import os
from scripts.pipeline import *
from scripts.config import *
import torch

torch.set_num_threads(1)



print("Starting main.py")
import os
print("Current working directory:", os.getcwd())
from scripts.config import *
print("Config imported")




            



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
        main()



