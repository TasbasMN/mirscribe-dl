import os, csv, gc, logging, multiprocessing
import pandas as pd
import torch
from typing import List
from concurrent.futures import ProcessPoolExecutor, as_completed
from scripts.functions import *
from scripts.targetnet import TargetNet
from scripts.config import *

# Global model variable that will be initialized in each worker process
MODEL = None

def init_worker(model_path, device_str, cfg):
    """Initialize the model in each worker process"""
    global MODEL
    
    device = torch.device(device_str)
    
    # Create model and move to device
    MODEL = TargetNet(cfg, with_esa=True, dropout_rate=0.5).to(device)
    
    # Load weights and set to eval mode
    MODEL.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
    MODEL.eval()


def analysis_pipeline(df, start_idx, end_idx, out_dir, vcf_id):

    df['id'] = df['id'].astype(str) + '_' + df['chr'].astype(str) + '_' + df['pos'].astype(str) + '_' + df['ref'] + '_' + df['alt']
    df = generate_is_mirna_column(df, grch=37)
    df = add_sequence_columns(df, upstream_offset=UPSTREAM_OFFSET, downstream_offset=DOWNSTREAM_OFFSET)
    case_1 = classify_and_get_case_1_mutations(df, vcf_id, start_idx, end_idx, out_dir)
    
    # gc
    del df
    gc.collect()
    
    # Use the global MODEL initialized for this process
    global MODEL
    device = next(MODEL.parameters()).device
    
    # Load miRNA sequences
    mirnas = pd.read_csv(MIRNA_CSV).set_index('mirna_accession')['sequence'].to_dict()

    # Make predictions using the process-local model
    df = predict_binding_to_df(MODEL, device, case_1, mirnas, S_MATRIX, BATCH_SIZE)
    df = post_process_predictions(df, DEC_SURF, FILTER_THRESH)

    return df


def analysis_pipeline_single(df, start_idx, end_idx, out_dir, vcf_id):
    # Convert chr to category if it's object type
    if 'chr' in df.columns and df['chr'].dtype == 'object':
        df['chr'] = df['chr'].astype('category')
    
    # Create unique IDs
    df['id'] = df['id'].astype(str) + '_' + df['chr'].astype(str) + '_' + df['pos'].astype(str) + '_' + df['ref'] + '_' + df['alt']

    # Preload chromosome if single chromosome data
    unique_chroms = df['chr'].unique()
    if len(unique_chroms) == 1:
        from scripts.sequence_utils import load_chrom
        load_chrom(unique_chroms[0])
        print(f"Pre-loaded chromosome {unique_chroms[0]} for faster access")

    df = generate_is_mirna_column_single(df, grch=37)
    df = add_sequence_columns_single(df, upstream_offset=UPSTREAM_OFFSET, downstream_offset=DOWNSTREAM_OFFSET)
    case_1 = classify_and_get_case_1_mutations(df, vcf_id, start_idx, end_idx, out_dir)
    case_1 = case_1[["id", "wt_seq", "mut_seq"]]
    
    # Release memory
    del df
    gc.collect()
    
    # Get model and device
    global MODEL
    device = next(MODEL.parameters()).device
    
    # Load miRNA sequences
    mirnas = pd.read_csv(MIRNA_CSV).set_index('mirna_accession')['sequence'].to_dict()

    # Make predictions and post-process
    df = predict_binding_to_df(MODEL, device, case_1, mirnas, S_MATRIX, BATCH_SIZE)
    df = post_process_predictions(df, DEC_SURF, FILTER_THRESH)

    return df


def process_chunk(chunk, start_idx, end_idx, out_dir, vcf_id):
    """Process a chunk of data using the worker's model instance"""
    global MODEL
    
    # Check if MODEL is initialized
    if MODEL is None:
        raise RuntimeError("Worker not properly initialized. MODEL is None.")
    
    # Get process ID for logging
    pid = multiprocessing.current_process().name
    print(f"Process {pid} processing chunk {start_idx}-{end_idx}")
    
    # Process chunk and save results
    result = analysis_pipeline(chunk, start_idx, end_idx, out_dir, vcf_id)
    result.to_csv(os.path.join(out_dir, f'result_{start_idx}_{end_idx}.csv'), index=False)
    
    print(f"Process {pid} completed chunk {start_idx}-{end_idx}")
    return start_idx, end_idx


def run_pipeline_single(vcf_path, chunksize, out_dir, vcf_id):
    # Read first chunk to determine chrom
    first_chunk = next(pd.read_csv(vcf_path, chunksize=1, sep="\t", header=None, names=VCF_COLNAMES))
    chrom = first_chunk['chr'].iloc[0]
    
    # Pre-load chrom
    from scripts.sequence_utils import load_chrom
    load_chrom(chrom)
    print(f"Single-chrom mode: Pre-loaded chrom {chrom} for entire pipeline")
    
    # Create process pool with initialized workers
    with ProcessPoolExecutor(
        max_workers=WORKERS,
        initializer=init_worker,
        initargs=(MODEL_PATH, "cuda" if torch.cuda.is_available() else "cpu", model_cfg)
    ) as executor:
        print(f"Starting multiprocessing pool with {WORKERS} workers")
        
        jobs = []
        start_idx = 0
        
        # Process chunks
        for chunk in pd.read_csv(vcf_path, chunksize=chunksize, sep="\t", header=None, names=VCF_COLNAMES):
            end_idx = start_idx + len(chunk) - 1
            job = executor.submit(
                process_chunk, chunk, start_idx, end_idx, out_dir, vcf_id)
            jobs.append(job)
            start_idx = end_idx + 1
        
        # Process results
        for job in as_completed(jobs):
            try:
                start_idx, end_idx = job.result()
                print(f"Completed chunk {start_idx}-{end_idx}")
            except Exception as e:
                print(f"Error processing chunk: {e}")


def run_pipeline(vcf_path, chunksize, out_dir, vcf_id):
    # Create process pool with initialized workers
    with ProcessPoolExecutor(
        max_workers=WORKERS,
        initializer=init_worker,
        initargs=(MODEL_PATH, "cuda" if torch.cuda.is_available() else "cpu", model_cfg)
    ) as executor:
        print(f"Starting multiprocessing pool with {WORKERS} workers")
        
        jobs = []
        start_idx = 0
        
        # Process chunks
        for chunk in pd.read_csv(vcf_path, chunksize=chunksize, sep="\t", header=None, names=VCF_COLNAMES):
            end_idx = start_idx + len(chunk) - 1
            job = executor.submit(
                process_chunk, chunk, start_idx, end_idx, out_dir, vcf_id)
            jobs.append(job)
            start_idx = end_idx + 1
        
        # Process results
        for job in as_completed(jobs):
            try:
                start_idx, end_idx = job.result()
                print(f"Completed chunk {start_idx}-{end_idx}")
            except Exception as e:
                print(f"Error processing chunk: {e}")


def compile_results(out_dir, out_file): 
    """Stitch multiple result CSV files into one and remove originals."""
    try:
        # Get and sort CSV files
        csv_files = [f for f in os.listdir(out_dir) 
                    if f.endswith('.csv') and f.startswith('result_')]
        csv_files.sort()
        
        if not csv_files:
            print("No matching CSV files found.")
            return

        out_path = os.path.join(out_dir, out_file)
        removed = []

        with open(out_path, 'w', newline='') as outfile:
            writer = csv.writer(outfile)
            has_header = False

            for fname in csv_files:
                fpath = os.path.join(out_dir, fname)
                with open(fpath, 'r', newline='') as infile:
                    reader = csv.reader(infile)
                    if not has_header:
                        writer.writerow(next(reader))
                        has_header = True
                    else:
                        next(reader)  # Skip header
                    writer.writerows(reader)
                
                os.remove(fpath)
                removed.append(fname)
        
        print(f"Stitched {len(removed)} files into {out_file}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
