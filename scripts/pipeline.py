import os
import csv
import gc
from typing import List
import pandas as pd
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch
from scripts.functions import *
from scripts.targetnet import TargetNet

from scripts.config import *

# Global model variable that will be initialized in each worker process
MODEL = None

def init_worker(model_path, device_str, model_cfg):
    """Initialize the model in each worker process"""
    global MODEL
    
    # Set device for this process
    device = torch.device(device_str)
    
    # Create a new model instance for this process
    model = TargetNet(model_cfg, with_esa=True, dropout_rate=0.5)
    MODEL = model.to(device)
    
    # Load model weights
    MODEL.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
    MODEL.eval()

def analysis_pipeline(df, start_index, end_index, output_dir, vcf_id):

    rnaduplex_output_file = os.path.join(
        output_dir, f"rnad_{vcf_id}_{start_index}_{end_index}.csv")

    invalid_rows_report_file = os.path.join(
        output_dir, f"invalid_rows_{vcf_id}.csv")
    fasta_output_file = os.path.join(
        output_dir, f"fasta_{vcf_id}_{start_index}_{end_index}.fa")

    df['id'] = df['id'].astype(str)
    df['id'] += '_' + df['chr'].astype(str) + '_' + df['pos'].astype(
        str) + '_' + df['ref'] + '_' + df['alt']

    # Step 1: Data Preprocessing
    # df = validate_ref_nucleotides(df, invalid_rows_report_file)
    df = generate_is_mirna_column(df, grch=37)
    df = add_sequence_columns(df, upstream_offset=UPSTREAM_OFFSET, downstream_offset=DOWNSTREAM_OFFSET)

    # Step 2: Data Processing
    case_1 = classify_and_get_case_1_mutations(
        df, vcf_id, start_index, end_index, output_dir)
    
    # gc
    del df
    gc.collect()
    
    # Use the global MODEL initialized for this process
    global MODEL
    
    # Get the device this model is on
    model_device = next(MODEL.parameters()).device
    
    # Load miRNA sequences
    mirna_dict = pd.read_csv(MIRNA_CSV).set_index('mirna_accession')['sequence'].to_dict()

    # Make predictions using the process-local model
    df = predict_binding_to_df(MODEL, model_device, case_1, mirna_dict, SCORE_MATRIX, BATCH_SIZE)

    df = post_process_predictions(df, DECISION_SURFACE, FILTER_THRESHOLD)

    # # Step 3: Prediction Preprocessing
    # df = process_rnaduplex_output(rnaduplex_output_file)
    # df = generate_mirna_conservation_column(df)
    # df.drop("mirna_accession", axis=1, inplace=True)
    # df = split_mutation_ids(df)
    # df['is_mutated'] = df['is_mutated'].isin(['mt', 'mut'])
    # df = add_sequence_columns(df)
    # df['mrna_sequence'] = df['wt_seq'].where(
    #     ~df['is_mutated'], df['mut_seq'])
    # column_names = ["chr", "pos", "ref", "alt", "upstream_seq",
    #                 "downstream_seq", "wt_seq", "mut_seq", "is_mutated"]
    # df.drop(columns=column_names, inplace=True)
    
    
    # df = generate_mre_sequence_column(df)
    
    # # add a mask that checks if the mutation is in the MRE region
    # mask_if_mutation_in_mre = (df.mrna_start < 32) & (df.mrna_end > 30)
    # df["is_mutation_in_mre"] = mask_if_mutation_in_mre
    # df.drop(columns=["mrna_start", "mrna_end",
    #         "mre_start", "mre_end"], inplace=True)
    
    # df = generate_local_au_content_column(df)
    # df.drop("mrna_sequence", axis=1, inplace=True)
    
    # df = generate_ta_sps_columns(df)
    # df = generate_alignment_string_from_dot_bracket(df)
    # df.drop(columns=["mirna_start", "mirna_end",
    #         "mirna_sequence"], inplace=True)
    
    # df = generate_match_count_columns(df)
    # df = generate_important_sites_column(df)
    # df = generate_seed_type_columns(df)
    # df.drop("alignment_string", axis=1, inplace=True)
    
    # df = generate_mre_au_content_column(df)
    # df.drop("mre_region", axis=1, inplace=True)

    # # Step 4: Prediction
    # df, id_array, binary_array = reorder_columns_for_prediction(df)

    # predictions = make_predictions_with_xgb(df)
    
    # # gc
    # del df
    # gc.collect()
    
    # df = create_results_df(id_array, predictions, binary_array,
    #                         filter_range=FILTER_THRESHOLD)
    
    # df.drop(columns=["binary_array"], inplace=True)

    return df


def analysis_pipeline_single(df, start_index, end_index, output_dir, vcf_id):


    invalid_rows_report_file = os.path.join(
        output_dir, f"invalid_rows_{vcf_id}.csv")
  

    # Optimize memory usage by applying appropriate data types
    # Convert objects to categorical where appropriate
    if 'chr' in df.columns and df['chr'].dtype == 'object':
        df['chr'] = df['chr'].astype('category')
    
    # Create unique IDs
    df['id'] = df['id'].astype(str)
    df['id'] += '_' + df['chr'].astype(str) + '_' + df['pos'].astype(
        str) + '_' + df['ref'] + '_' + df['alt']

    # Load the chromosome if all data is from a single chromosome
    unique_chroms = df['chr'].unique()
    if len(unique_chroms) == 1:
        from scripts.sequence_utils import load_chromosome
        load_chromosome(unique_chroms[0])
        print(f"Pre-loaded chromosome {unique_chroms[0]} for faster access")

    # Step 1: Data Preprocessing
    # df = validate_ref_nucleotides_single(df, invalid_rows_report_file)
    df = generate_is_mirna_column_single(df, grch=37)
    df = add_sequence_columns_single(df, upstream_offset=UPSTREAM_OFFSET, downstream_offset=DOWNSTREAM_OFFSET)

    # Step 2: Data Processing
    case_1 = classify_and_get_case_1_mutations(
        df, vcf_id, start_index, end_index, output_dir)
    
    # Optimize memory usage - only keep essential columns
    case_1 = case_1[["id", "wt_seq", "mut_seq"]]
    
    # Release memory from original dataframe
    del df
    gc.collect()
    
    # Use the global MODEL initialized for this process
    global MODEL
    
    # Get the device this model is on
    model_device = next(MODEL.parameters()).device
    
    # Load miRNA sequences
    mirna_dict = pd.read_csv(MIRNA_CSV).set_index('mirna_accession')['sequence'].to_dict()

    # Make predictions using the process-local model
    df = predict_binding_to_df(MODEL, model_device, case_1, mirna_dict, SCORE_MATRIX, BATCH_SIZE)

    # Post-process results
    df = post_process_predictions(df, DECISION_SURFACE, FILTER_THRESHOLD)

    # # Step 3: Prediction Preprocessing
    # df = process_rnaduplex_output(rnaduplex_output_file)
    # df = generate_mirna_conservation_column(df)
    # df.drop("mirna_accession", axis=1, inplace=True)
    # df = split_mutation_ids(df)
    # df['is_mutated'] = df['is_mutated'].isin(['mt', 'mut'])
    # df = add_sequence_columns(df)
    # df['mrna_sequence'] = df['wt_seq'].where(
    #     ~df['is_mutated'], df['mut_seq'])
    # column_names = ["chr", "pos", "ref", "alt", "upstream_seq",
    #                 "downstream_seq", "wt_seq", "mut_seq", "is_mutated"]
    # df.drop(columns=column_names, inplace=True)
    
    
    # df = generate_mre_sequence_column(df)
    
    # # add a mask that checks if the mutation is in the MRE region
    # mask_if_mutation_in_mre = (df.mrna_start < 32) & (df.mrna_end > 30)
    # df["is_mutation_in_mre"] = mask_if_mutation_in_mre
    # df.drop(columns=["mrna_start", "mrna_end",
    #         "mre_start", "mre_end"], inplace=True)
    
    # df = generate_local_au_content_column(df)
    # df.drop("mrna_sequence", axis=1, inplace=True)
    
    # df = generate_ta_sps_columns(df)
    # df = generate_alignment_string_from_dot_bracket(df)
    # df.drop(columns=["mirna_start", "mirna_end",
    #         "mirna_sequence"], inplace=True)
    
    # df = generate_match_count_columns(df)
    # df = generate_important_sites_column(df)
    # df = generate_seed_type_columns(df)
    # df.drop("alignment_string", axis=1, inplace=True)
    
    # df = generate_mre_au_content_column(df)
    # df.drop("mre_region", axis=1, inplace=True)

    # # Step 4: Prediction
    # df, id_array, binary_array = reorder_columns_for_prediction(df)

    # predictions = make_predictions_with_xgb(df)
    
    # # gc
    # del df
    # gc.collect()
    
    # df = create_results_df(id_array, predictions, binary_array,
    #                         filter_range=FILTER_THRESHOLD)
    
    # df.drop(columns=["binary_array"], inplace=True)

    return df


def process_chunk(chunk, start_index, end_index, output_dir, vcf_id, score_matrix, decision_surface, filter_threshold, batch_size):
    """Process a chunk of data using the worker's model instance"""
    global MODEL
    
    # Check if MODEL is initialized
    if MODEL is None:
        raise RuntimeError("Worker not properly initialized. MODEL is None.")
    
    # Get the worker's process ID for logging
    process_id = multiprocessing.current_process().name
    print(f"Process {process_id} processing chunk {start_index}-{end_index}")
    
    # Pass the global MODEL to the analysis pipeline
    result = analysis_pipeline(
        chunk, start_index, end_index, output_dir, vcf_id)
    
    # Write the result to a CSV file in the output directory
    result_file = os.path.join(
        output_dir, f'result_{start_index}_{end_index}.csv')
    result.to_csv(result_file, index=False)
    
    print(f"Process {process_id} completed chunk {start_index}-{end_index}")
    return start_index, end_index


def run_pipeline_single(vcf_full_path, chunksize, output_dir, vcf_id):

    # Read first chunk to determine which chromosome we're working with
    first_chunk = next(pd.read_csv(vcf_full_path, chunksize=1, sep="\t", header=None, names=VCF_COLNAMES))
    chromosome = first_chunk['chr'].iloc[0]
    
    # Pre-load the chromosome into memory
    from scripts.sequence_utils import load_chromosome
    load_chromosome(chromosome)
    print(f"Single-chromosome mode: Pre-loaded chromosome {chromosome} for entire pipeline")
    
    # Create a process pool with initialized workers
    # Each worker will have its own model instance
    with ProcessPoolExecutor(
        max_workers=WORKERS,
        initializer=init_worker,
        initargs=(MODEL_PATH, "cuda" if torch.cuda.is_available() else "cpu", model_cfg)
    ) as executor:
        
        print(f"Starting multiprocessing pool with {WORKERS} workers")
        
        # Submit chunk processing jobs to the executor



        futures = []
        start_index = 0
        
        # Reset reader to start from beginning
        for chunk in pd.read_csv(vcf_full_path, chunksize=chunksize, sep="\t", header=None, names=VCF_COLNAMES):
            end_index = start_index + len(chunk) - 1
            future = executor.submit(
                process_chunk, 
                chunk, 
                start_index, 
                end_index, 
                output_dir, 
                vcf_id,
                SCORE_MATRIX,
                DECISION_SURFACE,
                FILTER_THRESHOLD,
                BATCH_SIZE
            )
            futures.append(future)
            start_index = end_index + 1
        
        # Process results as they complete
        for future in as_completed(futures):
            try:
                start_index, end_index = future.result()
                print(f"Completed chunk {start_index}-{end_index}")
            except Exception as e:
                print(f"Error processing chunk: {e}")


def run_pipeline(vcf_full_path, chunksize, output_dir, vcf_id):

    # Create a process pool with initialized workers
    # Each worker will have its own model instance
    with ProcessPoolExecutor(
        max_workers=WORKERS,
        initializer=init_worker,
        initargs=(MODEL_PATH, "cuda" if torch.cuda.is_available() else "cpu", model_cfg)
    ) as executor:
        
        print(f"Starting multiprocessing pool with {WORKERS} workers")
        
        # Submit chunk processing jobs to the executor



        futures = []
        start_index = 0
        
        # Reset reader to start from beginning
        for chunk in pd.read_csv(vcf_full_path, chunksize=chunksize, sep="\t", header=None, names=VCF_COLNAMES):
            end_index = start_index + len(chunk) - 1
            future = executor.submit(
                process_chunk, 
                chunk, 
                start_index, 
                end_index, 
                output_dir, 
                vcf_id,
                SCORE_MATRIX,
                DECISION_SURFACE,
                FILTER_THRESHOLD,
                BATCH_SIZE
            )
            futures.append(future)
            start_index = end_index + 1
        
        # Process results as they complete
        for future in as_completed(futures):
            try:
                start_index, end_index = future.result()
                print(f"Completed chunk {start_index}-{end_index}")
            except Exception as e:
                print(f"Error processing chunk: {e}")


def compile_results(output_dir: str, final_output_filename: str) -> None:
    """
    Stitch multiple result CSV files into one and remove the original files.
    
    Args:
        output_dir (str): Directory containing the CSV files.
        final_output_filename (str): Name of the final stitched file.
    """
    try:
        # Get and filter CSV files
        csv_files = [f for f in os.listdir(output_dir) 
                     if f.endswith('.csv') and f.startswith('result_')]
        csv_files.sort()  # Ensure consistent ordering
        
        if not csv_files:
            print("No matching CSV files found.")
            return

        final_output_path = os.path.join(output_dir, final_output_filename)
        removed_files: List[str] = []

        with open(final_output_path, 'w', newline='') as outfile:
            writer = csv.writer(outfile)
            header_written = False

            for filename in csv_files:
                file_path = os.path.join(output_dir, filename)
                with open(file_path, 'r', newline='') as infile:
                    reader = csv.reader(infile)
                    if not header_written:
                        header = next(reader)
                        writer.writerow(header)
                        header_written = True
                    else:
                        next(reader)  # Skip header in subsequent files
                    writer.writerows(reader)
                
                # Add file removal here
                os.remove(file_path)
                removed_files.append(filename)
        
        print(f"Stitched {len(removed_files)} files into {final_output_filename}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
