import os
import csv
import gc
from typing import List
import pandas as pd
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from scripts.functions import *

from scripts.config import *


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
    df = validate_ref_nucleotides(df, invalid_rows_report_file)
    df = generate_is_mirna_column(df, grch=37)
    df = add_sequence_columns(df, upstream_offset=UPSTREAM_OFFSET, downstream_offset=DOWNSTREAM_OFFSET)

    # Step 2: Data Processing
    case_1 = classify_and_get_case_1_mutations(
        df, vcf_id, start_index, end_index, output_dir)
    
    # gc
    del df
    gc.collect()
    
    mirna_dict = pd.read_csv(MIRNA_CSV).set_index('mirna_accession')['sequence'].to_dict()

    df = predict_binding_to_df(MODEL, DEVICE, case_1, mirna_dict, SCORE_MATRIX)

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


def process_chunk(chunk, start_index, end_index, output_dir, vcf_id):

    result = analysis_pipeline(
        chunk, start_index, end_index, output_dir, vcf_id)

    # Write the result to a CSV file in the output directory
    result_file = os.path.join(
        output_dir, f'result_{start_index}_{end_index}.csv')
    result.to_csv(result_file, index=False)

    return start_index, end_index


def run_pipeline(vcf_full_path, chunksize, output_dir, vcf_id):

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:

        futures = []
        start_index = 0
        for chunk in pd.read_csv(vcf_full_path, chunksize=chunksize, sep="\t", header=None, names=VCF_COLNAMES):
            end_index = start_index + len(chunk) - 1
            future = executor.submit(
                process_chunk, chunk, start_index, end_index, output_dir, vcf_id)
            futures.append(future)
            start_index = end_index + 1

        for future in as_completed(futures):
            start_index, end_index = future.result()
