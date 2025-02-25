import logging
import numpy as np
import pandas as pd
from scripts.sequence_utils import *
from scripts.config import MIRNA_COORDS_DIR
from Bio import pairwise2
import torch
import torch.nn as nn

def validate_ref_nucleotides(df, report_path, verbose=False):

    if verbose:
        logging.info("Validating reference nucleotides...")

    # Add ref_len and alt_len columns
    df["ref_len"] = df["ref"].str.len()
    df["alt_len"] = df["alt"].str.len()

    # For rows where the reference length (ref_len) is greater than 1:
    # - Use the get_nucleotides_in_interval function to fetch the nucleotides
    #   in the interval [pos, pos + ref_len - 1] for the given chromosome (chr)
    df['nuc_at_pos'] = np.where(
        df['ref_len'] > 1,
        df.apply(lambda x: get_seq_interval(
            x['chr'], x['pos'], x["pos"] + x["ref_len"] - 1), axis=1),
        # For rows where the reference length (ref_len) is 1:
        # - Use the get_nucleotide_at_position function to fetch the nucleotide
        #   at the given position (pos) for the given chromosome (chr)
        np.where(
            df['ref_len'] == 1,
            df.apply(lambda x: get_nuc_at_pos(
                x['chr'], x['pos']), axis=1),
            # For all other cases, set the value to an empty string
            ""
        )
    )

    # Check if ref matches nucleotide_at_position
    mask = df['ref'] != df['nuc_at_pos']

    # Isolate invalid rows
    invalid_rows = df[mask]

    if not invalid_rows.empty:
        if verbose:
            logging.warning(
                f"Writing {len(invalid_rows)} invalid rows to {report_path}")

        # Check if the file exists
        file_exists = os.path.isfile(report_path)

        # Open the file in append mode ('a') or write mode ('w')
        mode = 'a' if file_exists else 'w'
        with open(report_path, mode) as f:
            # If the file is new, write the header
            if not file_exists:
                f.write("id\n")

            # Write each invalid row to the file
            for _, row in invalid_rows.iterrows():
                f.write(f"{row['id']}\n")

    return df[~mask].drop("nuc_at_pos", axis=1)



def validate_ref_nucleotides_single(df, report_path, verbose=False):
    """
    Validates that reference nucleotides match the reference genome.
    Optimized for single-chromosome processing with vectorized operations.
    """
    if verbose:
        logging.info("Validating reference nucleotides...")

    # Add ref_len and alt_len columns
    df["ref_len"] = df["ref"].str.len()
    df["alt_len"] = df["alt"].str.len()

    # Ensure chromosome is loaded
    from scripts.sequence_utils import CHROMOSOME_SEQUENCE, CHROMOSOME_ID, load_chrom
    
    # Get chromosome from first row (assuming single chromosome file)
    chromosome = str(df['chr'].iloc[0])
    
    # Load the chromosome if not already loaded
    if CHROMOSOME_ID != chromosome or CHROMOSOME_SEQUENCE is None:
        load_chrom(chromosome)
    
    # Define vectorized functions for nucleotide retrieval
    def get_ref_bases(row):
        # Convert from 1-based to 0-based
        start_idx = row['pos'] - 1
        end_idx = start_idx + row['ref_len']
        return CHROMOSOME_SEQUENCE[start_idx:end_idx]
        
    # Apply function to all rows
    df['nuc_at_pos'] = df.apply(get_ref_bases, axis=1)
    
    # Check if ref matches nucleotide_at_position
    mask = df['ref'] != df['nuc_at_pos']

    # Isolate invalid rows
    invalid_rows = df[mask]

    if not invalid_rows.empty:
        if verbose:
            logging.warning(
                f"Writing {len(invalid_rows)} invalid rows to {report_path}")

        # Check if the file exists
        file_exists = os.path.isfile(report_path)

        # Write invalid rows efficiently
        with open(report_path, 'a' if file_exists else 'w') as f:
            # If the file is new, write the header
            if not file_exists:
                f.write("id\n")
            
            # Write all IDs at once instead of row by row
            f.write('\n'.join(invalid_rows['id']) + '\n')

    # Return only valid rows, dropping the nucleotide validation column
    return df[~mask].drop("nuc_at_pos", axis=1)


def generate_is_mirna_column(df, grch):

    # Construct the miRNA coordinates file path
    mirna_coords_file = os.path.join(
        MIRNA_COORDS_DIR, f"grch{grch}_coordinates.csv")

    # Load miRNA coordinates
    coords = pd.read_csv(mirna_coords_file)

    # Initialize new columns
    df['is_mirna'] = 0
    df['mirna_accession'] = None
    df["pos"] = df["pos"].astype(int)
    

    # Iterate over each mutation in the mutations dataframe
    for index, row in df.iterrows():
        mutation_chr = row['chr']
        mutation_start = row['pos']

        # Find matching miRNAs
        matching_rnas = coords.loc[(coords['chr'] == mutation_chr) &
                                   (coords['start'] <= mutation_start) &
                                   (coords['end'] >= mutation_start)]

        if not matching_rnas.empty:
            # Update the 'is_mirna' and 'mirna_accession' columns
            df.at[index, 'is_mirna'] = 1
            df.at[index, 'mirna_accession'] = matching_rnas['mirna_accession'].values[0]

    return df

def generate_is_mirna_column_single(df, grch):
    """
    Vectorized implementation to check if mutations are in miRNA regions.
    This version avoids iterating over rows using pandas' optimized operations.
    """
    # Construct the miRNA coordinates file path
    mirna_coords_file = os.path.join(
        MIRNA_COORDS_DIR, f"grch{grch}_coordinates.csv")

    # Load miRNA coordinates
    coords = pd.read_csv(mirna_coords_file)
    
    # Ensure position is integer
    df["pos"] = df["pos"].astype(int)
    
    # Initialize new columns
    df['is_mirna'] = 0
    df['mirna_accession'] = None
    
    # For each unique chromosome in our data
    for chrom in df['chr'].unique():
        # Get all mutations on this chromosome
        chrom_df = df[df['chr'] == chrom]
        
        if chrom_df.empty:
            continue
            
        # Get all miRNAs on this chromosome
        chrom_coords = coords[coords['chr'] == chrom]
        
        if chrom_coords.empty:
            continue
            
        # For each position in this chromosome's mutations
        for idx, row in chrom_df.iterrows():
            pos = row['pos']
            
            # Find matching miRNAs (vectorized operation)
            matching = chrom_coords[(chrom_coords['start'] <= pos) & 
                                   (chrom_coords['end'] >= pos)]
            
            if not matching.empty:
                # Update values
                df.at[idx, 'is_mirna'] = 1
                df.at[idx, 'mirna_accession'] = matching['mirna_accession'].values[0]
    
    return df




def add_sequence_columns(df, upstream_offset=29, downstream_offset=10):
    grouped = df.groupby(['chr', 'pos'])

    def apply_func(group):
        group['upstream_seq'] = get_upstream(
            group['chr'].iloc[0], group['pos'].iloc[0], upstream_offset)
        group['downstream_seq'] = get_downstream(
            group['chr'].iloc[0], group['pos'].iloc[0], group['ref'].iloc[0], downstream_offset)
        group['wt_seq'] = group['upstream_seq'] + \
            group['ref'] + group['downstream_seq']
        group['mut_seq'] = group['upstream_seq'] + \
            group['alt'] + group['downstream_seq']
        return group

    df = grouped.apply(apply_func)

    return df.reset_index(drop=True)


def add_sequence_columns_single(df, upstream_offset=29, downstream_offset=10):
    """
    Add sequence columns to the dataframe using optimized chromosome caching.
    
    This version:
    1. Detects and loads the chromosome if all mutations are on one chromosome
    2. Uses vectorized operations where possible instead of groupby.apply
    3. Uses the optimized get_flanking_sequences function for efficiency
    """
    # Check if all mutations are on the same chromosome for optimization
    unique_chroms = df['chr'].unique()
    if len(unique_chroms) == 1:
        # Single chromosome optimization - preload it
        from scripts.sequence_utils import load_chrom
        load_chrom(unique_chroms[0])
    
    # Process each unique chr,pos combination
    result_dfs = []
    
    for (chrom, pos), group in df.groupby(['chr', 'pos'], observed=True):
        # Get sequences using optimized function that fetches both at once
        from scripts.sequence_utils import get_flanks
        upstream_seq, downstream_seq = get_flanks(
            chrom, pos, group['ref'].iloc[0], 
            upstream_offset=upstream_offset, 
            downstream_offset=downstream_offset
        )
        
        # Create a copy of the group to avoid SettingWithCopyWarning
        group_copy = group.copy()
        
        # Set sequence columns
        group_copy['upstream_seq'] = upstream_seq
        group_copy['downstream_seq'] = downstream_seq
        
        # Create wt and mut sequences directly
        group_copy['wt_seq'] = upstream_seq + group_copy['ref'] + downstream_seq
        group_copy['mut_seq'] = upstream_seq + group_copy['alt'] + downstream_seq
        
        result_dfs.append(group_copy)
    
    # Combine all processed groups
    return pd.concat(result_dfs).reset_index(drop=True)


def classify_and_get_case_1_mutations(df, vcf_id, start, end, output_dir):
    """
    Classifies mutations into case 1 and case 2, saves case 2 mutations to disk,
    and returns the case 1 mutations.

    Args:
        df (pandas.DataFrame): DataFrame containing mutation data.
        vcf_id (str): ID of the VCF file.
        start (int): Start position of the region.
        end (int): End position of the region.
        output_dir (str): Path to the output directory.

    Returns:
        pandas.DataFrame: DataFrame containing case 1 mutations.
    """
    # Classify case 1 and case 2 mutations
    case_1 = df[df.is_mirna == 0][["id", "wt_seq", "mut_seq"]]
    case_2 = df[df.is_mirna == 1][["id", "wt_seq", "mut_seq"]]
    
    


    if not case_2.empty:
        # Save case 2 mutations to disk if any exist
        case_2_file = os.path.join(output_dir, f"{vcf_id}_{start}_{end}_case_2.csv")
        case_2.to_csv(case_2_file, index=False)
        print(f"case 2 mutations: {len(case_2)}")
        

    return case_1


def generate_mrna_mirna_pairs(mrna_df, mirna_dict, seq_type='wt_seq'):
    """
    Generate all possible combinations of mRNA and miRNA sequences.
    
    Parameters:
    mrna_df: DataFrame with columns ['id', 'wt_seq', 'mut_seq']
    mirna_dict: Dictionary with mirna_accession as keys and sequences as values
    seq_type: str, either 'wt_seq' or 'mut_seq'
    
    Yields:
    tuple: (f"{mrna_id}_{mirna_acc}_{wt/mut}", mrna_sequence, mirna_sequence)
    """
    if seq_type not in ['wt_seq', 'mut_seq']:
        raise ValueError("seq_type must be either 'wt_seq' or 'mut_seq'")
    
    # Determine suffix based on seq_type
    suffix = "-wt" if seq_type == 'wt_seq' else "-mut"
        
    for _, mrna_row in mrna_df.iterrows():
        for mirna_acc, mirna_seq in mirna_dict.items():
            combined_id = f"{mrna_row['id']}-{mirna_acc}{suffix}"
            yield (
                combined_id,
                mrna_row[seq_type],
                mirna_seq
            )
            

### neural net

def extended_seed_alignment(mi_seq, cts_r_seq, score_matrix):
    """ extended seed alignment """
    alignment = pairwise2.align.globaldx(mi_seq[:10], cts_r_seq[5:15], score_matrix, one_alignment_only=True)[0]
    mi_esa = alignment[0]
    cts_r_esa = alignment[1]
    esa_score = alignment[2]
    return mi_esa, cts_r_esa, esa_score


def encode_RNA(mirna_seq, mirna_esa, cts_rev_seq, cts_rev_esa):
    """ one-hot encoder for RNA sequences with extended seed alignments """
    chars = {"A":0, "C":1, "G":2, "U":3, "-":4}
    x = np.zeros((len(chars) * 2, 50), dtype=np.float32)
    
    # Encode miRNA with ESA
    for i in range(len(mirna_esa)):
        x[chars[mirna_esa[i]], 5 + i] = 1
    for i in range(10, len(mirna_seq)):
        x[chars[mirna_seq[i]], 5 + i - 10 + len(mirna_esa)] = 1
    
    # Encode mRNA with ESA
    for i in range(5):
        x[chars[cts_rev_seq[i]] + len(chars), i] = 1
    for i in range(len(cts_rev_esa)):
        x[chars[cts_rev_esa[i]] + len(chars), i + 5] = 1
    for i in range(15, len(cts_rev_seq)):
        x[chars[cts_rev_seq[i]] + len(chars), i + 5 - 15 + len(cts_rev_esa)] = 1
    
    return x





# This function is no longer used as we've optimized the process in process_sequence_pairs
# Keeping it for backward compatibility
def prepare_sequence_for_model(mrna_seq, mirna_seq, score_matrix):
    """Prepare RNA sequences for model input"""
    # Convert to RNA format
    mirna_seq = mirna_seq.upper().replace("T", "U")
    mrna_seq = mrna_seq.upper().replace("T", "U")
    mrna_rev_seq = mrna_seq[::-1]
    
    # Alignment
    mirna_esa, mrna_rev_esa, _ = extended_seed_alignment(mirna_seq, mrna_rev_seq, score_matrix)
    
    # Encode
    x = encode_RNA(mirna_seq, mirna_esa, mrna_rev_seq, mrna_rev_esa)
    return torch.from_numpy(x).unsqueeze(0)

def get_model_predictions_batch(model: nn.Module, sequence_tensors: torch.Tensor, device: str) -> torch.Tensor:
    """Get predictions from model for a batch of sequences"""
    sequence_tensors = sequence_tensors.to(device)
    with torch.no_grad():
        preds = model(sequence_tensors)
        return torch.sigmoid(preds).cpu()

def prepare_mirna_for_alignment(mirna_seq):
    """Pre-process miRNA sequence for alignment"""
    return mirna_seq.upper().replace("T", "U")

def prepare_mrna_for_alignment(mrna_seq):
    """Pre-process mRNA sequence for alignment"""
    return mrna_seq.upper().replace("T", "U")[::-1]

def process_sequence_pairs(model, sequence_generator, device, score_matrix, batch_size=64):
    """Process sequence pairs in batches and return predictions with miRNA caching"""
    predictions = {}
    batch_ids = []
    batch_tensors = []
    
    # Cache for miRNA data to avoid redundant computations
    mirna_cache = {}
    
    for combined_id, mrna_seq, mirna_seq in sequence_generator:
        # Check if miRNA is in cache
        if mirna_seq not in mirna_cache:
            # Process miRNA and cache the intermediate representations
            processed_mirna = prepare_mirna_for_alignment(mirna_seq)
            mirna_cache[mirna_seq] = processed_mirna
        else:
            # Use cached miRNA
            processed_mirna = mirna_cache[mirna_seq]
            
        # Process mRNA (always needed as it's different for each pair)
        processed_mrna = prepare_mrna_for_alignment(mrna_seq)
        
        # Alignment and encoding
        mirna_esa, mrna_rev_esa, _ = extended_seed_alignment(processed_mirna, processed_mrna, score_matrix)
        x = encode_RNA(processed_mirna, mirna_esa, processed_mrna, mrna_rev_esa)
        tensor = torch.from_numpy(x).unsqueeze(0)
        
        batch_ids.append(combined_id)
        batch_tensors.append(tensor)
        
        # Process batch when it reaches the specified size
        if len(batch_tensors) >= batch_size:
            # Stack tensors into a single batch
            stacked_tensors = torch.cat(batch_tensors, dim=0)
            
            # Get predictions for the entire batch
            batch_preds = get_model_predictions_batch(model, stacked_tensors, device)
            
            # Store predictions
            for i, combined_id in enumerate(batch_ids):
                predictions[combined_id] = {
                    'id': combined_id,
                    'pred': batch_preds[i].item()
                }
            
            # Clear batch
            batch_ids = []
            batch_tensors = []
    
    # Process any remaining sequences
    if batch_tensors:
        stacked_tensors = torch.cat(batch_tensors, dim=0)
        batch_preds = get_model_predictions_batch(model, stacked_tensors, device)
        
        for i, combined_id in enumerate(batch_ids):
            predictions[combined_id] = {
                'id': combined_id,
                'pred': batch_preds[i].item()
            }
    
    return predictions

def create_comparison_results(wt_preds: dict, mut_preds: dict) -> pd.DataFrame:
    """Create DataFrame with WT vs MUT comparisons"""
    results = []
    for wt_id in wt_preds:
        # Convert WT ID to corresponding MUT ID
        mut_id = wt_id.replace('-wt', '-mut')
        if mut_id in mut_preds:
            results.append({
                'id': wt_id[:-3],  # remove '-wt'
                'wt_prediction': round(wt_preds[wt_id]['pred'], 3),
                'mut_prediction': round(mut_preds[mut_id]['pred'], 3),
                'pred_difference': round(mut_preds[mut_id]['pred'] - wt_preds[wt_id]['pred'], 3)
            })
    return pd.DataFrame(results)


def predict_binding_to_df(model, device, df, mirna_dict, score_matrix, batch_size=64):
    """Main function to orchestrate the prediction process"""
    model.eval()
    
    # Generate predictions for WT and MUT
    wt_generator = generate_mrna_mirna_pairs(df, mirna_dict, 'wt_seq')
    mut_generator = generate_mrna_mirna_pairs(df, mirna_dict, 'mut_seq')
    
    wt_predictions = process_sequence_pairs(model, wt_generator, device, score_matrix, batch_size)
    mut_predictions = process_sequence_pairs(model, mut_generator, device, score_matrix, batch_size)
    
    # Create final results
    return create_comparison_results(wt_predictions, mut_predictions)


def post_process_predictions(df, decision_surface, min_difference):

    """
    Filter predictions based on decision surface and minimum difference criteria.
    Fully vectorized implementation.
    
    Args:
        df: DataFrame with columns ['wt_prediction', 'mut_prediction', 'pred_difference']
        decision_surface: float, threshold for binary classification (0-1)
        min_difference: float, minimum absolute difference required between predictions
    
    Returns:
        Filtered DataFrame where predictions cross the decision surface and 
        have sufficient difference
    """
    return df[
        ((df['wt_prediction'] - decision_surface) * 
         (df['mut_prediction'] - decision_surface) < 0) & 
        (abs(df['pred_difference']) >= min_difference)
    ].copy()