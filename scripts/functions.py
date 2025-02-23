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
        df.apply(lambda x: get_nucleotides_in_interval(
            x['chr'], x['pos'], x["pos"] + x["ref_len"] - 1), axis=1),
        # For rows where the reference length (ref_len) is 1:
        # - Use the get_nucleotide_at_position function to fetch the nucleotide
        #   at the given position (pos) for the given chromosome (chr)
        np.where(
            df['ref_len'] == 1,
            df.apply(lambda x: get_nucleotide_at_position(
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


def add_sequence_columns(df, upstream_offset=29, downstream_offset=10):
    grouped = df.groupby(['chr', 'pos'])

    def apply_func(group):
        group['upstream_seq'] = get_upstream_sequence(
            group['chr'].iloc[0], group['pos'].iloc[0], upstream_offset)
        group['downstream_seq'] = get_downstream_sequence(
            group['chr'].iloc[0], group['pos'].iloc[0], group['ref'].iloc[0], downstream_offset)
        group['wt_seq'] = group['upstream_seq'] + \
            group['ref'] + group['downstream_seq']
        group['mut_seq'] = group['upstream_seq'] + \
            group['alt'] + group['downstream_seq']
        return group

    df = grouped.apply(apply_func)

    return df.reset_index(drop=True)


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


def encode_RNA(mirna_seq, mirna_esa, cts_rev_seq, cts_rev_esa, with_esa):
    """ one-hot encoder for RNA sequences with/without extended seed alignments """
    chars = {"A":0, "C":1, "G":2, "U":3}
    if not with_esa:
        x = np.zeros((len(chars) * 2, 40), dtype=np.float32)
        for i in range(len(mirna_seq)):
            x[chars[mirna_seq[i]], 5 + i] = 1
        for i in range(len(cts_rev_seq)):
            x[chars[cts_rev_seq[i]] + len(chars), i] = 1
    else:
        chars["-"] = 4
        x = np.zeros((len(chars) * 2, 50), dtype=np.float32)
        for i in range(len(mirna_esa)):
            x[chars[mirna_esa[i]], 5 + i] = 1
        for i in range(10, len(mirna_seq)):
            x[chars[mirna_seq[i]], 5 + i - 10 + len(mirna_esa)] = 1
        for i in range(5):
            x[chars[cts_rev_seq[i]] + len(chars), i] = 1
        for i in range(len(cts_rev_esa)):
            x[chars[cts_rev_esa[i]] + len(chars), i + 5] = 1
        for i in range(15, len(cts_rev_seq)):
            x[chars[cts_rev_seq[i]] + len(chars), i + 5 - 15 + len(cts_rev_esa)] = 1
    return x





def prepare_sequence_for_model(mrna_seq, mirna_seq, score_matrix):

    """Prepare RNA sequences for model input"""
    # Convert to RNA format
    mirna_seq = mirna_seq.upper().replace("T", "U")
    mrna_seq = mrna_seq.upper().replace("T", "U")
    mrna_rev_seq = mrna_seq[::-1]
    
    # Alignment
    mirna_esa, mrna_rev_esa, _ = extended_seed_alignment(mirna_seq, mrna_rev_seq, score_matrix)
    
    # Encode
    x = encode_RNA(mirna_seq, mirna_esa, mrna_rev_seq, mrna_rev_esa, with_esa=True)
    return torch.from_numpy(x).unsqueeze(0)

def get_model_prediction(model: nn.Module, sequence_tensor: torch.Tensor, device: str) -> float:
    """Get prediction from model"""
    sequence_tensor = sequence_tensor.to(device)
    with torch.no_grad():
        pred = model(sequence_tensor)
        return torch.sigmoid(pred).item()

def process_sequence_pairs(model, sequence_generator, device, score_matrix):

    """Process sequence pairs and return predictions"""
    predictions = {}
    for combined_id, mrna_seq, mirna_seq in sequence_generator:
        # Use the full combined_id as key instead of just base_name
        # This ensures we keep all mRNA-miRNA combinations
        predictions[combined_id] = {
            'id': combined_id,
            'pred': get_model_prediction(model, prepare_sequence_for_model(mrna_seq, mirna_seq, score_matrix), device)
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


def predict_binding_to_df(model, device, df, mirna_dict, score_matrix):

    """Main function to orchestrate the prediction process"""
    model.eval()
    
    # Generate predictions for WT and MUT
    wt_generator = generate_mrna_mirna_pairs(df, mirna_dict, 'wt_seq')
    mut_generator = generate_mrna_mirna_pairs(df, mirna_dict, 'mut_seq')
    
    wt_predictions = process_sequence_pairs(model, wt_generator, device, score_matrix)
    mut_predictions = process_sequence_pairs(model, mut_generator, device, score_matrix)
    
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