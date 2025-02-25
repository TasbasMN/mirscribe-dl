from functools import lru_cache
import pandas as pd
from scripts.config import *


if SINGLE_CHROMOSOME:# Global chromosome sequence cache 
    CHROMOSOME_SEQUENCE = None
    CHROMOSOME_ID = None
    CHROMOSOME_METADATA = None

def calculate_au_content(sequence):
    au_count = sequence.count('A') + sequence.count('T') + sequence.count('U')
    total_length = len(sequence)
    return au_count / total_length if total_length > 0 else None

def load_chromosome(chrom):
    """
    Load an entire chromosome into memory for fast access.
    
    Parameters:
    - chrom (str): The name of the chromosome.
    
    Returns:
    - Tuple containing the chromosome sequence and metadata
    """
    global CHROMOSOME_SEQUENCE, CHROMOSOME_ID, CHROMOSOME_METADATA
    
    # Already loaded this chromosome
    if CHROMOSOME_ID == str(chrom) and CHROMOSOME_SEQUENCE is not None:
        return
    
    # Convert chromosome to string
    chrom = str(chrom)
    
    # Load the chromosome file
    file_path = f"{GRCH37_DIR}/Homo_sapiens.GRCh37.dna.chromosome.{chrom}.fa"
    with open(file_path, 'r') as file:
        # Read header
        header = file.readline().strip()
        
        # Get file position after header
        header_end_pos = file.tell()
        
        # Read a line to determine line length
        first_line = file.readline().strip()
        line_length = len(first_line)
        
        # Reset to beginning of file
        file.seek(0)
        
        # Read entire file content
        content = file.read()
    
    # Remove header and all newlines to get continuous sequence
    sequence_start = content.find('\n') + 1
    sequence = content[sequence_start:].replace('\n', '')
    
    # Store chromosome and metadata
    CHROMOSOME_ID = chrom
    CHROMOSOME_SEQUENCE = sequence
    CHROMOSOME_METADATA = {
        'header': header,
        'header_end_pos': header_end_pos,
        'line_length': line_length
    }
    
    print(f"Loaded chromosome {chrom} with length {len(CHROMOSOME_SEQUENCE)}")


@lru_cache(maxsize=None)
def get_nucleotides_in_interval(chrom, start, end):
    """
    Given a chromosome name, start and end positions, this function reads the DNA sequence from the corresponding FASTA file and returns the nucleotides in the specified interval.

    Parameters:
    - chrom (str): The name of the chromosome.
    - start (int): The starting position of the interval.
    - end (int): The ending position of the interval.

    Returns:
    - nucleotides (str): The nucleotides in the specified interval.
    """

    # change chrom into str
    chrom = str(chrom)

    file_path = f"{GRCH37_DIR}/Homo_sapiens.GRCh37.dna.chromosome.{chrom}.fa"
    with open(file_path, 'r') as file:
        file.readline()
        byte_position = file.tell()
        line_length = len(file.readline().strip())
        start_offset = start - 1
        end_offset = end - 1
        num_start_new_lines = start_offset // line_length
        num_end_new_lines = end_offset // line_length
        start_byte_position = byte_position + start_offset + num_start_new_lines
        end_byte_position = byte_position + end_offset + num_end_new_lines
        file.seek(start_byte_position)

        # Read the nucleotides in the interval
        nucleotides = file.read(end_byte_position - start_byte_position + 1)

    # Remove newlines from the nucleotides
    nucleotides = nucleotides.replace('\n', '')

    return nucleotides




def get_nucleotides_in_interval_single(chrom, start, end):
    """
    Given a chromosome name, start and end positions, this function retrieves 
    nucleotides from the cached chromosome sequence.

    Parameters:
    - chrom (str): The name of the chromosome.
    - start (int): The starting position of the interval (1-based).
    - end (int): The ending position of the interval (1-based).

    Returns:
    - nucleotides (str): The nucleotides in the specified interval.
    """
    # Ensure chromosome is loaded
    if CHROMOSOME_ID != str(chrom) or CHROMOSOME_SEQUENCE is None:
        load_chromosome(chrom)
    
    # Convert to 0-based indexing
    start_idx = start - 1
    end_idx = end
    
    # Return the slice of sequence
    return CHROMOSOME_SEQUENCE[start_idx:end_idx]


@lru_cache(maxsize=None)
def get_nucleotide_at_position(chrom, position):
    """
    Given a chromosome name and a position, this function reads the DNA sequence from the corresponding FASTA file and returns the nucleotide at the specified position.

    Parameters:
    - chrom (str): The name of the chromosome.
    - position (int): The position of the nucleotide.

    Returns:
    - nucleotide (str): The nucleotide at the specified position.
    """
    file_path = f"{GRCH37_DIR}/Homo_sapiens.GRCh37.dna.chromosome.{chrom}.fa"
    with open(file_path, 'r') as file:
        file.readline()
        byte_position = file.tell()
        line_length = len(file.readline().strip())
        offset = position - 1
        num_new_lines = offset // line_length
        byte_position = byte_position + offset + num_new_lines
        file.seek(byte_position)

        # Read the nucleotide at the position
        nucleotide = file.read(1)
    return nucleotide

def get_nucleotide_at_position_single(chrom, position):
    """
    Given a chromosome name and a position, this function retrieves the nucleotide
    from the cached chromosome sequence.

    Parameters:
    - chrom (str): The name of the chromosome.
    - position (int): The position of the nucleotide (1-based).

    Returns:
    - nucleotide (str): The nucleotide at the specified position.
    """
    # Ensure chromosome is loaded
    if CHROMOSOME_ID != str(chrom) or CHROMOSOME_SEQUENCE is None:
        load_chromosome(chrom)
    
    # Convert to 0-based indexing
    idx = position - 1
    
    # Return the nucleotide
    return CHROMOSOME_SEQUENCE[idx]

@lru_cache(maxsize=None)
def get_upstream_sequence(chrom, pos, n=30):
    """
    Get the upstream sequence of length n from the given position.

    Args:
        chrom (str): The chromosome name.
        pos (int): The position.
        n (int, optional): The length of the upstream sequence. Defaults to 30.

    Returns:
        str: The upstream sequence.
    """
    int_pos = int(pos)
    upstream_start = max(1, int_pos - n)
    upstream_end = int_pos - 1
    return get_nucleotides_in_interval(chrom, upstream_start, upstream_end)

@lru_cache(maxsize=None)
def get_downstream_sequence(chrom, pos, ref, n=30):
    """
    Get the downstream sequence of length n from the given position.

    Args:
        chrom (str): The chromosome name.
        pos (int): The position.
        ref (str): The reference allele.
        n (int, optional): The length of the downstream sequence. Defaults to 30.

    Returns:
        str: The downstream sequence.
    """
    int_pos = int(pos)
    ref_len = len(ref)
    downstream_start = int_pos + ref_len
    downstream_end = downstream_start + n - 1
    return get_nucleotides_in_interval(chrom, downstream_start, downstream_end)

def get_flanking_sequences(chrom, pos, ref, upstream_offset=30, downstream_offset=30):
    """
    Get both upstream and downstream sequences in a single function call.
    This is more efficient than calling get_upstream_sequence and get_downstream_sequence separately.
    
    Args:
        chrom (str): The chromosome name.
        pos (int): The position.
        ref (str): The reference allele.
        upstream_offset (int): Length of upstream sequence. Defaults to 30.
        downstream_offset (int): Length of downstream sequence. Defaults to 30.
        
    Returns:
        tuple: (upstream_seq, downstream_seq)
    """
    # Ensure chromosome is loaded
    if CHROMOSOME_ID != str(chrom) or CHROMOSOME_SEQUENCE is None:
        load_chromosome(chrom)
        
    int_pos = int(pos)
    ref_len = len(ref)
    
    # Calculate upstream region
    upstream_start = max(1, int_pos - upstream_offset)
    upstream_end = int_pos - 1
    
    # Calculate downstream region
    downstream_start = int_pos + ref_len
    downstream_end = downstream_start + downstream_offset - 1
    
    # Get sequences directly from chromosome cache
    upstream_seq = CHROMOSOME_SEQUENCE[upstream_start-1:upstream_end]
    downstream_seq = CHROMOSOME_SEQUENCE[downstream_start-1:downstream_end]
    
    return upstream_seq, downstream_seq


@lru_cache(maxsize=None)
def get_mre_sequence(mrna_sequence, mrna_end, mirna_start, mirna_length):
    mre_end = mrna_end + mirna_start
    # Ensure MRE start is not negative
    mre_start = max(mre_end - mirna_length, 0)
    return mrna_sequence[mre_start:mre_end]
