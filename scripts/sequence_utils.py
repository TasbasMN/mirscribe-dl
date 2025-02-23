from functools import lru_cache
import pandas as pd
from scripts.config import *


def calculate_au_content(sequence):
    au_count = sequence.count('A') + sequence.count('T') + sequence.count('U')
    total_length = len(sequence)
    return au_count / total_length if total_length > 0 else None


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


@lru_cache(maxsize=None)
def get_upstream_sequence(chrom, pos, n=30):
    """
    Get the upstream sequence of length n from the given position.

    Args:
        row (pandas.Series): A row from the DataFrame containing the 'chr', 'pos', and 'ref_len' columns.
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


@lru_cache(maxsize=None)
def get_mre_sequence(mrna_sequence, mrna_end, mirna_start, mirna_length):
    mre_end = mrna_end + mirna_start
    # Ensure MRE start is not negative
    mre_start = max(mre_end - mirna_length, 0)
    return mrna_sequence[mre_start:mre_end]
