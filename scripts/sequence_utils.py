from functools import lru_cache
import pandas as pd
from scripts.config import *


if SINGLE_CHROMOSOME:# Global chromosome sequence cache 
    CHROMOSOME_SEQUENCE = None
    CHROMOSOME_ID = None
    CHROMOSOME_METADATA = None

def load_chrom(chr_id):
    """Load chromosome into memory for fast access."""
    global CHROMOSOME_SEQUENCE, CHROMOSOME_ID, CHROMOSOME_METADATA
    
    # Skip if already loaded
    if CHROMOSOME_ID == str(chr_id) and CHROMOSOME_SEQUENCE is not None:
        return
    
    chr_id = str(chr_id)
    
    # Load chromosome file
    path = f"{GRCH37_DIR}/Homo_sapiens.GRCh37.dna.chromosome.{chr_id}.fa"
    with open(path, 'r') as file:
        header = file.readline().strip()
        head_pos = file.tell()
        first_line = file.readline().strip()
        line_len = len(first_line)
        file.seek(0)
        content = file.read()
    
    # Extract sequence
    seq_start = content.find('\n') + 1
    seq = content[seq_start:].replace('\n', '')
    
    # Store data
    CHROMOSOME_ID = chr_id
    CHROMOSOME_SEQUENCE = seq
    CHROMOSOME_METADATA = {
        'header': header,
        'header_end_pos': head_pos,
        'line_length': line_len
    }
    
    print(f"Loaded chromosome {chr_id} with length {len(CHROMOSOME_SEQUENCE)}")



@lru_cache(maxsize=None)
def get_seq_interval(chr, start, end):
    """Get DNA sequence from specified chromosome interval."""
    chr = str(chr)

    path = f"{GRCH37_DIR}/Homo_sapiens.GRCh37.dna.chromosome.{chr}.fa"
    with open(path, 'r') as file:
        file.readline()
        byte_pos = file.tell()
        line_len = len(file.readline().strip())
        start_offset = start - 1
        end_offset = end - 1
        start_nl = start_offset // line_len
        end_nl = end_offset // line_len
        start_byte = byte_pos + start_offset + start_nl
        end_byte = byte_pos + end_offset + end_nl
        file.seek(start_byte)
        seq = file.read(end_byte - start_byte + 1)

    return seq.replace('\n', '')




def get_seq_interval_single(chrom, start, end):
    """Get nucleotides from cached chromosome sequence."""
    # Ensure chromosome is loaded
    if CHROMOSOME_ID != str(chrom) or CHROMOSOME_SEQUENCE is None:
        load_chrom(chrom)
    
    # Convert to 0-based indexing and return slice
    return CHROMOSOME_SEQUENCE[start-1:end]



@lru_cache(maxsize=None)
def get_nuc_at_pos(chrom, position):
    """Get nucleotide at specified chromosome position."""
    path = f"{GRCH37_DIR}/Homo_sapiens.GRCh37.dna.chromosome.{chrom}.fa"
    with open(path, 'r') as file:
        file.readline()
        byte_pos = file.tell()
        line_len = len(file.readline().strip())
        offset = position - 1
        nl_count = offset // line_len
        file.seek(byte_pos + offset + nl_count)
        return file.read(1)


def get_nuc_at_pos_single(chrom, position):
    """Get nucleotide from cached chromosome sequence."""
    # Ensure chromosome is loaded
    if CHROMOSOME_ID != str(chrom) or CHROMOSOME_SEQUENCE is None:
        load_chrom(chrom)
    
    # Return nucleotide (convert from 1-based to 0-based)
    return CHROMOSOME_SEQUENCE[position-1]


@lru_cache(maxsize=None)
def get_upstream(chrom, pos, n=30):
    """Get upstream sequence of length n."""
    pos = int(pos)
    start = max(1, pos - n)
    end = pos - 1
    return get_seq_interval(chrom, start, end)

@lru_cache(maxsize=None)
def get_downstream(chrom, pos, ref, n=30):
    """Get downstream sequence of length n."""
    pos = int(pos)
    start = pos + len(ref)
    end = start + n - 1
    return get_seq_interval(chrom, start, end)



def get_flanks(chrom, pos, ref, upstream_offset=30, downstream_offset=30):
    """Get both upstream and downstream sequences efficiently."""
    # Ensure chromosome is loaded
    if CHROMOSOME_ID != str(chrom) or CHROMOSOME_SEQUENCE is None:
        load_chrom(chrom)
        
    pos = int(pos)
    
    # Calculate regions
    up_start = max(1, pos - upstream_offset)
    up_end = pos - 1
    
    down_start = pos + len(ref)
    down_end = down_start + downstream_offset - 1
    
    # Get sequences from chromosome cache
    up_seq = CHROMOSOME_SEQUENCE[up_start-1:up_end]
    down_seq = CHROMOSOME_SEQUENCE[down_start-1:down_end]
    
    return up_seq, down_seq


@lru_cache(maxsize=None)
def get_mre_sequence(mrna_sequence, mrna_end, mirna_start, mirna_length):
    mre_end = mrna_end + mirna_start
    # Ensure MRE start is not negative
    mre_start = max(mre_end - mirna_length, 0)
    return mrna_sequence[mre_start:mre_end]
