# Neural Network Architecture and Data Processing

This document describes the neural network architecture used in MirScribe-DL for predicting miRNA-mRNA binding interactions, as well as the data preprocessing and transformation pipeline.

## TargetNet Architecture

TargetNet is a ResNet-style convolutional neural network designed specifically for miRNA-target prediction:

```
TargetNet
├── Stem Layer: Conv1D layers with kernel size 5
├── Stage 1: ResNet blocks with kernel size 3
├── Stage 2: ResNet blocks with kernel size 3
├── Global Max Pooling
└── Fully Connected Layer
```

### Understanding ResNet Architecture

ResNet (Residual Network) is a pioneering neural network architecture that introduced "skip connections" or "shortcuts" to solve the vanishing gradient problem in deep networks:

1. **Core Concept**: Unlike traditional sequential networks, ResNet allows information to skip layers via identity connections, enabling the training of much deeper networks.

2. **Residual Learning**: Instead of learning the desired underlying mapping directly, ResNet blocks learn the residual (difference) between input and output, which is often easier to optimize.

3. **Skip Connections**: The defining feature of ResNet is its skip connections, which add the input of a layer directly to its output:
   ```
   Output = F(Input) + Input
   ```
   where F() represents the convolution layers.

4. **Advantages**:
   - Solves vanishing/exploding gradient problems
   - Enables training of very deep networks
   - Improves gradient flow through the network
   - Provides faster convergence during training
   - Often results in better performance with fewer parameters

In TargetNet, we utilize a 1D variant of ResNet optimized for sequence data, where the skip connections help maintain important positional and sequential information throughout the network.

### Network Configuration

- **Input Dimensions**: 10 channels × 50 length (one-hot encoded RNA sequences with alignment)
- **Convolution Channels**: [16, 16, 32]
- **Number of Blocks**: [2, 1, 1]
- **Kernel Sizes**: 5 (stem), 3 (blocks)
- **Pooling Size**: 3
- **Activation**: ReLU
- **Dropout Rate**: 0.5
- **Skip Connections**: Yes

### ResNet Block Structure

Each ResNet block follows this structure:
1. ReLU activation
2. Dropout (0.5)
3. Convolution
4. ReLU activation
5. Dropout (0.5)
6. Convolution
7. Skip connection (adding input to output)

## Sequence Data Preprocessing

### Extended Seed Alignment (ESA)

Before feeding sequences into the neural network, we perform an extended seed alignment between the miRNA seed region and the potential target site:

1. Take the first 10 nucleotides from the miRNA sequence (positions 1-10)
2. Take a potential binding region from the reversed mRNA sequence (positions 5-15)
3. Perform a global alignment using the BioPython pairwise2 module
4. The alignment introduces gap characters ("-") to optimize matching

```python
def extended_seed_alignment(mi_seq, cts_r_seq, score_matrix):
    alignment = pairwise2.align.globaldx(mi_seq[:10], cts_r_seq[5:15], score_matrix, one_alignment_only=True)[0]
    mi_esa = alignment[0]  # Aligned miRNA seed region
    cts_r_esa = alignment[1]  # Aligned mRNA target region
    esa_score = alignment[2]  # Alignment score
    return mi_esa, cts_r_esa, esa_score
```

### RNA Sequence Encoding

After alignment, sequences are encoded into a numeric format suitable for the neural network:

1. **One-hot encoding**: Each nucleotide (A, C, G, U) and gap (-) is encoded into a 5-element one-hot vector
2. **Sequence arrangement**: The miRNA and mRNA sequences are arranged to maintain their relative positioning
3. **Final tensor shape**: The encoding produces a (10, 50) shaped tensor representing both sequences

```python
def encode_RNA(mirna_seq, mirna_esa, cts_rev_seq, cts_rev_esa):
    # Map nucleotides to indices
    chars = {"A": 0, "C": 1, "G": 2, "U": 3, "-": 4}
    # Initialize tensor with zeros
    x = np.zeros((10, 50), dtype=np.float32)
    
    # Encode miRNA with ESA
    for i in range(len(mirna_esa)):
        x[chars[mirna_esa[i]], 5 + i] = 1
    for i in range(10, len(mirna_seq)):
        x[chars[mirna_seq[i]], 5 + i - 10 + len(mirna_esa)] = 1
    
    # Encode mRNA with ESA
    for i in range(5):
        x[chars[cts_rev_seq[i]] + 5, i] = 1
    for i in range(len(cts_rev_esa)):
        x[chars[cts_rev_esa[i]] + 5, i + 5] = 1
    for i in range(15, len(cts_rev_seq)):
        x[chars[cts_rev_seq[i]] + 5, i + 5 - 15 + len(cts_rev_esa)] = 1
    
    return x
```

## Data Flow Pipeline

The complete process from raw sequences to predictions follows these steps:

1. **Preprocessing**:
   - Load reference sequences and mutation data
   - Generate wild-type and mutant sequences
   - Extract relevant regions with flanking sequences

2. **Batch Processing**:
   - Group sequences into batches for efficient processing
   - Cache miRNA sequences to avoid redundant computation
   - Apply transformations in parallel across multiple cores

3. **Neural Network Prediction**:
   - Convert each sequence pair into a tensor
   - Feed tensors through the TargetNet model
   - Apply sigmoid activation to get binding probability
   - Calculate the difference between wild-type and mutant binding probabilities

4. **Post-processing**:
   - Filter results based on significance thresholds
   - Identify mutations that change binding status across decision boundary
   - Return results sorted by binding probability difference

## Prediction Batching

To optimize performance, sequence pairs are processed in batches:

```python
def process_sequence_pairs(model, sequence_generator, device, score_matrix, batch_size=64):
    predictions = {}
    batch_ids = []
    batch_tensors = []
    
    # Cache for miRNA data to avoid redundant computations
    mirna_cache = {}
    
    for combined_id, mrna_seq, mirna_seq in sequence_generator:
        # Check if miRNA is in cache
        if mirna_seq not in mirna_cache:
            processed_mirna = prepare_mirna_for_alignment(mirna_seq)
            mirna_cache[mirna_seq] = processed_mirna
        else:
            processed_mirna = mirna_cache[mirna_seq]
            
        # Process mRNA and create tensor
        processed_mrna = prepare_mrna_for_alignment(mrna_seq)
        mirna_esa, mrna_rev_esa, _ = extended_seed_alignment(processed_mirna, processed_mrna, score_matrix)
        x = encode_RNA(processed_mirna, mirna_esa, processed_mrna, mrna_rev_esa)
        tensor = torch.from_numpy(x).unsqueeze(0)
        
        batch_ids.append(combined_id)
        batch_tensors.append(tensor)
        
        # Process batch when it reaches the specified size
        if len(batch_tensors) >= batch_size:
            stacked_tensors = torch.cat(batch_tensors, dim=0)
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
```

## Complete Sequence Transformation Pipeline

This section provides a detailed, step-by-step explanation of how genomic sequences are transformed for neural network analysis, following the data flow from raw genomic sequence to prediction-ready tensor.

### 1. Sequence Extraction from Genomic Context

When a mutation is identified in a VCF file, we extract flanking sequences to provide genomic context:

```python
def get_flanks(chrom, pos, ref, upstream_offset=29, downstream_offset=10):
    """Get both upstream and downstream sequences efficiently."""
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
```

For each variant, we create two sequences:
- **Wild-type sequence**: `upstream_seq + ref + downstream_seq`
- **Mutant sequence**: `upstream_seq + alt + downstream_seq`

By default, we use:
- 29 nucleotides upstream of the mutation
- The reference or alternate allele (at position 30)
- 10 nucleotides downstream

This creates a 40-nucleotide sequence (for a single-nucleotide variant) with the mutation at position 30.

Example:
```
Chromosome sequence: ...AGCTCTAGCTAGCTAGCTAGCGATTCGATCGTAGCTAGCTAGCTAGC...
                                                 ^ mutation site (G→C)

Extracted wild-type: AGCTCTAGCTAGCTAGCTAGCGATTCGATCGTAGCTA (40 nt)
                                         ^ mutation at position 30

Extracted mutant:    AGCTCTAGCTAGCTAGCTAGCGATTCGATCGTAGCTA (40 nt)
                                         ^ mutation at position 30
```

### 2. Sequence Preparation and Reversal

Each 40-nucleotide sequence undergoes two critical transformations:

1. **DNA to RNA conversion**: Replace 'T' with 'U' to convert DNA to RNA format
   ```python
   mrna_seq = mrna_seq.upper().replace("T", "U")
   ```

2. **Sequence reversal**: Reverse the sequence to simulate the biological target binding orientation
   ```python
   mrna_rev_seq = mrna_seq[::-1]  # e.g., ACGU → UGCA
   ```

After reversal, the mutation position shifts from position 30 to position 10 (counting from 0 in the reversed sequence).

Example:
```
Original:  AGCUCUAGCUAGCUAGCUAGCGAUUCGAUCGUAGCUA
                                 ^ mutation at position 30

Reversed:  AUCGAUGCUAGCUAGCUAGCUUAGCGAUCGAUCGCA
                    ^ mutation now at position 10
```

This positioning is crucial because in the next step, we align the miRNA seed region with a specific section of this reversed sequence that includes the mutation site.

### 3. Extended Seed Alignment (ESA)

The core of our approach is aligning the miRNA seed region with the potential target site that includes the mutation:

```python
# For a miRNA sequence like "AGCUGCCAAUUCGAUACGA"
# And a reversed mRNA sequence like "AUCGAUGCUAGCUAGCUAGCUUAGCGAUCGAUCGCA"
mi_seq_slice = mi_seq[:10]           # Take first 10 nt: "AGCUGCCAAU"
cts_r_seq_slice = cts_r_seq[5:15]    # Take nt 5-15 (includes mutation at pos 10): "UGCUAGCUAG"

# Global alignment introduces gaps to optimize matching
alignment = pairwise2.align.globaldx(mi_seq_slice, cts_r_seq_slice, score_matrix)
# Example result:
# mi_esa = "AGCUGCCAAU"
# cts_r_esa = "UGC-AGCUAG"  (a gap was introduced)
```

By extracting positions 5-15 from the reversed sequence, we ensure that:
1. The mutation (at position 10) is included in the alignment region
2. We have enough context (5nt before) for proper seed alignment
3. We capture the biological mechanism where miRNA seed regions (positions 2-8) align with mRNA targets

The alignment may introduce gaps ("-") in either sequence to maximize base-pairing, mimicking the biological process of RNA hybridization.

### 4. One-hot Encoding with Position-Aware Alignment

The final step creates a tensor that preserves both sequence identity and positional relationships:

```
Input Tensor Shape: (10, 50)
- First 5 rows: One-hot encoded miRNA nucleotides (A,C,G,U,-)
- Last 5 rows: One-hot encoded mRNA nucleotides (A,C,G,U,-)
```

Here's a visualization of the encoding:

```
    0  1  2  3  4  5  6  7  8 ... 49
    --------------------------------
A | 0  0  0  0  0  1  0  0  0 ...  0  ← miRNA A nucleotides
C | 0  0  0  0  0  0  0  1  0 ...  0  ← miRNA C nucleotides
G | 0  0  0  0  0  0  1  0  0 ...  0  ← miRNA G nucleotides
U | 0  0  0  0  0  0  0  0  0 ...  0  ← miRNA U nucleotides
- | 0  0  0  0  0  0  0  0  1 ...  0  ← miRNA gap positions
A | 0  0  0  0  0  0  0  0  0 ...  0  ← mRNA A nucleotides
C | 0  0  0  0  1  0  0  0  0 ...  0  ← mRNA C nucleotides
G | 0  0  0  0  0  0  0  0  0 ...  0  ← mRNA G nucleotides
U | 1  0  0  0  0  0  0  0  0 ...  0  ← mRNA U nucleotides
- | 0  0  0  0  0  0  0  1  0 ...  0  ← mRNA gap positions
```

The detailed encoding algorithm follows these steps:

1. **Initialize matrix**:
   ```python
   chars = {"A": 0, "C": 1, "G": 2, "U": 3, "-": 4}
   x = np.zeros((10, 50), dtype=np.float32)
   ```

2. **Encode miRNA with alignment**:
   ```python
   # First, encode aligned seed region (with gaps if any)
   for i in range(len(mirna_esa)):
       x[chars[mirna_esa[i]], 5 + i] = 1
   
   # Then encode rest of miRNA after seed region
   for i in range(10, len(mirna_seq)):
       x[chars[mirna_seq[i]], 5 + i - 10 + len(mirna_esa)] = 1
   ```

3. **Encode mRNA with alignment**:
   ```python
   # Encode first 5 nucleotides before aligned region
   for i in range(5):
       x[chars[cts_rev_seq[i]] + 5, i] = 1
   
   # Encode aligned region (with gaps if any)
   for i in range(len(cts_rev_esa)):
       x[chars[cts_rev_esa[i]] + 5, i + 5] = 1
   
   # Encode remaining nucleotides after aligned region
   for i in range(15, len(cts_rev_seq)):
       x[chars[cts_rev_seq[i]] + 5, i + 5 - 15 + len(cts_rev_esa)] = 1
   ```

4. **Finalize tensor**:
   ```python
   tensor = torch.from_numpy(x).unsqueeze(0)  # Add batch dimension
   ```

This encoding meticulously preserves:
- Nucleotide identity (which bases are present)
- Positional information (where each base is located)
- Alignment structure (how bases match up)
- Gap insertions (where alignment requires gaps)
- The relative position of the mutation within the alignment

The mutation site is specially positioned within the alignment region to ensure the neural network can detect how sequence changes affect binding potential.

### Tracing the Mutation Through the Pipeline

To clearly illustrate how a mutation is tracked through this process:

1. **Original VCF mutation**: A variant at chromosome position X
2. **Extracted sequence**: 40nt with mutation at position 30 (from 0-based index)
3. **Reversed sequence**: Mutation now at position 10 from the 5' end
4. **Alignment region**: Positions 5-15 of reversed sequence, including the mutation
5. **Encoded tensor**: Mutation is represented within the aligned region of the tensor

This precise positioning ensures that the neural network can accurately assess how the mutation affects miRNA binding.

The resulting tensor is then fed into the TargetNet neural network for binding prediction.

## Mutation Impact Assessment

The final step calculates the difference between wild-type and mutant predictions to assess the impact of mutations:

```python
def create_comparison_results(wt_preds, mut_preds):
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
```

Mutations that result in a significant change in binding probability and cross the decision threshold (typically 0.5) are considered functionally important.