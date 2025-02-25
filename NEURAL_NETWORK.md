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