# MirScribe-DL

A deep learning pipeline for predicting the effect of genetic mutations on microRNA-mRNA binding interactions.

## Overview

MirScribe-DL analyzes VCF (Variant Call Format) files containing genetic variants and predicts how these mutations affect miRNA regulation. The system uses a ResNet-based neural network (TargetNet) to predict binding affinity changes between miRNAs and their potential target sites.

## Features

- Prediction of mutation effects on miRNA-mRNA binding
- Validation of variant data against reference genome
- Classification of variants based on genomic context
- Multi-process pipeline for high-throughput analysis
- GPU acceleration for neural network prediction
- Memory-efficient processing of large VCF files
- Optimized sequence alignment and prediction

## Installation

### Prerequisites

- Python 3.8+
- PyTorch
- pandas
- numpy
- BioPython

### Reference Data

The pipeline requires reference genome data (GRCh37/hg19). Download the required chromosome files:

```bash
# Create directory structure
mkdir -p data/fasta/grch37/

# Download individual chromosomes
wget -P data/fasta/grch37/ ftp://ftp.ensembl.org/pub/grch37/current/fasta/homo_sapiens/dna/Homo_sapiens.GRCh37.dna.chromosome.{1..22}.fa.gz
wget -P data/fasta/grch37/ ftp://ftp.ensembl.org/pub/grch37/current/fasta/homo_sapiens/dna/Homo_sapiens.GRCh37.dna.chromosome.X.fa.gz
wget -P data/fasta/grch37/ ftp://ftp.ensembl.org/pub/grch37/current/fasta/homo_sapiens/dna/Homo_sapiens.GRCh37.dna.chromosome.Y.fa.gz
wget -P data/fasta/grch37/ ftp://ftp.ensembl.org/pub/grch37/current/fasta/homo_sapiens/dna/Homo_sapiens.GRCh37.dna.chromosome.MT.fa.gz

# Decompress the files (must be in the data/fasta/grch37/ directory)
cd data/fasta/grch37/
gunzip Homo_sapiens.GRCh37.dna.chromosome.*.fa.gz
cd ../../../
```

## Usage

Basic usage:

```bash
python main.py -f input/path/to/file.vcf -c 200 -b 64 -w 8
```

Parameters:
- `-f, --file_path`: Path to the VCF file (required)
- `-c, --chunksize`: Number of lines to process per chunk (default: 200)
- `-b, --batch_size`: Batch size for neural network predictions (default: 64)
- `-w, --workers`: Number of worker processes (default: all CPU cores)
- `-o, --output_dir`: Path to the output directory (default: ./results)
- `-s, --single-chromosome`: Optimize for processing a single-chromosome file
- `--cpu-only`: Force CPU usage even if CUDA is available
- `--gpu-devices`: Comma-separated list of GPU device indices to use

For single-chromosome processing (optimized mode):

```bash
python main.py -f input/vcfs/by_chr/chr1_1.vcf -s -c 200 -b 64 -w 8
```

For high-performance processing with GPU acceleration:

```bash
python main.py -f input/path/to/file.vcf -c 200 -b 128 -w 16 --gpu-devices 0,1
```

## Output

Results are saved in the specified output directory as CSV files with the following columns:
- Variant identifier
- Wild-type binding prediction
- Mutant binding prediction
- Prediction difference

## Project Structure

- `main.py`: Entry point with command-line interface
- `scripts/`: Core functionality modules
  - `config.py`: Configuration and argument parsing
  - `pipeline.py`: Multi-processing pipeline
  - `functions.py`: Sequence analysis functions
  - `sequence_utils.py`: Chromosome data handling
  - `targetnet.py`: Neural network model
- `data/`: Reference data
- `models/`: Pre-trained neural network weights
- `input/`: VCF files for analysis
- `results/`: Analysis output
