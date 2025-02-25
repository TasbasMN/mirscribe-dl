# MirScribe-DL Codebase Guidelines

## Commands
- Run main pipeline: `python main.py -f input/path/to/file.vcf -c CHUNKSIZE -b BATCH_SIZE -w NUM_WORKERS`
- Process VCF files: `python main.py -f input/sample/sample.vcf -c 200 -b 64 -w 8` 
- Run with large batch: `python main.py -f input/vcfs/by_chr/chr1_1.vcf -c 200 -b 128 -w 16`
- Force CPU usage: `python main.py -f input/path/to/file.vcf --cpu-only`
- Specify GPU devices: `python main.py -f input/path/to/file.vcf --gpu-devices 0,1,2`
- Run on cluster with 54 cores: `python main.py -f input/path/to/file.vcf -w 54 -b 128`
- Run Jupyter notebooks: `jupyter notebook aligner.ipynb`

## Code Style Guidelines
- **Imports**: Standard library first, third-party second (pandas, numpy, torch, Bio), local imports last
- **Formatting**: 4-space indentation, avoid excessive blank lines
- **Types**: Use limited type annotations for function parameters and return values
- **Naming**: snake_case for functions/variables, PascalCase for classes, UPPER_SNAKE_CASE for constants
- **Error Handling**: Use assertions for input validation, appropriate logging for errors
- **Documentation**: Document functions with purpose, parameters, and return values
- **Performance**: Use `@lru_cache` for caching results, ThreadPoolExecutor for I/O-bound parallel tasks

## Performance Optimizations
- **Batch Processing**: Use batch size parameter (-b) to process multiple sequences at once
- **GPU Acceleration**: Model automatically uses GPU when available
- **True Multiprocessing**: System uses ProcessPoolExecutor for parallel processing across multiple cores
- **Per-Process Models**: Each worker process has its own neural network model instance
- **GPU Device Selection**: Control which GPUs to use with --gpu-devices flag
- **Chromosome Caching**: Entire chromosome sequence is loaded into memory for faster access
- **Sequence Batching**: Process miRNA-mRNA sequence pairs in batches for GPU acceleration
- **miRNA Caching**: Reuse miRNA sequences to avoid redundant processing
- **Alignment Caching**: Cache alignment results with lru_cache to avoid recomputation
- **Optimized File I/O**: Single file read for chromosome data instead of repeated lookups
- **DataFrame Optimization**: Use categorical data types and memory-efficient data structures
- **Process Isolation**: Uses 'spawn' method for proper process isolation with CUDA

## Project Structure
- `main.py`: Entry point with argparse
- `scripts/`: Core functionality modules (config, functions, pipeline, targetnet)
- `data/`: Reference data (miRNA, fasta files)
- `models/`: Pre-trained model weights