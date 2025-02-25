import argparse
import os
import torch
from torch import device, cuda
from scripts.targetnet import TargetNet

def parse_arguments():
    """Parse command line arguments for VCF processing pipeline"""
    parser = argparse.ArgumentParser(
        description='Process a VCF file in chunks using true multiprocessing.')
    
    # File input settings
    parser.add_argument('-f', '--file_path', 
                        type=str,
                        default="input/sample/sample.vcf",
                        required=False,
                        help='Path to the VCF file')
    
    # Processing settings
    parser.add_argument("-c", '--chunksize', 
                        default=200,
                        type=int, 
                        help='Number of lines to process per chunk')
    parser.add_argument("-b", '--batch_size', 
                        default=64, 
                        type=int,
                        help='Batch size for neural network predictions')
    parser.add_argument('-w', '--workers', 
                        default=os.cpu_count(),
                        type=int, 
                        help='Number of worker processes (default: all CPU cores)')
    
    # GPU settings
    parser.add_argument('--cpu-only', 
                        action='store_true',
                        help='Force CPU usage even if CUDA is available')
    parser.add_argument('--gpu-devices',
                        type=str,
                        default=None,
                        help='Comma-separated list of GPU device indices to use (e.g., "0,1,2")')
    
    # Output settings
    parser.add_argument("-o", '--output_dir', 
                        type=str,
                        default='./results', 
                        help='Path to the output directory')
    parser.add_argument('-v', '--verbose', 
                        action='store_true',
                        help='Enable verbose logging')
    
    # Analysis settings
    parser.add_argument('--skip-rnaduplex', 
                        action='store_true',
                        help='Skip RNAduplex analysis')
    parser.add_argument('-t', '--threshold', 
                        default=0.2, 
                        type=float,
                        help='Threshold for filtering out pairs')
    parser.add_argument('-b', '--batch_size', default=64, type=int,
                        help='Batch size for neural network predictions')

    return parser.parse_args()

# Parse arguments and initialize global variables
args = parse_arguments()

# Input settings
VCF_FULL_PATH = args.file_path
VCF_COLNAMES = ["chr", "pos", "id", "ref", "alt"]
VCF_ID = os.path.basename(VCF_FULL_PATH).split(".")[0]

# Processing settings
CHUNKSIZE = args.chunksize
WORKERS = args.workers
BATCH_SIZE = args.batch_size
VERBOSE = args.verbose
SKIP_RNADUPLEX = args.skip_rnaduplex
FILTER_THRESHOLD = args.threshold
BATCH_SIZE = args.batch_size

# Output settings
OUTPUT_DIR = os.path.join(args.output_dir, f"{VCF_ID}_{CHUNKSIZE}")

# Path settings
GRCH37_DIR = "data/fasta/grch37"
MIRNA_COORDS_DIR = "data/mirna_coordinates"
MIRNA_CSV = "data/mirna/mirna.csv"

# Sequence analysis settings
UPSTREAM_OFFSET = 29
DOWNSTREAM_OFFSET = 10
DECISION_SURFACE = 0.5

# Model configuration
class ModelConfig:
    def __init__(self):
        self.skip_connection = True
        self.num_channels = [16, 16, 32]
        self.num_blocks = [2, 1, 1]
        self.stem_kernel_size = 5
        self.block_kernel_size = 3
        self.pool_size = 3

# Handle GPU settings
CPU_ONLY = args.cpu_only
GPU_DEVICES = args.gpu_devices

# Process GPU device configuration
if GPU_DEVICES and not CPU_ONLY:
    # Set visible devices based on user input
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_DEVICES
    print(f"Using GPU devices: {GPU_DEVICES}")

# Determine device to use
if CPU_ONLY:
    DEVICE = device("cpu")
    print("Forcing CPU usage as requested")
else:
    DEVICE = device("cuda" if cuda.is_available() else "cpu")
    if DEVICE.type == "cuda":
        print(f"Using CUDA with {torch.cuda.device_count()} visible devices")
    else:
        print("CUDA not available, using CPU")

# Model configuration
MODEL_PATH = "models/TargetNet.pt"
model_cfg = ModelConfig()

# For the main process, we'll initialize a model instance
# Each worker process will initialize its own model copy
if DEVICE.type == "cuda":
    # Do not initialize model in main process when running with CUDA
    # to avoid CUDA initialization issues
    MODEL = None
else:
    # For CPU mode, initialize model in main process
    model = TargetNet(model_cfg, with_esa=True, dropout_rate=0.5)
    MODEL = model.to(DEVICE)
    MODEL.load_state_dict(torch.load(MODEL_PATH, weights_only=True, map_location=DEVICE))

# RNA pairing score matrix
SCORE_MATRIX = {}
for c1 in 'ACGU':
    for c2 in 'ACGU':
        if (c1, c2) in [('A', 'U'), ('U', 'A'), ('G', 'C'), ('C', 'G')]:
            SCORE_MATRIX[(c1, c2)] = 1
        elif (c1, c2) in [('U', 'G'), ('G', 'U')]:
            SCORE_MATRIX[(c1, c2)] = 1
        else:
            SCORE_MATRIX[(c1, c2)] = 0