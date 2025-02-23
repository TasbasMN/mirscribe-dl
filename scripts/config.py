import argparse
import os
from torch import device, cuda
from scripts.targetnet import *



def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Process a VCF file in chunks using concurrent futures.')
    
    # Changed to optional argument with -f/--file
    parser.add_argument('-f', '--file_path', 
                        type=str,
                        default="input/sample/sample.vcf",
                        required=False,
                        help='Path to the VCF file')
    
    # Rest of your arguments
    parser.add_argument("-c", '--chunksize', default=200,
                        type=int, help='Number of lines to process per chunk')
    parser.add_argument("-o", '--output_dir', type=str,
                        default='./results', help='Path to the output directory')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Enable verbose logging')
    parser.add_argument('-w', '--workers', default=os.cpu_count(),
                        type=int, help='Number of concurrent workers')
    parser.add_argument('--skip-rnaduplex', action='store_true',
                        help='Skip RNAduplex analysis')
    parser.add_argument('-t', '--threshold', default=0.2, type=float,
                        help='Threshold for filtering out pairs')


    return parser.parse_args()



args = parse_arguments()

VCF_FULL_PATH = args.file_path
CHUNKSIZE = args.chunksize
VERBOSE = args.verbose
WORKERS = args.workers
SKIP_RNADUPLEX = args.skip_rnaduplex
FILTER_THRESHOLD = args.threshold


VCF_COLNAMES = ["chr", "pos", "id", "ref", "alt"]

VCF_ID = os.path.basename(VCF_FULL_PATH).split(".")[0]
OUTPUT_DIR = os.path.join(args.output_dir, f"{VCF_ID}_{CHUNKSIZE}")


class ModelConfig:
    def __init__(self):
        self.skip_connection = True
        self.num_channels = [16, 16, 32]
        self.num_blocks = [2, 1, 1]
        self.stem_kernel_size = 5
        self.block_kernel_size = 3
        self.pool_size = 3



DEVICE = device("cuda" if cuda.is_available() else "cpu")
MODEL_PATH = "models/TargetNet.pt"
model_cfg = ModelConfig()

# For ESA mode
model = TargetNet(model_cfg, with_esa=True, dropout_rate=0.5)
MODEL = model.to(DEVICE)
MODEL.load_state_dict(torch.load(MODEL_PATH, weights_only=True))


SCORE_MATRIX = {}
for c1 in 'ACGU':
    for c2 in 'ACGU':
        if (c1, c2) in [('A', 'U'), ('U', 'A'), ('G', 'C'), ('C', 'G')]:
            SCORE_MATRIX[(c1, c2)] = 1
        elif (c1, c2) in [('U', 'G'), ('G', 'U')]:
            SCORE_MATRIX[(c1, c2)] = 1
        else:
            SCORE_MATRIX[(c1, c2)] = 0
            

DECISION_SURFACE = 0.5


GRCH37_DIR = "data/fasta/grch37"
MIRNA_COORDS_DIR = "data/mirna_coordinates"
MIRNA_CSV = "data/mirna/mirna.csv"

UPSTREAM_OFFSET = 29
DOWNSTREAM_OFFSET = 10