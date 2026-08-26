import torch

RAW_DIR = "data/raw/gdsc"
PROCESSED_DIR = "data/processed"
CHECKPOINT_DIR = "models/checkpoints"
RESULTS_DIR = "assets/results"

TOP_N_GENES = 1000
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42
VARIANCE_SEEDS = [42, 7, 123]

ATOM_FEATURE_DIM = 20
GENE_FEATURE_DIM = TOP_N_GENES
HIDDEN_DIM = 128
DROPOUT = 0.2
ATTENTION_LEAKYRELU_SLOPE = 0.2
ENCODER_LEAKYRELU_SLOPE = 0.01

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 32
MAX_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 5

RELATIONS = ["bond", "ring", "fg"]

BOOTSTRAP_RESAMPLES = 1000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
