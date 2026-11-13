"""Paths, constants and hyperparameters."""

import os
from pathlib import Path

import torch

# ---- paths (override with environment variables if needed) ----
THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("CIMT_DATA_DIR", THIS_DIR / "data"))
OUTPUT_DIR = Path(os.environ.get("CIMT_OUTPUT_DIR", THIS_DIR / "results"))
RETFOUND_CHECKPOINT_PATH = Path(
    os.environ.get("RETFOUND_CHECKPOINT", THIS_DIR / "RETFound_mae_natureCFP.pth")
)

METADATA_JSON_PATH = DATA_DIR / "data_info.json"
BIOMARKERS_XLSX_PATH = DATA_DIR / "biomarkers.xlsx"
BIOMARKERS_SHEET_NAME = None  # None = first sheet; set the sheet name if the workbook has several
IMAGES_DIR = DATA_DIR / "raw"

CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOG_DIR = OUTPUT_DIR / "logs"
BIOMARKER_SCALER_PATH = CHECKPOINT_DIR / "biomarker_scaler.pkl"

for _d in (OUTPUT_DIR, CHECKPOINT_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---- biomarkers (VascX) ----
BIOMARKER_COLUMNS = [
    "median_temporal_angle_arteries", "median_temporal_angle_veins",
    "temporal_cre_arteries", "temporal_cre_veins",
    "md_diam_arteries", "md_diam_veins",
    "std_diam_arteries", "std_diam_veins",
    "vd_arteries", "vd_veins",
    "vd_etdrs_full_arteries", "vd_etdrs_full_veins",
    "md_tort_dist_length_skl_arteries", "md_tort_dist_length_skl_veins",
    "md_tort_dist_arteries", "md_tort_dist_veins",
    "lw_tort_dist_length_skl_arteries", "lw_tort_dist_length_skl_veins",
    "lw_tort_dist_arteries", "lw_tort_dist_veins",
    "md_tort_dist_vessels_length_skl_arteries", "md_tort_dist_vessels_length_skl_veins",
    "md_tort_dist_vessels_arteries", "md_tort_dist_vessels_veins",
    "md_bifangle_arteries", "md_bifangle_veins",
    "bifurcation_count_arteries", "bifurcation_count_veins",
    "mean_sparsity_vessels", "lapl_retina", "disc_fovea_distance_retina",
]
NUM_BIOMARKERS = len(BIOMARKER_COLUMNS)
IMAGE_ID_COLUMN_CANDIDATES = ["image_id", "image id", "imageid"]

# ---- demographics: age is already min-max scaled to [0, 1] in the metadata ----
DEMOGRAPHIC_FEATURES = ["age", "gender"]
NUM_DEMOGRAPHIC_FEATURES = len(DEMOGRAPHIC_FEATURES)
DEMOGRAPHIC_EMBED_DIM = 16

# ---- task ----
NUM_CLASSES = 2
CLASS_NAMES = ["normal", "thickened"]

# ---- data split ----
SPLIT_MODE = "predefined_group"  # "predefined_group" or "random"
GROUP_TO_SPLIT = {1: "train", 2: "val", 3: "test"}
RANDOM_SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}
RANDOM_SEED = 42

# ---- image preprocessing ----
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ---- model ----
RETFOUND_PATCH_SIZE = 16
RETFOUND_EMBED_DIM = 1024
RETFOUND_DEPTH = 24
RETFOUND_NUM_HEADS = 16

MODEL_MODE = "fusion"  # "image_only" or "fusion"
INTERNAL_PRETRAIN_MODES = ("demographic_only", "biomarker_only")

BIOMARKER_HIDDEN_DIMS = [64, 32]
BIOMARKER_EMBED_DIM = 32
FUSION_HIDDEN_DIM = 256  # width of the classification head of every base learner
DROPOUT = 0.3
FREEZE_BACKBONE = False

# ---- late-fusion meta-learner ----
META_LEARNER_C = 1.0  # scikit-learn default, not tuned

# ---- loss ----
USE_FOCAL_LOSS = True
FOCAL_LOSS_GAMMA = 2.0
LOSS_LABEL_SMOOTHING = 0.0  # only used when USE_FOCAL_LOSS is False

# ---- training ----
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_WORKERS = 8
EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10
LR_BACKBONE = 5e-5
LR_HEAD = 1e-3
WEIGHT_DECAY = 0.05
GRAD_CLIP_NORM = 1.0

SEED = 42
