# config.py
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

HUMAN_DIR = "human"
CLOTH_DIR = "cloth"
ANNOTATION_DIR = os.path.join(RAW_DATA_DIR, "annotation")

HUMAN_PROCESSED_DIR = os.path.join(PROCESSED_DATA_DIR, "human")
CLOTH_PROCESSED_DIR = os.path.join(PROCESSED_DATA_DIR, "cloth")
CLOTH_MASK_PROCESSED_DIR = os.path.join(PROCESSED_DATA_DIR, "cloth_mask")
POSE_PROCESSED_DIR = os.path.join(PROCESSED_DATA_DIR, "pose")
WARPED_CLOTH_PROCESSED_DIR = os.path.join(PROCESSED_DATA_DIR, "warped_cloth")

# Image dimensions
IMAGE_WIDTH = 768
IMAGE_HEIGHT = 1024
# or
# IMAGE_WIDTH = 512
# IMAGE_HEIGHT = 384

# Other settings
USE_CUDA = False  # or False if you don't have a GPU