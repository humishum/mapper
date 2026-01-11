"""Config and env management"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Data directory containing pointcloud folders
# DATA_DIR = Path(os.getenv("DATA_DIR", "/home/ape/repos/mapper/data/output_kings_canyon/kings_canyon_must3r_20251229_214719"))
DATA_DIR = Path("/home/ape/repos/mapper/data/output_kings_canyon/kings_canyon_must3r_20260105_211309")
DATA_DIR=Path("/home/ape/mapper_output/vggt_trail_test_20260111_013711/outputs")
print(f"DATA_DIR: {DATA_DIR}")
# Server configuration
PORT = int(os.getenv("PORT", "8000"))

# Processing configuration
MAX_POINTS = int(os.getenv("MAX_POINTS", "100000"))
DEFAULT_THRESHOLD = float(os.getenv("DEFAULT_THRESHOLD", "5.0"))

# Cache configuration
CACHE_SIZE = int(os.getenv("CACHE_SIZE", "5"))

