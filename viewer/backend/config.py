"""Config and env management"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Data directory containing pointcloud folders
DATA_DIR = Path(os.getenv("DATA_DIR", "/home/ape/mapper_output/122225-must3r"))

# Server configuration
PORT = int(os.getenv("PORT", "8000"))

# Processing configuration
MAX_POINTS = int(os.getenv("MAX_POINTS", "100000"))
DEFAULT_THRESHOLD = float(os.getenv("DEFAULT_THRESHOLD", "5.0"))

# Cache configuration
CACHE_SIZE = int(os.getenv("CACHE_SIZE", "5"))
