#!/bin/bash
# Export pointcloud data to JSON format for visualization

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VISUALIZER_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$VISUALIZER_DIR")"

# Activate virtual environment
source "$PROJECT_ROOT/.venv/bin/activate"

# Default values
DATA_DIR="$PROJECT_ROOT/data/output_100425"
OUTPUT="$VISUALIZER_DIR/frontend/public/data.json"
THRESHOLD=2.0
MAX_POINTS=5000
SEQUENCE_ID=1

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --data-dir)
      DATA_DIR="$2"
      shift 2
      ;;
    --output)
      OUTPUT="$2"
      shift 2
      ;;
    --threshold)
      THRESHOLD="$2"
      shift 2
      ;;
    --max-points)
      MAX_POINTS="$2"
      shift 2
      ;;
    --sequence-id)
      SEQUENCE_ID="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--data-dir DIR] [--output FILE] [--threshold N] [--max-points N] [--sequence-id N]"
      exit 1
      ;;
  esac
done

echo "=================================================="
echo "Exporting Pointcloud Data"
echo "=================================================="
echo "Data directory: $DATA_DIR"
echo "Output file: $OUTPUT"
echo "Threshold: $THRESHOLD"
echo "Max points: $MAX_POINTS"
echo "Sequence ID: $SEQUENCE_ID"
echo ""

# Create output directory if it doesn't exist
mkdir -p "$(dirname "$OUTPUT")"

# Run the export script
cd "$VISUALIZER_DIR"
python -m backend.export_data \
  --data-dir "$DATA_DIR" \
  --output "$OUTPUT" \
  --threshold "$THRESHOLD" \
  --max-points "$MAX_POINTS" \
  --sequence-id "$SEQUENCE_ID"

if [ $? -eq 0 ]; then
  echo ""
  echo "=================================================="
  echo "Export completed successfully!"
  echo "You can now run: ./scripts/dev.sh"
  echo "=================================================="
else
  echo ""
  echo "=================================================="
  echo "Export failed!"
  echo "=================================================="
  exit 1
fi

