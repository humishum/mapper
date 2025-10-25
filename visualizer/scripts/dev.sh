#!/bin/bash
# Start the development server

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VISUALIZER_DIR="$(dirname "$SCRIPT_DIR")"
FRONTEND_DIR="$VISUALIZER_DIR/frontend"

# Load nvm if available
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

# Check if data.json exists
if [ ! -f "$FRONTEND_DIR/public/data.json" ]; then
  echo "=================================================="
  echo "ERROR: data.json not found!"
  echo "=================================================="
  echo "Please run the export script first:"
  echo "  ./scripts/export_data.sh"
  echo ""
  echo "Or manually export:"
  echo "  python -m backend.export_data --data-dir ../data/output_100425"
  echo "=================================================="
  exit 1
fi

echo "=================================================="
echo "Starting Development Server"
echo "=================================================="
echo "Frontend directory: $FRONTEND_DIR"
echo ""

# Check if node_modules exists
if [ ! -d "$FRONTEND_DIR/node_modules" ]; then
  echo "Installing dependencies..."
  cd "$FRONTEND_DIR"
  npm install
  if [ $? -ne 0 ]; then
    echo "Failed to install dependencies"
    exit 1
  fi
fi

# Start the dev server
cd "$FRONTEND_DIR"
npm run dev

