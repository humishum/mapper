# Quick Start Guide

Get the visualizer up and running in a few simple steps:

## Step 0: Node.js Setup (First Time Only)

Make sure you have Node.js 18+ installed. If you're using the system Node (v12), upgrade using nvm:

```bash
# Install nvm
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash

# Load nvm
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

# Install Node 18
nvm install 18
nvm use 18
nvm alias default 18
```

Verify installation:
```bash
node --version  # Should show v18.x.x
```

## Step 1: Export Data (Python)

```bash
cd /home/ape/repos/mapper/visualizer
source ../.venv/bin/activate

python -m backend.export_data \
  --data-dir ../data/output_100425 \
  --threshold 2.0 \
  --max-points 50000 \
  --sequence-id 1
```

Or use the helper script:
```bash
./scripts/export_data.sh
```

This will create `frontend/public/data.json` with all your pointcloud data.

## Step 2: Install Frontend Dependencies

```bash
cd frontend
npm install
```

This only needs to be done once.

## Step 3: Start Development Server

```bash
npm run dev
```

Or use the helper script from the visualizer root:
```bash
./scripts/dev.sh
```

The visualizer will open in your browser at http://localhost:3000

## Customizing the Export

### Different Threshold
```bash
python -m backend.export_data --data-dir ../data/output_100425 --threshold 1.5
```

### More Points (Better Quality, Slower)
```bash
python -m backend.export_data --data-dir ../data/output_100425 --max-points 100000
```

### Different Data Directory
```bash
python -m backend.export_data --data-dir /path/to/your/data
```

## Building for Production

```bash
cd frontend
npm run build
```

Outputs to `frontend/dist/` - serve with any static file server.

## Troubleshooting

### "Module not found" errors
Make sure you're in the visualizer directory and the virtual environment is activated:
```bash
cd /home/ape/repos/mapper/visualizer
source ../.venv/bin/activate
```

### "Failed to load data" in browser
Make sure you ran the export script first to generate data.json

### No npm command
Install Node.js 18+ from https://nodejs.org/ or use your package manager:
```bash
# Ubuntu/Debian
sudo apt install nodejs npm

# Or use nvm for version management
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
nvm install 18
```

## Controls

- **Left Click + Drag**: Rotate camera
- **Scroll**: Zoom in/out
- **Right Click + Drag**: Pan
- **Control Panel**: Adjust point size, scale, opacity, and camera presets

Enjoy visualizing your 3D reconstructions! 🎉

