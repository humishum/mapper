#!/bin/bash
# Start script for the 3D Pointcloud Viewer

echo "Starting web viewer"
echo ""

# Check this is running from the viewer directory
if [ "$(basename "$(pwd)")" != "viewer" ]; then
    echo "Error: This script must be run from the viewer directory"
    echo "Current directory: $(pwd)"
    exit 1
fi


# Activate virtual environment
source venv/bin/activate

# Install Python dependencies
# echo "Installing Python dependencies..."
# pip install -q -r requirements.txt

# Start backend in background
echo "Starting backend server on port 8000..."
# cd backend
python -m backend.server &
BACKEND_PID=$!
# cd ..

# Wait for backend to start
sleep 2

# Start frontend
echo "Starting frontend on port 5173..."
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

echo ""
echo "========================================="
echo "3D Pointcloud Viewer is running!"
echo "========================================="
echo "Frontend: http://localhost:5173"
echo "Backend:  http://localhost:8000"
echo ""
echo "Press Ctrl+C to stop all servers"
echo ""

# Wait for Ctrl+C
trap "echo 'Stopping servers...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT
wait

