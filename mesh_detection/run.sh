#!/bin/bash
# Quick start script for standalone mesh detection

echo "======================================"
echo "Standalone Mesh Detection System"
echo "======================================"
echo ""

# Check if in correct directory
if [ ! -f "main.py" ]; then
    echo "Error: Please run this script from the mesh_detection directory"
    exit 1
fi

# Check if config exists
if [ ! -f "config.yaml" ]; then
    echo "Error: config.yaml not found"
    exit 1
fi

# Check dependencies
echo "Checking dependencies..."
python -c "import cv2, pyrealsense2, numpy, yaml" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Missing dependencies. Installing..."
    pip install -r requirements.txt
fi

echo ""
echo "Starting mesh detection system..."
echo "Press Ctrl+C to stop"
echo ""

# Run with default config or pass through arguments
if [ $# -eq 0 ]; then
    python main.py --config config.yaml
else
    python main.py "$@"
fi
