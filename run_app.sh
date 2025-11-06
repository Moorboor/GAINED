#!/bin/bash
# GAINED Therapy Session Analysis App - Quick Start Script

echo "=================================="
echo "GAINED - Therapy Session Analysis"
echo "=================================="
echo ""

# Navigate to project directory
cd "$(dirname "$0")"

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Navigate to app directory
cd app

# Run the app
echo "Starting Dash app..."
echo ""
echo "🚀 The app will be available at: http://localhost:8050"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python app.py

