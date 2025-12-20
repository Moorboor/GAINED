#!/bin/bash
# GAINED Therapy Session Analysis App - Quick Start Script

echo "=================================="
echo "GAINED - Therapy Session Analysis"
echo "=================================="
echo ""

# Navigate to project directory
cd "$(dirname "$0")"

# Navigate to app directory
cd app

# Run the app using venv Python directly
echo "Starting Dash app..."
echo ""
echo "🚀 The app will be available at: http://localhost:8050"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

../venv/bin/python app.py

