#!/usr/bin/env python3
"""
Simple script to run the Streamlit app for the subtitle search engine.
"""
import os
import subprocess
import sys
from pathlib import Path

def main():
    # Get the current directory
    current_dir = Path(__file__).parent.absolute()
    
    # Create necessary directories if they don't exist
    os.makedirs(current_dir / "data", exist_ok=True)
    os.makedirs(current_dir / "models", exist_ok=True)
    os.makedirs(current_dir / "src", exist_ok=True)
    os.makedirs(current_dir / "app", exist_ok=True)
    
    # Check if the source files exist, otherwise create them
    required_files = [
        current_dir / "src" / "search_engine.py",
        current_dir / "src" / "enhanced_search.py",
        current_dir / "app" / "streamlit_app.py"
    ]
    
    missing_files = [f for f in required_files if not f.exists()]
    if missing_files:
        print(f"Missing required files: {', '.join(str(f) for f in missing_files)}")
        print("Please ensure all source files are in the correct locations.")
        return 1
    
    # Install requirements if not already installed
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit", "numpy", "pandas", 
                               "scikit-learn", "matplotlib", "seaborn", "nltk", "tqdm", "sentence-transformers"])
        print("Dependencies installed successfully.")
    except subprocess.CalledProcessError:
        print("Failed to install dependencies. Please install them manually:")
        print("pip install streamlit numpy pandas scikit-learn matplotlib seaborn nltk tqdm sentence-transformers")
    
    # Run the Streamlit app
    app_path = current_dir / "app" / "streamlit_app.py"
    print(f"Starting Streamlit app: {app_path}")
    subprocess.call(["streamlit", "run", str(app_path)])

if __name__ == "__main__":
    sys.exit(main())
