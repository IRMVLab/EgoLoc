"""
EgoLoc3D - Main entry point for 3D temporal interaction localization
A wrapper script for easier command-line usage
"""
import sys
from pathlib import Path

# Add current directory to path if needed
sys.path.insert(0, str(Path(__file__).parent))

from script.cli import main

if __name__ == "__main__":
    main()

