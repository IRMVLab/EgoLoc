"""
Configuration management for EgoLoc3D
"""
from pathlib import Path
from typing import Optional, Dict, Any
import os


class Config:
    """Centralized configuration for EgoLoc3D pipeline"""
    
    # Paths
    REPO_ROOT: Optional[Path] = None
    VDA_DIR: Optional[Path] = None
    HAMER_DIR: Optional[Path] = None
    
    # Depth estimation
    DEPTH_SCALE_M: float = 3.0  # pixel value 255 ↔ 3 m (linear scaling)
    
    # ICP registration
    ICP_THRESHOLD: float = 0.03  # metres
    
    # Speed computation
    TIME_INTERVAL: float = 1.0  # seconds per frame
    
    # VLM settings
    MAX_TOKENS: int = 500
    TEMPERATURE: float = 0.1
    TOP_P: float = 0.5
    MAX_RETRIES: int = 5
    
    # Frame selection
    GRID_SIZE: int = 3
    MAX_FEEDBACKS: int = 1
    MIN_PEAK_DISTANCE: int = 2
    
    # Speed filtering (for adaptive methods)
    SAVGOL_POLYORDER: int = 2
    SPLINE_S_FACTOR: float = 1e-3  # multiplied by mean_speed
    MIN_PROMINENCE_FACTOR: float = 0.3  # multiplied by mean_speed
    
    @classmethod
    def find_repo_root(cls) -> Path:
        """Find EgoLoc repository root"""
        if cls.REPO_ROOT is not None:
            return cls.REPO_ROOT
        
        # Start from egoloc3d package directory
        current = Path(__file__).resolve().parent.parent  # Go up to EgoLoc root
        if (current / "Video-Depth-Anything").exists():
            cls.REPO_ROOT = current
            return cls.REPO_ROOT
        
        # Fallback: search up the directory tree
        current = Path(__file__).resolve().parent
        while current != current.parent:
            if (current / "Video-Depth-Anything").exists():
                cls.REPO_ROOT = current
                return cls.REPO_ROOT
            current = current.parent
        
        raise RuntimeError("Could not find EgoLoc repository root (Video-Depth-Anything not found)")
    
    @classmethod
    def get_vda_dir(cls) -> Path:
        """Get Video-Depth-Anything directory"""
        if cls.VDA_DIR is not None:
            return cls.VDA_DIR
        cls.VDA_DIR = cls.find_repo_root() / "Video-Depth-Anything"
        return cls.VDA_DIR
    
    @classmethod
    def get_hamer_dir(cls) -> Path:
        """Get HaMeR directory"""
        if cls.HAMER_DIR is not None:
            return cls.HAMER_DIR
        repo_root = cls.find_repo_root()
        # Check common locations
        for possible in [repo_root / "hamer", repo_root / "HaMeR"]:
            if possible.exists():
                cls.HAMER_DIR = possible
                return cls.HAMER_DIR
        raise RuntimeError("Could not find HaMeR directory")
    
    @classmethod
    def load_credentials(cls, env_path: str) -> Dict[str, Any]:
        """Load API credentials from .env file"""
        import dotenv
        creds = dotenv.dotenv_values(env_path)
        
        # Ensure required keys exist (with empty strings as defaults)
        required = ["OPENAI_API_KEY", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_DEPLOYMENT_NAME"]
        for key in required:
            if key not in creds:
                creds[key] = ""
        
        return creds

