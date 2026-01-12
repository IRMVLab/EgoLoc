"""
Depth estimation using Video-Depth-Anything
"""
import os
import subprocess
import numpy as np
from pathlib import Path
from typing import List, Optional

from .config import Config


def _is_invalid_inv(inv: np.ndarray) -> bool:
    """
    Return True when the inverse-depth tensor is all-NaN/Inf or nearly flat.
    """
    return (not np.isfinite(inv).any()) or np.nanstd(inv) < 1e-4


def _unpack_depth_npz(depth_dir: Path) -> None:
    """Convert VDA's *_depths.npz to individual .npy, skipping ones that exist."""
    npz_files = list(depth_dir.glob("*_depths.npz"))
    if not npz_files:
        return
    
    arr = np.load(npz_files[0])["depths"]  # (N, H, W)
    created = 0
    for i, depth in enumerate(arr):
        out_f = depth_dir / f"pred_depth_{i:06d}.npy"
        if out_f.exists():
            continue
        np.save(out_f, depth.astype(np.float32))
        created += 1
    if created:
        print(f"[VDA] Unpacked {created} new depth tensors")


def _invalid_depth_indices(depth_dir: Path) -> List[int]:
    """Return a list of frame indices whose depth tensors are unusable."""
    bad_idx = []
    for f in depth_dir.glob("pred_depth_*.npy"):
        idx = int(f.stem.split("_")[-1])
        inv = np.load(f, mmap_mode="r")
        if _is_invalid_inv(inv):
            bad_idx.append(idx)
    return bad_idx


def _remove_depth_tensors(depth_dir: Path, indices: List[int]) -> None:
    """Delete pred_depth_XXXXXX.npy for the given indices (if they exist)."""
    for idx in indices:
        f = depth_dir / f"pred_depth_{idx:06d}.npy"
        if f.exists():
            f.unlink()


def load_depth(depth_dir: Path, idx: int) -> Optional[np.ndarray]:
    """
    Load depth tensor for a frame and convert to metric depth in metres.
    
    VDA stores **inverse depth** (bigger = nearer).
    Convert to metric depth in metres.
    
    Args:
        depth_dir: Directory containing pred_depth_*.npy files
        idx: Frame index
    
    Returns:
        Depth map in metres (H, W), or None if invalid/missing
    """
    f = depth_dir / f"pred_depth_{idx:06d}.npy"
    if not f.exists():
        return None
    
    inv = np.load(f).astype(np.float32)  # (H, W)
    if _is_invalid_inv(inv):
        return None
    
    depth = Config.DEPTH_SCALE_M / (inv + 1e-6)  # invert to metric depth
    return depth


def generate_depth_video_vda(
    video_path: str,
    depth_out_path: str,
    *,
    device: str = "cuda",
    encoder: str = "vits",
    max_repairs: int = 5
) -> Path:
    """
    Run Video-Depth-Anything to generate depth maps with auto-repair.
    
    Args:
        video_path: Path to input video
        depth_out_path: Output directory for depth tensors
        device: Computation device ('cuda' or 'cpu')
        encoder: VDA encoder ('vits' or 'vitl')
        max_repairs: Maximum number of repair attempts
    
    Returns:
        Path to depth output directory
    """
    video_path = Path(video_path).resolve()
    depth_out_path = Path(depth_out_path).resolve()
    depth_out_path.mkdir(parents=True, exist_ok=True)
    
    vda_dir = Config.get_vda_dir()
    
    # Check if depth already exists
    if (depth_out_path / "pred_depth_000000.npy").exists():
        print(f"[VDA] Reusing cached depth outputs in {depth_out_path}")
        
        # Check quality and repair if needed
        for attempt in range(max_repairs):
            bad_idx = _invalid_depth_indices(depth_out_path)
            if not bad_idx:
                break
            
            pct = len(bad_idx) / len(list(depth_out_path.glob("pred_depth_*.npy"))) * 100
            print(f"[VDA] {len(bad_idx)} invalid depth tensors ({pct:.1f}%) - repairing")
            
            _remove_depth_tensors(depth_out_path, bad_idx)
            
            # Re-run VDA for missing frames
            cmd = [
                "python",
                "run.py",
                "--input_video",
                str(video_path),
                "--output_dir",
                str(depth_out_path),
                "--encoder",
                encoder,
                "--save_npz",
            ]
            if device == "cpu":
                cmd.append("--fp32")
            
            env = os.environ.copy()
            env["PYTHONPATH"] = f"{vda_dir}:{env.get('PYTHONPATH', '')}"
            
            print(f"[VDA] Repair run: {' '.join(cmd)}")
            subprocess.run(cmd, check=True, cwd=vda_dir, env=env)
            _unpack_depth_npz(depth_out_path)
        else:
            print(f"[VDA] Warning: Repair failed after {max_repairs} attempts")
        
        return depth_out_path
    
    # Generate depth maps
    print("[VDA] Generating depth maps...")
    cmd = [
        "python",
        "run.py",
        "--input_video",
        str(video_path),
        "--output_dir",
        str(depth_out_path),
        "--encoder",
        encoder,
        "--save_npz",
    ]
    if device == "cpu":
        cmd.append("--fp32")
    
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{vda_dir}:{env.get('PYTHONPATH', '')}"
    
    print(f"[VDA] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=vda_dir, env=env)
    
    # Unpack NPZ to individual .npy files
    _unpack_depth_npz(depth_out_path)
    
    if not any(depth_out_path.glob("pred_depth_*.npy")):
        raise RuntimeError(
            "[VDA] No pred_depth_*.npy tensors were produced - check VDA output above."
        )
    
    print(f"[VDA] Depth generation complete: {depth_out_path}")
    return depth_out_path
