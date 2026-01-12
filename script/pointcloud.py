"""
Point cloud generation from RGB + Depth
"""
import cv2
import numpy as np
import open3d as o3d
from pathlib import Path
from typing import Optional, Tuple

from .depth_estimation import load_depth


def generate_pointclouds(
    depth_dir: Path,
    video_path: str,
    pcd_out_dir: Path,
    intrinsics: Optional[Tuple[float, float, float, float]] = None
) -> None:
    """
    Create colored .ply point clouds for every frame with valid depth.
    
    Args:
        depth_dir: Directory with pred_depth_*.npy files (in metres)
        video_path: Original RGB video (for colors)
        pcd_out_dir: Output directory for <idx>.ply files
        intrinsics: Optional (fx, fy, cx, cy). If None, uses default assumptions.
    """
    pcd_out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    # Setup camera intrinsics
    if intrinsics is None:
        # Default: assume fx = fy = max(H, W), principal point at center
        fx = fy = max(H, W)
        cx, cy = W / 2.0, H / 2.0
    else:
        fx, fy, cx, cy = intrinsics
    
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(W, H, fx, fy, cx, cy)
    
    depth_files = sorted(depth_dir.glob("pred_depth_*.npy"))
    print(f"[PCD] Generating point clouds from {len(depth_files)} depth maps...")
    
    created = 0
    for dfile in depth_files:
        idx = int(dfile.stem.split("_")[-1])
        
        # Read RGB frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame_bgr = cap.read()
        if not ok:
            continue
        
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        # Load depth map
        depth_m = load_depth(depth_dir, idx)
        if depth_m is None:
            continue
        
        # Open3D expects depth in millimetres
        depth_o3d = o3d.geometry.Image((depth_m * 1000).astype(np.uint16))
        color_o3d = o3d.geometry.Image(frame_rgb)
        
        # Create RGBD image
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d,
            depth_o3d,
            depth_scale=1000.0,
            depth_trunc=4.0,
            convert_rgb_to_intensity=False
        )
        
        # Generate point cloud
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
        
        # Save point cloud
        pcd_path = pcd_out_dir / f"{idx}.ply"
        o3d.io.write_point_cloud(str(pcd_path), pcd, write_ascii=False)
        created += 1
        
        if (created + 1) % 50 == 0:
            print(f"[PCD] Generated {created + 1} point clouds...")
    
    cap.release()
    print(f"[PCD] Generated {created} point clouds in {pcd_out_dir}")

