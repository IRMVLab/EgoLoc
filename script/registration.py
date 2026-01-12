"""
ICP-based hand position registration for global alignment
"""
import os
import json
import numpy as np
import open3d as o3d
from pathlib import Path
from typing import Dict, List, Optional

from .config import Config


def register_hand_positions(
    pcd_root: str,
    hand3d_root: str,
    save_reg_root: str,
    threshold: Optional[float] = None
) -> None:
    """
    Align per-frame 3D hand positions to the first frame using point-to-point ICP.
    
    Args:
        pcd_root: Directory containing colored point clouds organized as
            `<video>/frame.ply`
        hand3d_root: Directory with camera-coordinate 3D hand JSON files
        save_reg_root: Output directory for globally registered hand trajectories
        threshold: ICP correspondence distance threshold in metres (default from Config)
    """
    if threshold is None:
        threshold = Config.ICP_THRESHOLD
    
    os.makedirs(save_reg_root, exist_ok=True)
    pcd_root_path = Path(pcd_root)
    hand3d_root_path = Path(hand3d_root)
    save_reg_root_path = Path(save_reg_root)
    
    # Iterate each video folder
    for video_name in sorted(os.listdir(pcd_root_path)):
        pcd_dir = pcd_root_path / video_name
        hand3d_path = hand3d_root_path / f"{video_name}.json"
        
        if not pcd_dir.is_dir() or not hand3d_path.exists():
            continue
        
        print(f"[ICP] Registering hand positions for {video_name}...")
        
        # Load camera-frame hand positions
        with open(hand3d_path, 'r') as f:
            hand3d = json.load(f)
        
        # List PLY files in order
        plys = sorted(
            [f for f in os.listdir(pcd_dir) if f.endswith('.ply')],
            key=lambda x: int(Path(x).stem)
        )
        
        if len(plys) == 0:
            print(f"[ICP] Warning: No PLY files found in {pcd_dir}")
            continue
        
        first_pcd = None  # Reference cloud (frame 0)
        reg_hand_dict: Dict[str, List[float]] = {}  # Output: frameID → [x,y,z]
        prev_pcd = None
        
        for i, ply_name in enumerate(plys):
            frame_id = str(i)  # 0-based frame indexing
            ply_path = pcd_dir / ply_name
            
            # Load point cloud
            pcd = o3d.io.read_point_cloud(str(ply_path))
            pts = np.asarray(pcd.points)
            
            if len(pts) == 0:
                print(f"[ICP] Warning: Empty point cloud at {ply_path}")
                continue
            
            pcd.points = o3d.utility.Vector3dVector(pts)
            
            if i == 0:
                # First frame: no registration, store original hand position
                if frame_id in hand3d or str(i + 1) in hand3d:
                    # Handle both 0-based and 1-based indexing
                    h0_key = frame_id if frame_id in hand3d else str(i + 1)
                    h0 = np.array(hand3d[h0_key])
                    reg_hand_dict[frame_id] = h0.tolist()
            else:
                # Cache reference once
                if first_pcd is None:
                    first_pcd = prev_pcd
                
                # ICP registration: current → first frame
                reg = o3d.pipelines.registration.registration_icp(
                    pcd,  # source cloud
                    first_pcd,  # target cloud
                    threshold,  # correspondence distance (m)
                    np.eye(4),  # initial guess = identity
                    o3d.pipelines.registration.TransformationEstimationPointToPoint()
                )
                T = reg.transformation  # 4×4 rigid transform
                
                # Transform hand position
                h_key = frame_id if frame_id in hand3d else str(i + 1)
                h = np.array(hand3d.get(h_key, [np.nan, np.nan, np.nan]))
                
                if not np.any(np.isnan(h)):
                    h4 = np.append(h, 1)  # homogeneous coordinate
                    h_reg = (T @ h4)[:3].tolist()  # apply ICP transform
                    reg_hand_dict[frame_id] = h_reg
            
            prev_pcd = pcd
        
        # Save registered trajectory
        output_path = save_reg_root_path / f"{video_name}.json"
        with open(output_path, 'w') as f:
            json.dump(reg_hand_dict, f, indent=2)
        
        print(f"[ICP] Registered {len(reg_hand_dict)} frames for {video_name}")


def load_registered_hand_positions(
    reg_json_path: str,
    use_1based: bool = True
) -> Dict[str, List[float]]:
    """
    Load registered hand positions from JSON file.
    
    Args:
        reg_json_path: Path to registered hand positions JSON
        use_1based: If True, converts frame IDs to 1-based indexing
    
    Returns:
        Dict mapping frame_id (string) to [x, y, z]
    """
    with open(reg_json_path, 'r') as f:
        data = json.load(f)
    
    if use_1based:
        # Convert 0-based to 1-based if needed
        result = {}
        for k, v in data.items():
            # If key is "0", "1", etc., convert to 1-based
            try:
                frame_idx = int(k)
                result[str(frame_idx + 1)] = v
            except ValueError:
                result[k] = v
        return result
    
    return data

