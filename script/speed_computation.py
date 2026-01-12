"""
Speed computation from 3D hand positions with interpolation
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path


def compute_speeds_from_dict(
    hand_dict: Dict[str, List[float]],
    time_interval: float = 1.0
) -> List[Tuple[int, float]]:
    """
    Compute speeds from 3D hand position dictionary with interpolation
    
    Args:
        hand_dict: Dict mapping frame index (str or int) to [x, y, z] coordinates
        time_interval: Time interval between frames in seconds
    
    Returns:
        List of [frame_index, speed] tuples
    """
    # Convert keys to integers and sort
    frames = sorted(int(k) for k in hand_dict.keys())
    if not frames:
        return []
    
    # Detect dimension (assume all frames have same dimension)
    first_coords = None
    for v in hand_dict.values():
        if isinstance(v, (list, tuple)) and len(v) > 0:
            first_coords = v
            break
    if first_coords is None:
        return []
    dim = len(first_coords)
    
    # Build complete frame sequence
    min_f, max_f = frames[0], frames[-1]
    all_frames = np.arange(min_f, max_f + 1, dtype=int)
    
    # Build position array, fill missing frames with NaN
    pos = np.full((len(all_frames), dim), np.nan, dtype=float)
    frame_to_idx = {f: i for i, f in enumerate(all_frames)}
    for f_str, coords in hand_dict.items():
        f = int(f_str)
        if f in frame_to_idx:
            pos[frame_to_idx[f]] = coords
    
    # Linear interpolation for each coordinate axis to fill NaN values
    for d in range(dim):
        arr = pos[:, d]
        nans = np.isnan(arr)
        if np.all(nans):
            continue
        valid_idx = np.where(~nans)[0]
        valid_vals = arr[valid_idx]
        if len(valid_idx) > 1:
            interp_vals = np.interp(np.arange(len(arr)), valid_idx, valid_vals)
            arr[nans] = interp_vals[nans]
        pos[:, d] = arr
    
    # Compute speeds
    speeds = []
    # First frame speed is 0
    speeds.append((int(all_frames[0]), 0.0))
    
    for i in range(1, len(all_frames)):
        prev = pos[i - 1]
        curr = pos[i]
        if np.any(np.isnan(prev)) or np.any(np.isnan(curr)):
            speeds.append((int(all_frames[i]), float('nan')))
        else:
            dist = np.linalg.norm(curr - prev)
            speed = float(dist / time_interval)
            speeds.append((int(all_frames[i]), speed))
    
    return speeds


def compute_speeds_from_registered_dict(
    registered_dict: Dict[str, List[float]],
    time_interval: float = 1.0
) -> Dict[int, float]:
    """
    Compute speeds from registered (ICP-aligned) hand positions
    
    Args:
        registered_dict: Dict mapping frame_id (str) to registered [x, y, z]
        time_interval: Time interval between frames in seconds
    
    Returns:
        Dict mapping frame_index (int, 0-based) to speed
    """
    speeds = compute_speeds_from_dict(registered_dict, time_interval)
    
    # Convert to 0-based frame indexing for compatibility
    speed_dict = {}
    for frame_id, speed in speeds:
        # frame_id is 1-based from dict, convert to 0-based
        speed_dict[frame_id - 1] = speed
    
    return speed_dict


def save_speed_json(speed_data: List[Tuple[int, float]], output_path: str):
    """
    Save speed data to JSON file
    
    Args:
        speed_data: List of (frame_index, speed) tuples
        output_path: Path to save JSON file
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to list format compatible with original code
    json_data = [
        [int(frame), None if np.isnan(speed) else float(speed)]
        for frame, speed in speed_data
    ]
    
    with open(output_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"[Speed] Saved speed data to {output_path}")


def load_speed_json(json_path: str) -> List[Tuple[int, float]]:
    """
    Load speed data from JSON file
    
    Args:
        json_path: Path to JSON file
    
    Returns:
        List of (frame_index, speed) tuples
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Handle both list format [[frame, speed], ...] and dict format
    if isinstance(data, list):
        return [(int(item[0]), float(item[1])) if item[1] is not None else (int(item[0]), np.nan) 
                for item in data]
    elif isinstance(data, dict):
        return [(int(k), float(v)) if v is not None else (int(k), np.nan) 
                for k, v in data.items()]
    else:
        raise ValueError(f"Unsupported JSON format in {json_path}")

