"""
Core pipeline engine for 3D temporal interaction localization
"""
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .config import Config
from .depth_estimation import generate_depth_video_vda
from .pointcloud import generate_pointclouds
from .hand_detection import HaMeRHandDetector
from .registration import register_hand_positions, load_registered_hand_positions
from .speed_computation import compute_speeds_from_registered_dict, save_speed_json
from .frame_selection import (
    extract_local_minima_frames_by_type,
    adaptive_sample_speed,
    sample_keyframe_around_minima,
    select_frames_near_average,
    get_contact_separation_pairs
)
from .vlm_inference import (
    scene_understanding,
    PROMPT_CONTACT,
    PROMPT_SEPARATION,
    PROMPT_STATE
)
from .feedback import refine_frame_with_feedback
from .utils import (
    create_frame_grid_with_keyframe,
    create_frame_grid_state
)


def extract_3d_speed_and_visualize(
    video_path: str,
    output_dir: str,
    credentials: Dict,
    *,
    device: str = "cuda",
    encoder: str = "vits"
) -> Tuple[Dict[int, float], str, str]:
    """
    Extract 3D hand speed from video using HaMeR + VDA + ICP.
    
    Args:
        video_path: Path to input video
        output_dir: Output directory for intermediate and final results
        credentials: API credentials (not used here, but kept for compatibility)
        device: Computation device ('cuda' or 'cpu')
        encoder: VDA encoder ('vits' or 'vitl')
    
    Returns:
        (speed_dict, speed_json_path, speed_vis_path)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_name = Path(video_path).stem
    
    # Step 1: Generate depth maps using VDA
    depth_dir = output_dir / "depth"
    generate_depth_video_vda(video_path, str(depth_dir), device=device, encoder=encoder)
    
    # Step 2: Generate point clouds
    pcd_dir = output_dir / "pointclouds" / video_name
    if not (pcd_dir / "0.ply").exists():
        generate_pointclouds(depth_dir, video_path, pcd_dir)
    else:
        print(f"[Pipeline] Reusing cached point clouds in {pcd_dir}")
    
    # Step 3: Detect hand positions using HaMeR
    print("[Pipeline] Detecting hand positions using HaMeR...")
    hand_detector = HaMeRHandDetector(device=device)
    cam_hand_dir = output_dir / "hand3d_cam"
    cam_hand_dir.mkdir(exist_ok=True)
    cam_hand_json = cam_hand_dir / f"{video_name}.json"
    
    # Process video frames (hand positions saved to JSON)
    hand_detector.process_video(video_path, str(cam_hand_json))
    
    # Step 4: ICP registration
    print("[Pipeline] Registering hand positions using ICP...")
    reg_out_dir = output_dir / "registered_hands"
    register_hand_positions(
        str(pcd_dir.parent),
        str(cam_hand_dir),
        str(reg_out_dir),
        threshold=Config.ICP_THRESHOLD
    )
    
    # Step 5: Compute speeds from registered positions
    reg_hand_json = reg_out_dir / f"{video_name}.json"
    reg_hand = load_registered_hand_positions(str(reg_hand_json), use_1based=True)
    
    speed_dict = compute_speeds_from_registered_dict(
        reg_hand,
        time_interval=Config.TIME_INTERVAL
    )
    
    # Save speed JSON
    speed_json_path = output_dir / f"{video_name}_speed.json"
    speed_list = [(frame + 1, speed) for frame, speed in sorted(speed_dict.items())]
    save_speed_json(speed_list, str(speed_json_path))
    
    # Visualize speed curve
    speed_vis_path = output_dir / f"{video_name}_speed_vis.png"
    plt.figure(figsize=(12, 4))
    plt.plot(list(speed_dict.keys()), list(speed_dict.values()), label="3D Hand Speed")
    plt.xlabel("Frame")
    plt.ylabel("Speed (m/s)")
    plt.title(f"3D Hand Speed - {video_name}")
    plt.tight_layout()
    plt.savefig(speed_vis_path)
    plt.close()
    
    print(f"[Pipeline] Speed extraction complete: {speed_json_path}")
    
    return speed_dict, str(speed_json_path), str(speed_vis_path)


def process_task(
    credentials: Dict,
    video_path: str,
    speed_json_path: str,
    grid_size: int,
    total_frames: int,
    max_feedback: int = 1,
    use_feedback: bool = True,
    video_type: str = "short",
    use_vda: bool = False
) -> List[Tuple[str, int]]:
    """
    Process task to identify Contact/Separation events.
    
    Returns:
        List of (event_type, frame_index) tuples
    """
    video_name = Path(video_path).stem
    
    # Load speed data
    with open(speed_json_path, 'r') as f:
        speed_data = json.load(f)
    
    all_frames = np.array([x[0] for x in speed_data])
    all_speeds = np.array([x[1] for x in speed_data])
    
    # Extract local minima based on video type
    # Use VDA-optimized extraction only for long videos when speed is generated from video (using VDA)
    minima_indices, minima_speeds = extract_local_minima_frames_by_type(
        speed_json_path,
        video_type=video_type,
        video_id=video_name,
        use_vda=use_vda
    )
    
    if len(minima_indices) == 0:
        print("[Pipeline] No minima frames found")
        return []
    
    results = []
    
    # Process each minima
    remaining_minima = minima_indices.copy()
    remaining_speeds = minima_speeds.copy()
    
    while remaining_minima:
        # Sample minima based on speed
        selected_minima = adaptive_sample_speed(remaining_minima, remaining_speeds)
        idx = remaining_minima.index(selected_minima)
        remaining_minima.pop(idx)
        remaining_speeds.pop(idx)
        
        # Sample keyframe around minima
        keyframe_index = sample_keyframe_around_minima(
            selected_minima,
            all_frames,
            all_speeds,
            window=2
        )
        
        # Create state comparison grid
        state_frame_indices, _ = select_frames_near_average(
            [keyframe_index], 3, total_frames, []
        )
        state_indices = [state_frame_indices[0], state_frame_indices[-1]]
        image_state = create_frame_grid_state(video_path, state_indices)
        
        # Check state
        state = scene_understanding(
            credentials, image_state, PROMPT_STATE, principle="state"
        )
        
        if state not in ["Contact", "Separation"]:
            continue
        
        # Create selection grid
        frame_indices, _ = select_frames_near_average(
            [keyframe_index], grid_size, total_frames, []
        )
        image = create_frame_grid_with_keyframe(video_path, frame_indices, grid_size)
        
        # Select frame using VLM
        prompt = PROMPT_CONTACT if state == "Contact" else PROMPT_SEPARATION
        description = scene_understanding(credentials, image, prompt)
        
        if description == -1:
            continue
        
        index_specified = max(min(int(description) - 1, len(frame_indices) - 1), 0)
        final_frame = frame_indices[index_specified]
        
        # Feedback loop (optional) with in-context learning
        if use_feedback and max_feedback > 0:
            success, final_frame = refine_frame_with_feedback(
                credentials=credentials,
                video_path=video_path,
                initial_frame=final_frame,
                state=state,
                all_frames=all_frames,
                all_speeds=all_speeds,
                total_frames=total_frames,
                grid_size=grid_size,
                max_feedback=max_feedback
            )
            
            if not success:
                print(f"[Pipeline] ⚠️ Feedback failed, skipping this minima (state: {state}, frame: {final_frame})")
                continue
        
        results.append((state, final_frame))
        print(f"[Pipeline] Detected {state} @ frame {final_frame}")
    
    return results


def convert_video(
    video_path: str,
    credentials: Dict,
    output_dir: str,
    speed_json_path: str,
    grid_size: int = 3,
    max_feedback: int = 1,
    use_feedback: bool = True,
    video_type: str = "short",
    use_vda: bool = False
) -> List[Tuple[int, int]]:
    """
    Main conversion function: find contact/separation pairs.
    
    Returns:
        List of (contact_frame, separation_frame) tuples
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    # Process task to get all events
    results = process_task(
        credentials,
        video_path,
        speed_json_path,
        grid_size,
        total_frames,
        max_feedback,
        use_feedback,
        video_type,
        use_vda
    )
    
    # Load speed data for pairing
    with open(speed_json_path, 'r') as f:
        speed_data = json.load(f)
    speed_list = [(frame, speed) for frame, speed in speed_data]
    
    # Pair contacts and separations
    pairs = get_contact_separation_pairs(results, speed_list, video_type)
    
    return pairs

