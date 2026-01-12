"""
Frame selection utilities for temporal interaction localization
"""
import numpy as np
import json
import re
from pathlib import Path
from typing import List, Tuple, Optional
from scipy.signal import savgol_filter, find_peaks
from scipy.interpolate import UnivariateSpline


def extract_local_minima_frames(
    speed_json_path: str,
    video_id: Optional[str] = None
) -> Tuple[List[int], List[float]]:
    """
    Extract local minima frames from speed curve (for short videos).
    
    Args:
        speed_json_path: Path to speed JSON file (list of [frame, speed])
        video_id: Optional video identifier for logging
    
    Returns:
        (minima_frame_indices, minima_speeds) - both sorted by speed (ascending)
    """
    thresh = 30
    savgol_polyorder = 2
    spline_s = 1e-4
    savgol_mode = 'nearest'
    
    if not Path(speed_json_path).exists():
        print(f"File not found: {speed_json_path}")
        return [], []
    
    if video_id is None:
        # Extract video_id from path
        video_name = Path(speed_json_path).stem.replace("_with_speed", "")
        match = re.search(r'\d+', video_name)
        video_id = match.group(0) if match else video_name
    
    try:
        with open(speed_json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"JSON read failed: {speed_json_path}, error: {e}")
        return [], []
    
    if not isinstance(data, list) or len(data) == 0:
        print(f"Data is empty: {speed_json_path}")
        return [], []
    
    filtered_data = [
        (frame, speed)
        for frame, speed in data
        if isinstance(speed, (int, float)) and not np.isnan(speed) and 0 < speed < thresh
    ]
    
    if len(filtered_data) < 4:
        print(f"Video {video_id} has too few data points ({len(filtered_data)}), skipping!")
        return [], []
    
    frames, speeds = zip(*filtered_data)
    data_len = len(speeds)
    
    # Savitzky-Golay filtering
    window_length = min(7, data_len)
    if window_length % 2 == 0:
        window_length -= 1
    if window_length < 3:
        window_length = 3
    if window_length >= data_len:
        window_length = data_len - 1 if data_len % 2 == 0 else data_len
        if window_length < 3:
            window_length = 3
    
    try:
        speeds_smooth_sg = savgol_filter(
            speeds,
            window_length=window_length,
            polyorder=min(savgol_polyorder, window_length - 1),
            mode=savgol_mode
        )
    except Exception as e:
        print(f"Savitzky-Golay failed: {speed_json_path}, error: {e}")
        return [], []
    
    # Spline interpolation
    try:
        spl = UnivariateSpline(frames, speeds_smooth_sg, s=spline_s)
        frames_smooth = np.linspace(min(frames), max(frames), 300)
        speeds_smooth = spl(frames_smooth)
    except Exception as e:
        print(f"Spline interpolation failed: {speed_json_path}, error: {e}")
        return [], []
    
    # Minima detection
    try:
        minima_indices, _ = find_peaks(-speeds_smooth)
        extrema_frames = frames_smooth[minima_indices]
    except Exception as e:
        print(f"Peak detection failed: {speed_json_path}, error: {e}")
        return [], []
    
    if len(extrema_frames) == 0:
        print(f"Video {video_id} found no minima!")
        return [], []
    
    # Map back to original frames and speeds
    extrema_frames_int = np.rint(extrema_frames).astype(int)
    extrema_frame_speed_pairs = []
    
    for ef in extrema_frames_int:
        close_indices = np.where(np.isclose(frames, ef))[0]
        if len(close_indices) > 0:
            idx = close_indices[0]
            matched_frame = frames[idx]
            speed = speeds[idx]
            extrema_frame_speed_pairs.append((matched_frame, speed))
        else:
            print(f"Minima frame {ef} not in original frame list, trying to find nearest point")
            idx_nearest = np.argmin(np.abs(frames - ef))
            nearest_frame = frames[idx_nearest]
            speed = speeds[idx_nearest]
            extrema_frame_speed_pairs.append((nearest_frame, speed))
    
    if not extrema_frame_speed_pairs:
        print(f"Video {video_id} matched no valid minima frames!")
        return [], []
    
    # Sort by speed (ascending)
    sorted_pairs = sorted(extrema_frame_speed_pairs, key=lambda x: x[1])
    sorted_frames = [pair[0] for pair in sorted_pairs]
    sorted_speeds = [pair[1] for pair in sorted_pairs]
    
    print(f"Video {video_id} minima frame indices (sorted by speed): {sorted_frames}")
    
    return sorted_frames, sorted_speeds


def extract_local_minima_frames_adaptive(
    speed_json_path: str,
    video_id: Optional[str] = None
) -> Tuple[List[int], List[float]]:
    """
    Adaptive version: Extract local minima frames based on total frames and speed (for long videos).
    
    Args:
        speed_json_path: Path to speed JSON file (list of [frame, speed])
        video_id: Optional video identifier for logging
    
    Returns:
        (minima_frame_indices, minima_speeds) - both sorted by speed (ascending)
    """
    if not Path(speed_json_path).exists():
        print(f"File not found: {speed_json_path}")
        return [], []
    
    if video_id is None:
        video_name = Path(speed_json_path).stem.replace("_with_speed", "")
        video_id = video_name
    
    try:
        with open(speed_json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"JSON read failed: {speed_json_path}, error: {e}")
        return [], []
    
    if not isinstance(data, list) or len(data) == 0:
        print(f"Data is empty or format error: {speed_json_path}")
        return [], []
    
    frames_all, speeds_all = zip(*data)
    frames_all = np.array(frames_all, float)
    speeds_all = np.array(speeds_all, float)
    
    mask = np.isfinite(speeds_all) & (speeds_all > 0)
    frames = frames_all[mask]
    speeds = speeds_all[mask]
    N = len(speeds)
    
    if N < 4:
        print(f"Too few valid data points: {N}")
        return [], []
    
    # Adaptive parameters based on data length
    polyorder = max(2, int(N / 13))
    window_length = max(8, int(N / 7))
    if window_length % 2 == 0:
        window_length += 1
    window_length = min(window_length, N - (1 if N % 2 == 0 else 0))
    
    mean_speed = np.mean(speeds)
    spline_s = mean_speed * 1e-3
    min_prominence = mean_speed * 0.3
    min_peak_distance = max(int(N * 0.03), 2)
    
    # Savitzky-Golay filtering
    try:
        speeds_sg = savgol_filter(
            speeds,
            window_length=window_length,
            polyorder=min(polyorder, window_length - 1),
            mode='nearest'
        )
    except Exception:
        speeds_sg = speeds
    
    # Spline interpolation
    try:
        spl = UnivariateSpline(frames, speeds_sg, s=spline_s)
        frames_smooth = np.linspace(frames.min(), frames.max(), max(300, N))
        speeds_smooth = spl(frames_smooth)
    except Exception:
        frames_smooth, speeds_smooth = frames, speeds_sg
    
    # Peak detection
    peaks, props = find_peaks(
        -speeds_smooth,
        prominence=min_prominence,
        distance=min_peak_distance
    )
    
    cand_frames = np.unique(np.rint(frames_smooth[peaks]).astype(int))
    pairs = []
    
    for f in cand_frames:
        idx = np.argmin(np.abs(frames - f))
        pairs.append((int(frames[idx]), float(speeds[idx])))
    
    pairs.sort(key=lambda x: x[1])
    
    # Filter by minimum distance
    selected = []
    for f, s in pairs:
        if all(abs(f - sf) >= min_peak_distance for sf, _ in selected):
            selected.append((f, s))
    
    result = [f for f, s in selected]
    result_speeds = [s for f, s in selected]
    
    print(f"{video_id} extracted {len(result)} minima frames: {result}")
    
    return result, result_speeds


def extract_local_minima_frames_adaptive_long_vda(
    speed_json_path: str,
    video_id: Optional[str] = None
) -> Tuple[List[int], List[float]]:
    """
    Adaptive version for VDA data: Extract local minima frames for long videos using IQR threshold filtering.
    
    This function combines:
    - IQR-based adaptive threshold filtering (for VDA/MoGe2 large value range data)
    - Adaptive window length based on data length (for long videos)
    - Adaptive parameters based on mean speed
    
    Args:
        speed_json_path: Path to speed JSON file (list of [frame, speed])
        video_id: Optional video identifier for logging
    
    Returns:
        (minima_frame_indices, minima_speeds) - both sorted by speed (ascending)
    """
    savgol_polyorder = 2
    savgol_mode = 'nearest'
    
    if not Path(speed_json_path).exists():
        print(f"File not found: {speed_json_path}")
        return [], []
    
    if video_id is None:
        video_name = Path(speed_json_path).stem.replace("_with_speed", "")
        video_id = video_name
    
    try:
        with open(speed_json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"JSON read failed: {speed_json_path}, error: {e}")
        return [], []
    
    if not isinstance(data, list) or len(data) == 0:
        print(f"Data is empty: {speed_json_path}")
        return [], []
    
    # Extract all valid speed values for statistics
    all_speeds = [
        speed for _, speed in data
        if isinstance(speed, (int, float)) and not np.isnan(speed) and speed > 0
    ]
    
    if len(all_speeds) < 4:
        print(f"Video {video_id} has too few valid speed data points ({len(all_speeds)}), skipping!")
        return [], []
    
    # Use IQR method to filter outliers and compute adaptive threshold
    speeds_array = np.array(all_speeds)
    Q1 = np.percentile(speeds_array, 25)
    Q3 = np.percentile(speeds_array, 75)
    IQR = Q3 - Q1
    # Use 95th percentile as upper bound, or IQR method upper bound (take the larger one)
    upper_bound_iqr = Q3 + 2.0 * IQR  # Use 2x IQR as upper bound
    upper_bound_percentile = np.percentile(speeds_array, 95)
    adaptive_thresh = max(upper_bound_iqr, upper_bound_percentile)
    
    # Compute mean and median (before filtering outliers)
    mean_speed = np.mean(speeds_array)
    median_speed = np.median(speeds_array)
    
    print(f"📊 Video {video_id} speed statistics: min={speeds_array.min():.3f}, max={speeds_array.max():.3f}, "
          f"mean={mean_speed:.3f}, median={median_speed:.3f}, threshold={adaptive_thresh:.3f}")
    
    # Filter data using adaptive threshold
    filtered_data = [
        (frame, speed)
        for frame, speed in data
        if isinstance(speed, (int, float)) and not np.isnan(speed) and 0 < speed < adaptive_thresh
    ]
    
    if len(filtered_data) < 4:
        print(f"Video {video_id} has too few data points after filtering ({len(filtered_data)}), skipping!")
        return [], []
    
    frames, speeds = zip(*filtered_data)
    frames = np.array(frames, float)
    speeds = np.array(speeds, float)
    N = len(speeds)
    
    # Adaptive parameters based on data length (for long videos)
    polyorder = max(2, int(N / 13))
    window_length = max(8, int(N / 7))
    if window_length % 2 == 0:
        window_length += 1
    window_length = min(window_length, N - (1 if N % 2 == 0 else 0))
    
    # Adaptive parameters based on filtered mean speed
    filtered_mean_speed = np.mean(speeds)
    spline_s = filtered_mean_speed * 1e-3  # Adaptive spline_s parameter
    min_prominence = filtered_mean_speed * 0.3  # Adaptive prominence parameter
    min_peak_distance = max(int(N * 0.03), 2)  # Minimum peak distance, at least 2 frames
    
    # Savitzky-Golay filtering
    try:
        speeds_sg = savgol_filter(
            speeds,
            window_length=window_length,
            polyorder=min(polyorder, window_length - 1),
            mode=savgol_mode
        )
    except Exception as e:
        print(f"Savitzky-Golay failed: {speed_json_path}, error: {e}")
        speeds_sg = speeds
    
    # Spline interpolation
    try:
        spl = UnivariateSpline(frames, speeds_sg, s=spline_s)
        frames_smooth = np.linspace(frames.min(), frames.max(), max(300, N))
        speeds_smooth = spl(frames_smooth)
    except Exception as e:
        print(f"Spline interpolation failed: {speed_json_path}, error: {e}")
        frames_smooth, speeds_smooth = frames, speeds_sg
    
    # Peak detection
    try:
        peaks, props = find_peaks(
            -speeds_smooth,
            prominence=min_prominence,
            distance=min_peak_distance
        )
    except Exception as e:
        print(f"Peak detection failed: {speed_json_path}, error: {e}")
        return [], []
    
    if len(peaks) == 0:
        print(f"Video {video_id} found no minima!")
        return [], []
    
    cand_frames = np.unique(np.rint(frames_smooth[peaks]).astype(int))
    pairs = []
    
    # Map back to original data
    for f in cand_frames:
        idx = np.argmin(np.abs(frames - f))
        pairs.append((int(frames[idx]), float(speeds[idx])))
    
    pairs.sort(key=lambda x: x[1])
    
    # Filter by minimum distance
    selected = []
    for f, s in pairs:
        if all(abs(f - sf) >= min_peak_distance for sf, _ in selected):
            selected.append((f, s))
    
    result = [f for f, s in selected]
    result_speeds = [s for f, s in selected]
    
    print(f"🖨️ {video_id} extracted {len(result)} minima frames: {result}")
    
    return result, result_speeds


def extract_local_minima_frames_by_type(
    speed_json_path: str,
    video_type: str = "short",
    video_id: Optional[str] = None,
    use_vda: bool = False
) -> Tuple[List[int], List[float]]:
    """
    Extract local minima frames based on video type.
    
    Args:
        speed_json_path: Path to speed JSON file
        video_type: 'short' or 'long'
        video_id: Optional video identifier for logging
        use_vda: Whether to use VDA-optimized extraction (for VDA/MoGe2 data with large value ranges)
    
    Returns:
        (minima_frame_indices, minima_speeds) - both sorted by speed (ascending)
    """
    if video_type == "short":
        return extract_local_minima_frames(speed_json_path, video_id)
    else:
        if use_vda:
            return extract_local_minima_frames_adaptive_long_vda(speed_json_path, video_id)
        else:
            return extract_local_minima_frames_adaptive(speed_json_path, video_id)


def adaptive_sample_speed(
    minima_indices: List[int],
    minima_speeds: List[float]
) -> int:
    """
    Sample a minima frame based on speed (lower speed = higher probability).
    
    Args:
        minima_indices: List of frame indices
        minima_speeds: Corresponding speeds
    
    Returns:
        Selected frame index
    """
    if len(minima_indices) == 0:
        raise ValueError("minima_indices is empty")
    
    speeds = np.array(minima_speeds)
    eps = 1e-8
    inv_speeds = 1 / (speeds + eps)
    probabilities = inv_speeds / np.sum(inv_speeds)
    selected_frame = np.random.choice(minima_indices, p=probabilities)
    return int(selected_frame)


def sample_keyframe_around_minima(
    selected_minima: int,
    frames: np.ndarray,
    speeds: np.ndarray,
    window: int = 2
) -> int:
    """
    Sample a keyframe around a minima point using speed-weighted sampling.
    
    Args:
        selected_minima: The selected minima frame
        frames: Array of all frame indices
        speeds: Array of corresponding speeds
        window: Window size (frames before and after minima)
    
    Returns:
        Selected keyframe index
    """
    idx = np.where(frames == selected_minima)[0]
    if len(idx) == 0:
        return selected_minima
    
    idx = idx[0]
    candidate_indices = []
    candidate_speeds = []
    
    for offset in range(-window, window + 1):
        candidate_idx = idx + offset
        if 0 <= candidate_idx < len(frames):
            candidate_indices.append(frames[candidate_idx])
            candidate_speeds.append(speeds[candidate_idx])
    
    if len(candidate_indices) == 0:
        return selected_minima
    
    candidate_speeds = np.array(candidate_speeds)
    eps = 1e-8
    inv_speeds = 1 / (candidate_speeds + eps)
    probabilities = inv_speeds / np.sum(inv_speeds)
    selected_keyframe = np.random.choice(candidate_indices, p=probabilities)
    return int(selected_keyframe)


def select_frames_near_average(
    filter_indices: List[int],
    grid_size: int,
    total_frames: int,
    invalid_list: Optional[List[int]] = None
) -> Tuple[List[int], int]:
    """
    Select frames around average index for grid display.
    
    Args:
        filter_indices: List of candidate frame indices
        grid_size: Grid size (grid_size^2 frames will be selected)
        total_frames: Total number of frames in video
        invalid_list: List of invalid frame indices to exclude
    
    Returns:
        (selected_frame_indices, index_of_average_frame_in_list)
    """
    if invalid_list is None:
        invalid_list = []
    
    if len(filter_indices) == 0:
        return [], 0
    
    avg_index = round(np.mean(filter_indices))
    used_frame_indices = []
    
    if avg_index not in invalid_list:
        used_frame_indices.append(avg_index)
    
    start_index = avg_index
    end_index = avg_index
    
    # Expand around average
    while len(used_frame_indices) < grid_size ** 2:
        if start_index > 0:
            start_index -= 1
            if start_index not in invalid_list:
                used_frame_indices.insert(0, start_index)
        
        if len(used_frame_indices) >= grid_size ** 2:
            break
        
        if end_index < total_frames - 1:
            end_index += 1
            if end_index not in invalid_list:
                used_frame_indices.append(end_index)
        
        if start_index <= 0 and end_index >= total_frames - 1:
            break
    
    used_frame_indices = used_frame_indices[:grid_size ** 2]
    
    # Find index of average frame
    if avg_index in used_frame_indices:
        index = used_frame_indices.index(avg_index)
    else:
        index = 0
    
    return used_frame_indices, index


def select_and_filter_keyframes_with_anchor(
    selected_indices: List[int],
    total_indices: List[int],
    grid_size: int,
    search_anchor: str,
    total_frames: int
) -> List[int]:
    """
    Filter keyframes by anchor position (start/end of video).
    
    Args:
        selected_indices: Selected frame indices
        total_indices: All available frame indices
        grid_size: Minimum number of frames needed
        search_anchor: 'start' or 'end'
        total_frames: Total number of frames
    
    Returns:
        Filtered frame indices
    """
    if search_anchor == "start":
        filtered_indices = [idx for idx in selected_indices if idx < total_frames // 2]
        if len(filtered_indices) < grid_size:
            remaining = [i for i in total_indices if i not in filtered_indices and i < total_frames // 2]
            filtered_indices.extend(remaining[:grid_size - len(filtered_indices)])
    elif search_anchor == "end":
        filtered_indices = [idx for idx in selected_indices if idx >= total_frames // 2]
        if len(filtered_indices) < grid_size:
            remaining = [i for i in total_indices if i not in filtered_indices and i >= total_frames // 2]
            filtered_indices.extend(remaining[:grid_size - len(filtered_indices)])
    else:
        raise ValueError("search_anchor must be 'start' or 'end'")
    
    return sorted(filtered_indices)


def get_contact_separation_pairs(
    results: List[Tuple[str, int]],
    speed_data: List[Tuple[int, float]],
    video_type: str = "short"
) -> List[Tuple[int, int]]:
    """
    Pair Contact and Separation events from VLM results.
    
    Args:
        results: List of (event_type, frame) tuples from VLM
        speed_data: List of (frame, speed) tuples
        video_type: 'short' (single action) or 'long' (multiple actions)
    
    Returns:
        List of (contact_frame, separation_frame) tuples
    """
    speed_dict = {frame: speed for frame, speed in speed_data}
    
    # Separate Contact and Separation events
    contacts = []
    separations = []
    
    for event_type, frame in results:
        if frame not in speed_dict:
            continue
        if event_type == "Contact":
            contacts.append((frame, speed_dict[frame]))
        elif event_type == "Separation":
            separations.append((frame, speed_dict[frame]))
    
    contacts.sort()
    separations.sort()
    
    pairs = []
    contact_candidates = []
    
    # Sort all events by frame
    all_events = sorted(
        [("C", frame, speed) for frame, speed in contacts] +
        [("S", frame, speed) for frame, speed in separations],
        key=lambda x: x[1]
    )
    
    for event in all_events:
        event_type, frame, speed = event
        
        if event_type == "C":
            # Keep contact candidates with decreasing speed
            if not contact_candidates or speed < contact_candidates[-1][1]:
                contact_candidates.append((frame, speed))
        elif event_type == "S":
            if contact_candidates:
                # Pair with best (lowest speed) contact
                best_contact = min(contact_candidates, key=lambda x: x[1])
                pairs.append((best_contact[0], frame))
                contact_candidates = []
    
    # For short videos, keep only the pair with maximum distance
    if video_type == "short" and len(pairs) > 1:
        pair_distances = [(c, s, s - c) for c, s in pairs]
        best_pair = max(pair_distances, key=lambda x: x[2])
        return [(best_pair[0], best_pair[1])]
    
    return pairs

