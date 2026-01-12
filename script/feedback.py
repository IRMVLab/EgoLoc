"""
Feedback mechanism for frame refinement with in-context learning
"""
import numpy as np
from typing import Dict, Tuple

from .vlm_inference import (
    scene_understanding,
    PROMPT_FEEDBACK_CONTACT,
    PROMPT_FEEDBACK_SEPARATION,
    PROMPT_CONTACT_WITH_NEGATIVE,
    PROMPT_SEPARATION_WITH_NEGATIVE
)
from .frame_selection import select_frames_near_average
from .utils import create_frame_grid_with_keyframe


def refine_frame_with_feedback(
    credentials: Dict,
    video_path: str,
    initial_frame: int,
    state: str,
    all_frames: np.ndarray,
    all_speeds: np.ndarray,
    total_frames: int,
    grid_size: int,
    max_feedback: int
) -> Tuple[bool, int]:
    """
    Refine frame selection using feedback loop with in-context learning.
    
    This function implements a feedback mechanism that:
    1. Verifies if the initial frame is correct
    2. If incorrect, uses the frame as a negative example
    3. Samples nearby candidate frames based on speed
    4. Regenerates grid image with new candidates
    5. Uses in-context learning (negative example + new grid) to refine selection
    
    Args:
        credentials: API credentials for VLM
        video_path: Path to video file
        initial_frame: Initial frame index to verify
        state: Event state ("Contact" or "Separation")
        all_frames: Array of all frame indices with speed data
        all_speeds: Array of speeds corresponding to all_frames
        total_frames: Total number of frames in video
        grid_size: Size of frame grid
        max_feedback: Maximum number of feedback iterations
    
    Returns:
        Tuple of (success: bool, refined_frame: int)
        - success: True if feedback succeeded or frame is correct
        - refined_frame: Final refined frame index
    """
    feedback_prompt = (
        PROMPT_FEEDBACK_CONTACT if state == "Contact"
        else PROMPT_FEEDBACK_SEPARATION
    )
    
    tried_frames = set()
    final_frame = initial_frame
    
    for fb_count in range(max_feedback):
        tried_frames.add(final_frame)
        single_image = create_frame_grid_with_keyframe(video_path, [final_frame], 1)
        
        # First verification: check if current frame is correct
        feedback_result = scene_understanding(
            credentials, single_image, feedback_prompt, principle="feedback"
        )
        
        print(f"[Feedback] Iteration {fb_count + 1}/{max_feedback}: VLM output = {feedback_result}")
        
        if feedback_result and feedback_result.strip() == "1":
            print(f"[Feedback] ✅ Iteration {fb_count + 1} succeeded, stopping feedback loop")
            return True, final_frame
        
        # Negative feedback: save negative example for in-context learning
        negative_example = single_image
        
        # Find candidate frames nearby
        feedback_window = 3
        candidates = [
            i for i in range(final_frame - feedback_window, final_frame + feedback_window + 1)
            if 0 <= i < total_frames and i not in tried_frames
        ]
        
        if not candidates:
            print("[Feedback] ⚠️ No more candidate frames, breaking feedback loop")
            return False, final_frame
        
        # Weighted sampling by speed
        candidate_speeds = [
            all_speeds[np.where(all_frames == i)[0][0]]
            if np.any(all_frames == i) else 9999
            for i in candidates
        ]
        inv_speeds = 1 / (np.array(candidate_speeds) + 1e-8)
        probabilities = inv_speeds / inv_speeds.sum()
        new_candidate_frame = int(np.random.choice(candidates, p=probabilities))
        
        # Regenerate grid image based on new candidate frame
        new_frame_indices, _ = select_frames_near_average(
            [new_candidate_frame], grid_size, total_frames, []
        )
        new_grid_image = create_frame_grid_with_keyframe(video_path, new_frame_indices, grid_size)
        
        # Use in-context learning: provide negative example and new grid to VLM
        if state == "Contact":
            in_context_prompt = PROMPT_CONTACT_WITH_NEGATIVE
        else:
            in_context_prompt = PROMPT_SEPARATION_WITH_NEGATIVE
        
        # Call VLM with negative example for in-context learning
        new_description = scene_understanding(
            credentials, new_grid_image, in_context_prompt,
            principle=None, negative_example=negative_example
        )
        
        print(f"[Feedback] In-context learning VLM output: {new_description}")
        
        if new_description and new_description != -1:
            # Update final_frame to VLM's new selection
            index_specified = max(min(int(new_description) - 1, len(new_frame_indices) - 1), 0)
            final_frame = new_frame_indices[index_specified]
            # Continue loop to verify this new final_frame if we have more attempts
            if fb_count < max_feedback - 1:
                continue
            else:
                # No more attempts, use this as final result
                print(f"[Feedback] ✅ Using in-context learning result: frame {final_frame} (from grid index {index_specified + 1})")
                return True, final_frame
        else:
            # If VLM returns -1 or invalid, use speed-sampled frame
            final_frame = new_candidate_frame
            # Continue loop to verify if we have more attempts
            if fb_count < max_feedback - 1:
                continue
            else:
                # No more attempts, use this as final result
                print(f"[Feedback] ✅ Using speed-sampled frame: {final_frame} (VLM returned invalid)")
                return True, final_frame
    
    # If we exhausted all feedback attempts without success
    return False, final_frame

