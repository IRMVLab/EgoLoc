"""
Utility functions for video processing and visualization
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional


def image_resize(image: np.ndarray, width: Optional[int] = None, height: Optional[int] = None, 
                 inter: int = cv2.INTER_AREA) -> np.ndarray:
    """Resize image while preserving aspect ratio"""
    if width is None and height is None:
        return image
    (h, w) = image.shape[:2]
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    return cv2.resize(image, dim, interpolation=inter)


def create_frame_grid_with_keyframe(
    video_path: str,
    frame_indices: List[int],
    grid_size: int
) -> np.ndarray:
    """Create a numbered frame grid image"""
    spacer = 0
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frm = cap.read()
        if not ok:
            frm = (
                np.zeros_like(frames[0])
                if frames
                else np.zeros((200, 200, 3), np.uint8)
            )
        frm = image_resize(frm, width=200)
        frames.append(frm)
    cap.release()
    
    while len(frames) < grid_size**2:
        frames.append(np.zeros_like(frames[0]))
    
    fh, fw = frames[0].shape[:2]
    grid_h = grid_size * fh + (grid_size - 1) * spacer
    grid_w = grid_size * fw + (grid_size - 1) * spacer
    grid = np.ones((grid_h, grid_w, 3), np.uint8) * 255
    
    for i in range(grid_size):
        for j in range(grid_size):
            k = i * grid_size + j
            frm = frames[k]
            max_d = int(min(frm.shape[:2]) * 0.5)
            cc = (frm.shape[1] - max_d // 2, max_d // 2)
            overlay = frm.copy()
            cv2.circle(overlay, cc, max_d // 2, (255, 255, 255), -1)
            frm = cv2.addWeighted(overlay, 0.3, frm, 0.7, 0)
            cv2.circle(frm, cc, max_d // 2, (255, 255, 255), 2)
            fs = max_d / 50
            txtsz = cv2.getTextSize(str(k + 1), cv2.FONT_HERSHEY_SIMPLEX, fs, 2)[0]
            tx = frm.shape[1] - txtsz[0] // 2 - max_d // 2
            ty = txtsz[1] // 2 + max_d // 2
            cv2.putText(
                frm, str(k + 1), (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 2
            )
            y1, y2 = i * (fh + spacer), (i + 1) * fh + i * spacer
            x1, x2 = j * (fw + spacer), (j + 1) * fw + j * spacer
            grid[y1:y2, x1:x2] = frm
    
    return grid


def create_frame_grid_state(
    video_path: str,
    frame_indices: List[int],
    grid_size: tuple = (1, 2)
) -> np.ndarray:
    """Create a state comparison grid (2 frames side by side)"""
    spacer = 0
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            frame = np.zeros((400, 400, 3), dtype=np.uint8)
        frame = image_resize(frame, width=400)
        frames.append(frame)
    cap.release()
    
    total_needed = grid_size[0] * grid_size[1]
    while len(frames) < total_needed:
        frames.append(np.zeros_like(frames[0]))
    
    fh, fw = frames[0].shape[:2]
    gh = grid_size[0] * fh + (grid_size[0] - 1) * spacer
    gw = grid_size[1] * fw + (grid_size[1] - 1) * spacer
    grid_img = np.ones((gh, gw, 3), dtype=np.uint8) * 255
    
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            idx = i * grid_size[1] + j
            frame = frames[idx]
            y1 = i * (fh + spacer)
            y2 = y1 + fh
            x1 = j * (fw + spacer)
            x2 = x1 + fw
            grid_img[y1:y2, x1:x2] = frame
    
    return grid_img


def visualize_frame(video_path: str, idx: int, out_path: str, label: Optional[str] = None):
    """Visualize a single frame with optional label"""
    if idx < 0:
        return
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print(f"[Utils] Could not seek to frame {idx} in {video_path}")
        return
    if label:
        cv2.putText(frame, label, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
    cv2.imwrite(out_path, frame)
    print(f"[Utils] Visualized frame {idx} to {out_path}")

