"""
EgoLoc-Plus 3‑D Demo – *bimanual* edition
Assumed repo layout (relative to this file):
EgoLoc/
│
├── ./                    ← this script lives here (anywhere inside EgoLoc)
├── HaMeR/                ← git clone https://github.com/geopavlakos/hamer.git
└── Video-Depth-Anything/ ← git clone https://github.com/DepthAnything/Video-Depth-Anything.git
"""

import argparse
import base64
import json
import math
import os
import subprocess
import time
import warnings
from pathlib import Path
import open3d as o3d 

import cv2
import dotenv  # read .env creds for GPT-4o
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage  # connected‑component helper
from scipy.ndimage import gaussian_filter1d
from typing import List, Dict, Tuple, Optional, Any

try:
    import torch
except ImportError:  # Allow import on machines without torch
    torch = None

# External repos that *must* exist inside EgoLoc root
try:
    import hamer  # type: ignore
except ImportError as e:
    raise ImportError(
        "HaMeR repo not found.  Make sure you cloned it inside the EgoLoc root."
    ) from e

try:
    from hamer.vitpose_model import ViTPoseModel  # type: ignore
except ImportError as e:
    raise ImportError(
        "vitpose_model.py not found.  It ships with HaMeR; "
        "confirm your PYTHONPATH contains that folder."
    ) from e

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
plt.switch_backend("Agg")  # headless plotting

# Helper to unpack Video-Depth-Anything *_depths.npz to per-frame .npy
def _unpack_depth_npz(depth_dir: Path) -> None:
    """Convert VDA’s *_depths.npz to individual .npy, skipping ones that exist."""
    npz_files = list(depth_dir.glob("*_depths.npz"))
    if not npz_files:
        return  # nothing to unpack

    arr = np.load(npz_files[0])["depths"]  # (N, H, W)
    created = 0
    for i, depth in enumerate(arr):
        out_f = depth_dir / f"pred_depth_{i:06d}.npy"
        if out_f.exists():          # keep the good tensor we already have
            continue
        np.save(out_f, depth.astype(np.float32))
        created += 1
    if created:
        print(f"[VDA] Unpacked {created} new tensors")



# Paths & repo‑root helpers
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = next(
    p
    for p in [SCRIPT_DIR] + list(SCRIPT_DIR.parents)
    if (p / "Video-Depth-Anything").exists()
)

VDA_DIR = REPO_ROOT / "Video-Depth-Anything"

DEPTH_SCALE_M = 3.0  # pixel value 255 corresponds to 3m (linear scaling)
MAX_FEEDBACKS = 1

# Depth-quality helpers
def _is_invalid_inv(inv: np.ndarray) -> bool:
    """
    Return True when the inverse-depth tensor is all-NaN/Inf or nearly flat.
    """
    return (not np.isfinite(inv).any()) or np.nanstd(inv) < 1e-4

def count_bad_depth_tensors(depth_dir: Path) -> Tuple[int, int]:
    """
    Scan *.npy files in *depth_dir* and count how many are unusable.
    """
    bad, total = 0, 0
    for f in depth_dir.glob("pred_depth_*.npy"):
        total += 1
        inv = np.load(f, mmap_mode="r")  # memory-mapped to avoid loading everything
        if _is_invalid_inv(inv):
            bad += 1
    return bad, total

# Basic helpers
def get_json_path(video_name: str, base_dir: str):
    return os.path.join(base_dir, f"{video_name}_speed.json")

def get_json_path_for_hand(video_name: str, base_dir: str, hand: str) -> str:
    """Return speed JSON path for a specific hand or combined.

    hand ∈ {"right", "left", "either"}; "either" maps to the combined file.
    """
    hand = (hand or "either").lower()
    if hand == "either":
        return os.path.join(base_dir, f"{video_name}_speed.json")
    if hand not in {"right", "left"}:
        raise ValueError("hand must be one of: right, left, either")
    return os.path.join(base_dir, f"{video_name}_speed_{hand}.json")

def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    """Resize *image* while preserving aspect ratio."""
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

def image_resize_for_vlm(frame, inter=cv2.INTER_AREA):
    """Resize an image so the short side ≤ 768 px and long side ≤ 2000 px."""
    h, w = frame.shape[:2]
    aspect = w / h
    max_short, max_long = 768, 2000
    if aspect > 1:
        new_w = min(w, max_long)
        new_h = int(new_w / aspect)
        if new_h > max_short:
            new_h = max_short
            new_w = int(new_h * aspect)
    else:
        new_h = min(h, max_long)
        new_w = int(new_h * aspect)
        if new_w > max_short:
            new_w = max_short
            new_h = int(new_w / aspect)
    return cv2.resize(frame, (new_w, new_h), interpolation=inter)

def extract_json_part(text_output: str) -> Optional[str]:
    """Extract the JSON fragment {"points":[...]} from GPT text output."""
    text = text_output.strip().replace(" ", "").replace("\n", "")
    try:
        idx = text.index('{"points":')
        frag = text[idx:]
        end = frag.index("}") + 1
        return frag[:end]
    except ValueError:
        print("Text received:", text_output)
        return None

def scene_understanding(credentials: Dict[str, Any], frame: np.ndarray, prompt: str, *, flag: Optional[str] = None, raw: bool = False):
    """
    Vision-language helper.

    ── Return modes ───────────────────────────────────────────────
    • default ( flag is None  and  raw is False ):
        → (first_point, full_text)
    • feedback / “state” checks ( flag is not None  or  raw is True ):
        → full_text  (str)
    """

    frame = image_resize_for_vlm(frame)
    _, buf = cv2.imencode(".jpg", frame)
    b64 = base64.b64encode(buf).decode("utf-8")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{b64}",
                        "detail": "high",
                    },
                },
            ],
        }
    ]

    import openai

    if credentials.get("AZURE_OPENAI_API_KEY"):
        from openai import AzureOpenAI

        client = AzureOpenAI(
            api_version="2024-02-01",
            azure_endpoint=credentials["AZURE_OPENAI_ENDPOINT"],
            api_key=credentials["AZURE_OPENAI_API_KEY"],
        )
        params = {"model": credentials["AZURE_OPENAI_DEPLOYMENT_NAME"]}
    else:
        from openai import OpenAI

        client = OpenAI(
            api_key=credentials["OPENAI_API_KEY"],
            base_url="https://api.chatanywhere.tech/v1",
        )
        params = {"model": "gpt-4o"}

    params.update(dict(messages=messages, max_tokens=200, temperature=0.1, top_p=0.5))

    retries = 0
    while retries < 5:
        try:
            result = client.chat.completions.create(**params)
            break
        except (openai.RateLimitError, openai.APIStatusError):
            time.sleep(2)
            retries += 1
    else:
        raise RuntimeError("Failed to obtain GPT-4o response.")

    content = result.choices[0].message.content

    if raw or flag is not None:
        return content

    frag = extract_json_part(content)
    if frag is None:
        return -1, content
    try:
        pts = json.loads(frag)["points"]
        return (pts[0] if pts else -1), content
    except Exception:  # malformed JSON or no "points"
        return -1, content

def create_frame_grid_with_keyframe(video_path: str, frame_indices: List[int], grid_size: int) -> np.ndarray:
    """Return a numbered frame grid (BGR uint8)."""
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


def select_top_n_frames_from_json(json_path: str, n: int, frame_index: Optional[int] = None, flag: Optional[str] = None, receive_flag: Optional[str] = None):
    """Return indices with lowest speed; behaviour altered by *flag*."""
    with open(json_path, "r") as f:
        data = json.load(f)
    items = list(data.items())
    if frame_index is None:
        valid = [(int(i), s) for i, s in items if s != 0 and not math.isnan(s)]
    else:
        if flag == "feedback":
            valid = [
                (int(i), s)
                for i, s in items
                if s != 0 and not math.isnan(s) and int(i) > frame_index
            ]
            inval = [
                int(i)
                for i, s in items
                if s != 0 and not math.isnan(s) and int(i) <= frame_index
            ]
        elif flag == "speed":
            valid = [
                (int(i), s)
                for i, s in items
                if s != 0 and not math.isnan(s) and s < frame_index
            ]
            inval = [
                int(i)
                for i, s in items
                if s != 0 and not math.isnan(s) and s >= frame_index
            ]
        else:
            valid = [
                (int(i), s)
                for i, s in items
                if s != 0 and not math.isnan(s) and int(i) != frame_index
            ]
            inval = [
                int(i)
                for i, s in items
                if s != 0 and not math.isnan(s) and int(i) == frame_index
            ]
    top = [idx for idx, _ in sorted(valid, key=lambda x: x[1])[:n]]
    # Fallback: if no valid frames found, use evenly spaced frames
    if not top:
        print(f"[WARNING] No valid speed data found in {json_path}, using fallback frames")
        total_frames = len(items) if items else n
        if total_frames > 0:
            # Generate evenly spaced frame indices
            step = max(1, total_frames // n)
            top = list(range(0, min(total_frames, n * step), step))[:n]
        else:
            top = list(range(n))  # Last resort fallback
    
    if receive_flag is None:
        return top
    
    # Handle inval for receive_flag case
    if 'inval' not in locals():
        inval = []
    return inval, top


def select_frames_near_average(indices: List[int], grid_size: int, total_frames: int, invalid: List[int], min_index: Optional[int] = None):
    """Return *grid_size²* unique frames centred on average of *indices*."""

    if not indices or len(indices) == 0:
        avg = total_frames // 2
    else:
        mean_val = np.mean(indices)
        if np.isnan(mean_val) or np.isinf(mean_val):
            # Use middle of video as fallback
            avg = total_frames // 2
        else:
            avg = round(mean_val)
    
    # Ensure avg is within valid range
    avg = max(0, min(avg, total_frames - 1))
    used = []
    if avg not in invalid and (min_index is None or avg > min_index):
        used.append(avg)
    left, right = avg - 1, avg + 1
    while len(used) < grid_size**2:
        if left >= 0:
            if left not in invalid and (min_index is None or left > min_index):
                used.insert(0, left)
            left -= 1
        if len(used) >= grid_size**2:
            break
        if right < total_frames:
            if right not in invalid and (min_index is None or right > min_index):
                used.append(right)
            right += 1
    return used[: grid_size**2]


def select_and_filter_keyframes_with_anchor(sel: List[int], total_idx: List[int], grid_size: int, anchor: str, video_path: str):
    """Keep frames in first/second half depending on *anchor* ('start'/'end')."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if anchor == "start":
        filtered = [i for i in sel if i < total_frames // 2]
        if len(filtered) < grid_size:
            extra = [
                i for i in total_idx if i not in filtered and i < total_frames // 2
            ]
            filtered.extend(extra[: grid_size - len(filtered)])
    elif anchor == "end":
        filtered = [i for i in sel if i >= total_frames // 2]
        if len(filtered) < grid_size:
            extra = [
                i for i in total_idx if i not in filtered and i >= total_frames // 2
            ]
            filtered.extend(extra[: grid_size - len(filtered)])
    else:
        raise ValueError("anchor must be 'start' or 'end'")
    return sorted(filtered)


def determine_by_state(credentials, video_path, action, grid_size, total_frames, frame_index, anchor, speed_folder):
    """Ask GPT-4o if *frame_index* is valid contact/separation."""
    prompt = (
        (
            "I will show an image of hand-object interaction. "
            "You need to determine whether the hand and the object are in obvious contact. "
        )
        if anchor == "start"
        else (
            "I will show an image of hand-object interaction. "
            "You need to determine whether the hand and the object are clearly separate. "
        )
    ) + "Return only a single character: 1 for yes, 0 for no."

    grid = create_frame_grid_with_keyframe(video_path, [frame_index], 1)
    res = scene_understanding(credentials, grid, prompt, raw=True)
    s = str(res).strip()
    if (s[:1] == "1") or frame_index > total_frames - 5:
        return frame_index
    return process_task(
        credentials,
        video_path,
        action,
        grid_size,
        total_frames,
        anchor,
        speed_folder,
        frame_index,
        flag="feedback",
    )


def determine_by_speed(credentials, video_path, action, grid_size, total_frames, frame_index, anchor, speed_folder, *, hand: str = "either"):
    """Reject frames whose speed exceeds 30-percentile threshold."""
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    json_path = get_json_path_for_hand(video_name, speed_folder, hand)
    with open(json_path, "r") as f:
        data = json.load(f)
    valid = [(int(i), s) for i, s in data.items() if s != 0 and not math.isnan(s)]
    if not valid:
        return process_task(
            credentials,
            video_path,
            action,
            grid_size,
            total_frames,
            anchor,
            speed_folder,
            frame_index,
            flag="speed",
        )
    valid.sort(key=lambda x: x[1])
    threshold = valid[int(0.3 * len(valid))][1]
    cur_speed = data.get(str(frame_index), 0)
    if cur_speed <= threshold:
        return frame_index
    return process_task(
        credentials,
        video_path,
        action,
        grid_size,
        total_frames,
        anchor,
        speed_folder,
        frame_index,
        flag="speed",
        hand=hand,
    )


def feedback_contact(credentials, video_path, action, grid_size, total_frames, frame_start, max_fb, anchor, speed_folder, *, hand: str = "either"):
    cnt = 0
    cur = frame_start
    while cnt < max_fb:
        new = determine_by_state(
            credentials,
            video_path,
            action,
            grid_size,
            total_frames,
            cur,
            anchor,
            speed_folder,
        )
        if new is None:
            return None
        if new == cur:
            new_sp = determine_by_speed(
                credentials,
                video_path,
                action,
                grid_size,
                total_frames,
                cur,
                anchor,
                speed_folder,
                hand=hand,
            )
            if new_sp == cur:
                return cur
            cur = new_sp
        else:
            cur = new
        cnt += 1
    return cur


def feedback_separation(credentials, video_path, action, grid_size, total_frames, frame_end, max_fb, anchor, speed_folder, *, hand: str = "either"):
    return feedback_contact(
        credentials,
        video_path,
        action,
        grid_size,
        total_frames,
        frame_end,
        max_fb,
        anchor,
        speed_folder,
        hand=hand,
    )


def process_task(credentials, video_path, action, grid_size, total_frames, anchor, speed_folder, frame_index=None, flag=None, *, hand: str = "either"):
    """Core routine that builds frame grid, asks GPT-4o for best frame."""
    prompt_start = (
        f"I will show an image grid of numbered frames showing a hand-object action. "
        f"Pick the number closest to when the action ({action}) STARTS. "
        f"If the action is absent, choose -1. "
        'Return JSON strictly as {"points": [<number>]} with no extra text.'
    )
    prompt_end = (
        f"I will show an image grid of numbered frames showing a hand-object action. "
        f"Pick the number closest to when the action ({action}) ENDS. "
        f"If the action has not ended, choose -1. "
        'Return JSON strictly as {"points": [<number>]} with no extra text.'
    )
    prompt = prompt_start if anchor == "start" else prompt_end

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    json_path = get_json_path_for_hand(video_name, speed_folder, hand)

    if frame_index is None:
        top = select_top_n_frames_from_json(json_path, 4)
        total_idx = select_top_n_frames_from_json(json_path, total_frames)
        filt = select_and_filter_keyframes_with_anchor(
            top, total_idx, 4, anchor, video_path
        ) or sorted(top)
        used = select_frames_near_average(filt, grid_size, total_frames, [])
    else:
        if flag == "feedback":
            invalid, top = select_top_n_frames_from_json(
                json_path, 4, frame_index, flag, receive_flag="right"
            )
            total_idx = select_top_n_frames_from_json(
                json_path, total_frames, frame_index, flag
            )
            filt = select_and_filter_keyframes_with_anchor(
                top, total_idx, 4, anchor, video_path
            ) or sorted(top)
            used = select_frames_near_average(
                filt, grid_size, total_frames, invalid, min_index=frame_index
            )
        elif flag == "speed":
            invalid, top = select_top_n_frames_from_json(
                json_path, 4, frame_index, flag, receive_flag="right"
            )
            total_idx = select_top_n_frames_from_json(
                json_path, total_frames, frame_index, flag
            )
            filt = select_and_filter_keyframes_with_anchor(
                top, total_idx, 4, anchor, video_path
            ) or sorted(top)
            used = select_frames_near_average(filt, grid_size, total_frames, invalid)
        else:
            invalid, top = select_top_n_frames_from_json(
                json_path, 4, frame_index, receive_flag="right"
            )
            total_idx = select_top_n_frames_from_json(
                json_path, total_frames, frame_index
            )
            filt = select_and_filter_keyframes_with_anchor(
                top, total_idx, 4, anchor, video_path
            ) or sorted(top)
            used = select_frames_near_average(filt, grid_size, total_frames, invalid)

    grid = create_frame_grid_with_keyframe(video_path, used, grid_size)
    result = scene_understanding(credentials, grid, prompt)
    if isinstance(result, tuple):
        choice, _ = result
    else:
        choice = result
    if choice == -1:
        return None
    idx = max(0, min(int(choice) - 1, len(used) - 1))
    return used[idx]


# Depth video generation via Video‑Depth‑Anything
def generate_depth_video_vda(video_path: str, depth_out_path: str, *, device: str = "cuda", encoder: str = "vits") -> Path:
    """Run Video‑Depth‑Anything once and save the raw depth video."""
    video_path = Path(video_path).resolve()
    depth_out_path = Path(depth_out_path).resolve()
    depth_out_path.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        "run.py",
        "--input_video",
        str(video_path),
        "--output_dir",
        str(depth_out_path),
        "--encoder",
        encoder,  # or "vitl" if you want the large model
        "--save_npz",  # save per‑frame tensors
    ]
    if device == "cpu":
        cmd.append("--fp32")  # avoids half‑precision on CPU

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{VDA_DIR}:{env.get('PYTHONPATH', '')}"

    print("[VDA] Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=VDA_DIR, env=env)

    # Unpack & sanity‑check
    _unpack_depth_npz(depth_out_path)
    if not any(depth_out_path.glob("pred_depth_*.npy")):
        raise RuntimeError(
            "[VDA] No pred_depth_*.npy tensors were produced – check the VDA output above."
        )

    return depth_out_path

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

# Debug logging helpers removed
# _log_zero_speed no longer needed – provide empty stub to avoid NameError if
# residual references remain.
def _log_zero_speed(*args, **kwargs):
    pass

# Depth loading helper
def _load_depth(depth_dir: Path, idx: int) -> Optional[np.ndarray]:
    """
    VDA stores **inverse depth** (bigger = nearer).  
    Convert to metric depth in metres and keep a useful range.
    """
    f = depth_dir / f"pred_depth_{idx:06d}.npy"
    if not f.exists():
        return None
    inv = np.load(f).astype(np.float32)          # (H, W)
    if _is_invalid_inv(inv):                     # ← early-reject unusable tensor
        return None
    depth = DEPTH_SCALE_M / (inv + 1e-6)  # invert once, not twice
    return depth

# Simple camera projection helper
def _pixel_to_camera(u: float, v: float, z: float, W: int, H: int):
    fx = fy = max(W, H)
    cx, cy = W / 2.0, H / 2.0
    X = (u - cx) * z / fx
    Y = (v - cy) * z / fy
    return X, Y, z

# HaMeR / ViTPose – created once, reused
_HAMER_CACHE: Dict[str, ViTPoseModel] = {}


def _get_vitpose_model(device: str = "cuda") -> ViTPoseModel:
    """Return a cached ViTPoseModel (no Detectron2 dependency)."""
    if "cpm" in _HAMER_CACHE:
        return _HAMER_CACHE["cpm"]

    import hamer.vitpose_model as _vpm  # local import avoids side‑effects if unused

    _vpm.ROOT_DIR = "./hamer"
    _vpm.VIT_DIR =  "./hamer/third-party/ViTPose"

    cfg_rel = Path(
        "configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/"
        "ViTPose_huge_wholebody_256x192.py"
    )
    ckpt_rel = Path("_DATA/vitpose_ckpts/vitpose+_huge/wholebody.pth")

    for _name, _dic in _vpm.ViTPoseModel.MODEL_DICT.items():
        _dic["config"] = str(Path(_vpm.VIT_DIR) / cfg_rel)
        _dic["model"] = str(Path(_vpm.ROOT_DIR) / ckpt_rel)

    _HAMER_CACHE["cpm"] = ViTPoseModel(device)
    return _HAMER_CACHE["cpm"]


# Wrist detection helper (depth‑guided + ViTPose)
def _wrist_from_frame(
    frame_bgr: np.ndarray,
    gray_depth: np.ndarray,
    cpm: ViTPoseModel,
) -> Dict[str, Optional[Tuple[float, float]]]:
    """
    Detect both wrists (left & right).  
    Returns: {"right": (u, v) | None, "left": (u, v) | None}
    """
    # 1) Depth-guided candidate ROIs – expand to the closest *15 %* of depth
    #    pixels (empirically more tolerant when the hand is not the single
    #    nearest object). If this still fails we will fall back to a global
    #    ViTPose pass later.
    nearest_pct = 15.0  # tweak here if needed (10 to 15 helps many cases)
    nearest = gray_depth < np.percentile(gray_depth, nearest_pct)
    labels, n_lbl = ndimage.label(nearest)
    if n_lbl == 0:
        # no obvious near-depth blobs – let global fallback handle it
        n_lbl = 0

    sizes = ndimage.sum(nearest, labels, range(1, n_lbl + 1))
    blobs = 1 + np.argsort(sizes)[-2:]          # label IDs

    wrists = {"right": None, "left": None}
    for lbl in blobs:
        mask = labels == lbl
        ys, xs = np.where(mask)
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        roi_bgr = frame_bgr[y0 : y1 + 1, x0 : x1 + 1]

        bbox = np.array([[0, 0, roi_bgr.shape[1] - 1, roi_bgr.shape[0] - 1, 1.0]],
                         dtype=np.float32)
        pose = cpm.predict_pose(roi_bgr[:, :, ::-1], [bbox])[0]
        kpts = pose["keypoints"]
        kpts_l, kpts_r = kpts[-42:-21], kpts[-21:]

        for tag, hk in (("left", kpts_l), ("right", kpts_r)):
            # lowered confidence threshold to 0.25 (more tolerant)
            valid = hk[:, 2] > 0.25
            if valid.sum() <= 3:
                continue
            u = x0 + hk[0, 0]
            v = y0 + hk[0, 1]
            # if two candidates, keep the nearer one
            if wrists[tag] is None or gray_depth[int(v), int(u)] < gray_depth[int(wrists[tag][1]), int(wrists[tag][0])]:
                wrists[tag] = (float(u), float(v))

    # 2) Fallback: if either wrist is still missing, run ViTPose on the
    #    full frame. This is slower but guarantees at least one attempt.
    if wrists["left"] is None or wrists["right"] is None:
        full_bbox = np.array([[0, 0, frame_bgr.shape[1] - 1, frame_bgr.shape[0] - 1, 1.0]],
                             dtype=np.float32)
        pose_full = cpm.predict_pose(frame_bgr[:, :, ::-1], [full_bbox])[0]
        kpts_full = pose_full["keypoints"]
        kpts_l_full, kpts_r_full = kpts_full[-42:-21], kpts_full[-21:]

        for tag, hk in (("left", kpts_l_full), ("right", kpts_r_full)):
            if wrists[tag] is not None:
                continue  # already found via depth-guided ROI
            valid = hk[:, 2] > 0.25
            if valid.sum() <= 3:
                continue
            u, v = hk[0, 0], hk[0, 1]
            wrists[tag] = (float(u), float(v))
    return wrists

# Hand Position Registration With ICP and Frame 0 Alignment Helper
def register_hand_positions(pcd_root, hand3d_root, save_reg_root, threshold=0.03):
    """
    Align per-frame 3-D hand positions to the first frame using point-to-point
    ICP.

    Args:
        pcd_root (str): Directory containing colored point clouds organised as
            `<video>/frame.ply.
        hand3d_root (str): Directory with camera-coordinate 3-D hand JSON files.
        save_reg_root (str): Output directory where globally registered hand
            trajectories will be saved.
        threshold (float, optional): ICP correspondence distance threshold in
            metres. Defaults to 0.03.

    Returns:
        None
    """
    os.makedirs(save_reg_root, exist_ok=True)
    for video in sorted(os.listdir(pcd_root)):
        pcd_dir = os.path.join(pcd_root, video)
        hand3d_path = os.path.join(hand3d_root, f"{video}.json")
        if not os.path.isdir(pcd_dir) or not os.path.isfile(hand3d_path):
            continue
        hand3d = json.load(open(hand3d_path))
        plys = sorted([f for f in os.listdir(pcd_dir) if f.endswith('.ply')],
                      key=lambda x: int(os.path.splitext(x)[0]))

        first_pcd = None  # reference cloud (frame 1)
        odoms = []  # unused - kept from orig code
        reg_hand_dict = {}  # output dict: frameID to (x,y,z)
        prev_pcd = None  # helps store first_pcd

        for i, ply in enumerate(plys):
            frame_id = str(i + 1)  # JSON is 1-based indexing
            pcd = o3d.io.read_point_cloud(os.path.join(pcd_dir, ply))
            pts = np.asarray(pcd.points)
            pcd.points = o3d.utility.Vector3dVector(pts)

            if i == 0:  # first frame: no registration
                odoms.append(np.eye(4))  # placeholder (unused)
                if frame_id in hand3d:
                    h0 = np.array(hand3d[frame_id])  # camera-frame wrist point
                    reg_hand_dict[frame_id] = h0.tolist()  # frame 1 becomes origin
            else:
                if first_pcd is None:  # cache reference once
                    first_pcd = prev_pcd

                reg = o3d.pipelines.registration.registration_icp(
                    pcd,  # source cloud
                    first_pcd,  # target cloud
                    threshold,  # correspondence distance (m)
                    np.eye(4),  # initial guess = identity
                    o3d.pipelines.registration.TransformationEstimationPointToPoint()
                )
                T = reg.transformation  # 4x4 rigid transform

                h = np.array(hand3d.get(str(i + 1), [np.nan, np.nan, np.nan]))
                h4 = np.append(h, 1)  # homogeneous coordinate
                h_reg = (T @ h4)[:3].tolist()  # apply ICP transform
                if frame_id in hand3d:
                    reg_hand_dict[frame_id] = h_reg

            prev_pcd = pcd  # keep for first_pcd assignment

        # save globally registered trajectory for this video
        with open(os.path.join(save_reg_root, f"{video}.json"), 'w') as f:
            json.dump(reg_hand_dict, f, indent=4)
        print(f"Computed globally registered 3-D hand trajectory for {video}")

# Robust percentile-flavoured detector
def contact_separation_from_speed(
    speed_dict: Dict[int, float],
    *,
    active_pct: float = 30.0,     # % of frames considered “active”
    sigma: int = 5
) -> Tuple[int, int]:
    """
    Return first contact & last separation indices in a *very flat* speed curve.

    Strategy:
    1) Smooth with Gaussian σ = *sigma* frames.
    2) Treat the top *active_pct* percent of speeds as “motion”.
    3) The first/last frames inside that mask are contact/separation.

    Works even when absolute values are ≪ 1e-4.
    """
    speeds = gaussian_filter1d(np.array(list(speed_dict.values())), sigma)
    if np.allclose(speeds, 0):
        return -1, -1                       # nothing but zeros

    # define activity mask by percentile
    thresh = np.percentile(speeds, 100 - active_pct)
    active = speeds >= thresh
    if not active.any():
        return -1, -1

    contact = int(np.argmax(active))
    separation = int(len(active) - np.argmax(active[::-1]))
    return contact, separation

# VLM Refinement
def refine_with_vlm(current_idx: int, creds: Dict[str, Any], video_path: str, prompt: str) -> int:
    """Ask GPT‑4(o) to confirm or shift the key frame.  Fall back to current_idx."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return current_idx

    res = scene_understanding(creds, frame, prompt)
    if isinstance(res, tuple):
        choice, _ = res
    else:
        choice = res
    try:
        new_idx = int(choice)
    except Exception:
        return current_idx
    return current_idx if new_idx in (-1, None) else new_idx

# 3‑D hand‑speed extraction + visualisation (SIMPLIFIED VERSION)
def extract_3d_speed_and_visualize(video_path: str, output_dir: str, *, device: str = "cuda", encoder: str = "vits"):
    """
    Bimanual 3‑D speed extraction with depth and ICP registration.
    - Detect both wrists per frame (depth‑guided + ViTPose)
    - Build coloured point clouds and register to frame 1 for global coords
    - Save per‑hand speeds: <video>_speed_right.json and _left.json
    - Save combined speed (max per frame): <video>_speed.json
    - Emit separate speed plots for both hands and combined
    """
    cpm = _get_vitpose_model(device)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_name = Path(video_path).stem

    depth_dir = output_dir / "depth"
    depth_vis_path = depth_dir / "depth_vis.mp4"
    debug_dir = output_dir / "debug_frames"  # debug visualization
    debug_dir.mkdir(parents=True, exist_ok=True)

    vda_ready = (depth_dir / "pred_depth_000000.npy").exists()
    if not depth_vis_path.exists() or not vda_ready:
        print("[3D‑Pipeline] Generating depth (tensors + video) …")
        depth_dir.mkdir(parents=True, exist_ok=True)
        generate_depth_video_vda(video_path, depth_dir, device=device, encoder=encoder)
    else:
        print("[3D‑Pipeline] Reusing cached depth outputs in", depth_dir)
    
    max_repairs = 5          # keeps run-time bounded
    for attempt in range(max_repairs):
        bad_idx = _invalid_depth_indices(depth_dir)
        if not bad_idx:
            break

        pct = len(bad_idx) / len(list(depth_dir.glob("pred_depth_*.npy"))) * 100
        print(f"[depth] {len(bad_idx)} tensors invalid  ({pct:.1f} %) – repairing")

        _remove_depth_tensors(depth_dir, bad_idx)
        generate_depth_video_vda(video_path, depth_dir, device=device, encoder=encoder)
    else:
        raise RuntimeError("Depth repair failed after multiple attempts – aborting.")

    pcd_dir = output_dir / "pointclouds" / video_name
    if not (pcd_dir / "0.ply").exists():
        print("[PCD] Building point clouds …")
        _generate_pointclouds(depth_dir, video_path, pcd_dir)
    else:
        print("[PCD] Reusing cached point clouds in", pcd_dir)

    cam_hand_dir = output_dir / "hand3d_cam"
    cam_hand_dir.mkdir(exist_ok=True)
    cam_hand_json = cam_hand_dir / f"{video_name}.json"

    cap_rgb = cv2.VideoCapture(video_path)
    if not cap_rgb.isOpened():
        raise RuntimeError("Could not open RGB video.")

    total_frames = int(cap_rgb.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[3D‑Pipeline] Processing {total_frames} frames...")

    # Track both hands and store camera coords for registration
    speed_right: Dict[int, float] = {}
    speed_left: Dict[int, float] = {}
    prev_right_3d = None
    prev_left_3d = None
    right_detections = 0
    left_detections = 0
    total_frames_seen = 0
    cam_hand_right: Dict[str, List[float]] = {}
    cam_hand_left: Dict[str, List[float]] = {}

    for idx in range(total_frames):
        ok_rgb, frame_bgr = cap_rgb.read()
        if not ok_rgb:
            speed_right[idx] = 0.0
            continue

        gray_depth = _load_depth(depth_dir, idx)
        if gray_depth is None:
            speed_right[idx] = 0.0
            continue

        H, W = gray_depth.shape

        # Detect both hands (for visualization) but only use right for speed
        wrists = _wrist_from_frame(frame_bgr, gray_depth, cpm)
        total_frames_seen += 1
        
        # Create debug visualization every 30 frames
        if idx % 30 == 0:
            debug_frame = frame_bgr.copy()
            detection_info = f"Frame {idx}: "
            
            # Draw circles for both hands
            for hand_name, wrist in wrists.items():
                if wrist is not None:
                    u, v = wrist
                    color = (0, 0, 255) if hand_name == "right" else (255, 0, 0)  # Red for right, Blue for left
                    cv2.circle(debug_frame, (int(u), int(v)), 8, color, -1)
                    detection_info += f"{hand_name}✓ "
                else:
                    detection_info += f"{hand_name}✗ "
            
            # Add text overlay
            cv2.putText(debug_frame, detection_info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imwrite(str(debug_dir / f"debug_{idx:06d}.jpg"), debug_frame)

        # Helper to compute per-hand camera-frame 3D and per-frame speed
        def _accumulate_for_hand(tag: str):
            nonlocal prev_right_3d, prev_left_3d, right_detections, left_detections
            wrist = wrists.get(tag)
            if wrist is None:
                if tag == "right":
                    speed_right[idx] = 0.0
                    prev_right_3d = None
                else:
                    speed_left[idx] = 0.0
                    prev_left_3d = None
                return

            u, v = wrist
            u_safe = max(0, min(int(u), W - 1))
            v_safe = max(0, min(int(v), H - 1))
            z_depth = float(gray_depth[v_safe, u_safe])
            if z_depth <= 0 or z_depth > 5.0:
                if tag == "right":
                    speed_right[idx] = 0.0
                    prev_right_3d = None
                else:
                    speed_left[idx] = 0.0
                    prev_left_3d = None
                return

            X, Y, Z = _pixel_to_camera(u, v, z_depth * 1000, W, H)
            cur = np.array([X, Y, Z])
            if tag == "right":
                cam_hand_right[str(idx + 1)] = [float(X), float(Y), float(Z)]
                if prev_right_3d is None:
                    speed_right[idx] = 0.0
                else:
                    speed_right[idx] = float(np.linalg.norm(cur - prev_right_3d))
                prev_right_3d = cur
                right_detections += 1
            else:
                cam_hand_left[str(idx + 1)] = [float(X), float(Y), float(Z)]
                if prev_left_3d is None:
                    speed_left[idx] = 0.0
                else:
                    speed_left[idx] = float(np.linalg.norm(cur - prev_left_3d))
                prev_left_3d = cur
                left_detections += 1

        _accumulate_for_hand("right")
        _accumulate_for_hand("left")

        if (idx + 1) % 100 == 0:
            r_rate = (right_detections / total_frames_seen) * 100 if total_frames_seen > 0 else 0
            l_rate = (left_detections / total_frames_seen) * 100 if total_frames_seen > 0 else 0
            print(f"[3D‑Pipeline] Processed {idx + 1}/{total_frames} | Det. rate – R: {r_rate:.1f}% | L: {l_rate:.1f}%")

    cap_rgb.release()

    # Save camera-frame coordinates per hand
    cam_hand_right_dir = cam_hand_dir / "right"
    cam_hand_left_dir = cam_hand_dir / "left"
    cam_hand_right_dir.mkdir(parents=True, exist_ok=True)
    cam_hand_left_dir.mkdir(parents=True, exist_ok=True)
    with open(cam_hand_right_dir / f"{video_name}.json", "w") as f:
        json.dump(cam_hand_right, f, indent=2)
    with open(cam_hand_left_dir / f"{video_name}.json", "w") as f:
        json.dump(cam_hand_left, f, indent=2)

    # Run ICP registration per hand to get globally consistent coordinates
    reg_out_right = output_dir / "registered_hands_right"
    reg_out_left = output_dir / "registered_hands_left"
    register_hand_positions(str(pcd_dir.parent), str(cam_hand_right_dir), str(reg_out_right))
    register_hand_positions(str(pcd_dir.parent), str(cam_hand_left_dir), str(reg_out_left))
    
    # Load registered coordinates and calculate world-frame speeds
    with open(reg_out_right / f"{video_name}.json") as f:
        reg_hand_right = json.load(f)
    with open(reg_out_left / f"{video_name}.json") as f:
        reg_hand_left = json.load(f)
    
    # Recalculate speed using globally registered coordinates
    def _world_speed_from_reg(reg_hand: Dict[str, List[float]]) -> Dict[int, float]:
        out: Dict[int, float] = {}
        prev = None
        for i in range(total_frames):
            key = str(i + 1)
            if key in reg_hand:
                xyz = np.array(reg_hand[key], dtype=float)
                if prev is None:
                    out[i] = 0.0
                else:
                    out[i] = float(np.linalg.norm(xyz - prev))
                prev = xyz
            else:
                out[i] = 0.0
                prev = None
        return out

    world_speed_right = _world_speed_from_reg(reg_hand_right)
    world_speed_left = _world_speed_from_reg(reg_hand_left)
    world_speed_combined: Dict[int, float] = {
        i: float(max(world_speed_right.get(i, 0.0), world_speed_left.get(i, 0.0)))
        for i in range(total_frames)
    }

    r_rate = (right_detections / total_frames) * 100 if total_frames > 0 else 0
    l_rate = (left_detections / total_frames) * 100 if total_frames > 0 else 0
    print(f"[STATS] Detection rates – Right: {r_rate:.1f}% | Left: {l_rate:.1f}%")
    for tag, sp in ("right", world_speed_right), ("left", world_speed_left):
        valid = [v for v in sp.values() if v > 0]
        print(f"[STATS] {tag.capitalize()} valid speed frames: {len(valid)}/{total_frames}")

    # Save world-frame speed per hand and combined
    speed_json_right = output_dir / f"{video_name}_speed_right.json"
    speed_json_left = output_dir / f"{video_name}_speed_left.json"
    speed_json_combined = output_dir / f"{video_name}_speed.json"
    with open(speed_json_right, "w") as f:
        json.dump(world_speed_right, f, indent=2)
    with open(speed_json_left, "w") as f:
        json.dump(world_speed_left, f, indent=2)
    with open(speed_json_combined, "w") as f:
        json.dump(world_speed_combined, f, indent=2)

    # Apply clipping and smoothing for visualization
    # Prepare arrays for plotting convenience
    frames = list(world_speed_right.keys())
    speeds_r = np.array(list(world_speed_right.values()))
    speeds_l = np.array(list(world_speed_left.values()))
    speeds_c = np.array(list(world_speed_combined.values()))
    
    # 1. Clipping: Remove unreasonable speed values (outliers)
    # Reasonable hand speed: up to 150mm per frame (adjustable threshold)
    speed_threshold = 150.0  # mm per frame - adjustable
    
    def _clip_and_smooth(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        clipped = arr.copy()
        mask = (clipped > speed_threshold) & (clipped > 0)
        clipped[mask] = speed_threshold
        non_zero_mask = clipped > 0
        if non_zero_mask.any():
            nz = clipped[non_zero_mask]
            if len(nz) > 3:
                clipped[non_zero_mask] = gaussian_filter1d(nz, sigma=2.0)
        return clipped, mask

    speeds_r_s, out_r = _clip_and_smooth(speeds_r)
    speeds_l_s, out_l = _clip_and_smooth(speeds_l)
    speeds_c_s, out_c = _clip_and_smooth(speeds_c)
    
    # Plot per-hand and combined
    def _plot(arr: np.ndarray, clipped: np.ndarray, mask: np.ndarray, title: str, out_path: Path):
        plt.figure(figsize=(12, 6))
        non_zero = arr > 0
        if non_zero.any():
            plt.plot(np.array(frames)[non_zero], arr[non_zero], 'lightgray', alpha=0.3, linewidth=0.8, label='Original')
        nzs = clipped > 0
        if nzs.any():
            plt.plot(np.array(frames)[nzs], clipped[nzs], 'b-', linewidth=2, label='Clipped & smoothed')
        if mask.any():
            plt.plot(np.array(frames)[mask], arr[mask], 'orange', marker='x', linestyle='None', markersize=4, label='Clipped outliers')
        plt.xlabel("Frame Index")
        plt.ylabel("Speed (mm/frame)")
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()

    vis_right = output_dir / f"{video_name}_speed_vis_right.png"
    vis_left = output_dir / f"{video_name}_speed_vis_left.png"
    vis_combined = output_dir / f"{video_name}_speed_vis.png"
    _plot(speeds_r, speeds_r_s, out_r, f"Right Hand Speed - {video_name}", vis_right)
    _plot(speeds_l, speeds_l_s, out_l, f"Left Hand Speed - {video_name}", vis_left)
    _plot(speeds_c, speeds_c_s, out_c, f"Combined (max) Speed - {video_name}", vis_combined)

    print(f"[OUTPUT] Speed data (right): {speed_json_right}")
    print(f"[OUTPUT] Speed data (left):  {speed_json_left}")
    print(f"[OUTPUT] Speed data (combined): {speed_json_combined}")
    print(f"[OUTPUT] Speed vis (right): {vis_right}")
    print(f"[OUTPUT] Speed vis (left):  {vis_left}")
    print(f"[OUTPUT] Speed vis (combined): {vis_combined}")
    print(f"[OUTPUT] Debug frames: {debug_dir}")

    # Backward-compatible return of the combined artefacts
    return world_speed_combined, str(speed_json_combined), str(vis_combined), str(depth_vis_path)

# Point Cloud Generation
def _generate_pointclouds(depth_dir: Path,  video_path: str, pcd_out_dir: Path, intrinsics: Optional[Tuple[float,float,float,float]] = None):
    """
    Create a coloured .ply point cloud for every frame whose depth tensor exists.
      • depth_dir   : folder with pred_depth_000000.npy … (metres)
      • video_path  : original RGB video (to grab colours)
      • pcd_out_dir : output <idx>.ply files
    """
    pcd_out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    if intrinsics is None:
        fx = fy = max(H, W)          # <-- quick default; replace with calibrated fx,fy
        cx, cy = W / 2.0, H / 2.0
    else:
        fx, fy, cx, cy = intrinsics

    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(W, H, fx, fy, cx, cy)

    depth_files = sorted(depth_dir.glob("pred_depth_*.npy"))
    for dfile in depth_files:
        idx = int(dfile.stem.split("_")[-1])
        # grab colour frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame_bgr = cap.read()
        if not ok:
            continue
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        depth_m = _load_depth(depth_dir, idx)      # **same     depth   maths** everywhere
        if depth_m is None:
            continue

        # Open3D expects depth in millimetres by default (depth_scale=1000)
        depth_o3d = o3d.geometry.Image((depth_m * 1000).astype(np.uint16))
        color_o3d = o3d.geometry.Image(frame_rgb)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d, depth_scale=1000.0,
            depth_trunc=4.0, convert_rgb_to_intensity=False
        )
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
        o3d.io.write_point_cloud(str(pcd_out_dir / f"{idx}.ply"), pcd, write_ascii=False)
    cap.release()
    print(f"[PCD] Generated {len(depth_files)} point clouds in {pcd_out_dir}")

# Video Convert
def convert_video(video_path: str, action: str, credentials: Dict[str, Any], grid_size: int, speed_folder: str, max_feedbacks: int = MAX_FEEDBACKS, repeat_times: int = 3, *, hand: str = "either"):
    """Driver wrapper (identical logic to 2-D)."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    contact_list, separate_list = [], []
    for _ in range(repeat_times):
        frame_start = process_task(
            credentials,
            video_path,
            action,
            grid_size,
            total_frames,
            "start",
            speed_folder,
            hand=hand,
        )
        frame_end = process_task(
            credentials,
            video_path,
            action,
            grid_size,
            total_frames,
            "end",
            speed_folder,
            hand=hand,
        )
        frame_contact = feedback_contact(
            credentials,
            video_path,
            action,
            grid_size,
            total_frames,
            frame_start,
            max_feedbacks,
            "start",
            speed_folder,
            hand=hand,
        )
        frame_separate = feedback_separation(
            credentials,
            video_path,
            action,
            grid_size,
            total_frames,
            frame_end,
            max_feedbacks,
            "end",
            speed_folder,
            hand=hand,
        )
        contact_list.append(frame_contact)
        separate_list.append(frame_separate)

    contact_list = [x for x in contact_list if x not in (None, -1)]
    separate_list = [x for x in separate_list if x not in (None, -1)]
    final_contact = -1 if not contact_list else int(round(np.mean(contact_list)))
    final_separate = -1 if not separate_list else int(round(np.mean(separate_list)))
    return final_contact, final_separate

# Tiny visual helper
def visualize_frame(video_path: str, idx: int, out_path: str, label: Optional[str] = None):
    if idx < 0:
        return
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print(f"[WARN] Could not seek to frame {idx} in {video_path}")
        return
    if label:
        cv2.putText(frame, label, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
    cv2.imwrite(out_path, frame)
    print(f"Visualized frame {idx} to {out_path}")


# Diagnostic function for debugging hand detection issues
def diagnose_video_and_detection(
    video_path: str,
    depth_dir: Path,
    output_dir: Path,
    *,
    device: str = "cuda",
    sample_frames: int = 10,
    save_all_frames: bool = False,
) -> None:
    """
    Diagnostic function to understand why hand detection might be failing.
    Samples frames and reports on video properties, depth quality, and detection success.
    If *save_all_frames* is True, RGB frames with detected wrists (red dots) **and**
    corresponding depth visualisations will be saved for *every* frame, not just the
    sampled ones.
    """
    print(f"\n=== DIAGNOSTIC REPORT FOR {video_path} ===")
    
    # 1. Video properties analysis
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("ERROR: Cannot open video file")
        return
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
    
    print(f"Video Properties:")
    print(f"   • Resolution: {width}x{height}")
    print(f"   • Total frames: {total_frames}")
    print(f"   • FPS: {fps}")
    print(f"   • Codec: {fourcc_str}")
    
    # 2. Sample frame analysis
    cpm = _get_vitpose_model(device)
    detection_success = 0
    depth_success = 0
    
    if save_all_frames:
        sample_indices = np.arange(total_frames, dtype=int)
    else:
        sample_indices = np.linspace(0, total_frames - 1, sample_frames, dtype=int)

    # output folders for JPG dumps
    rgb_dump_dir = output_dir / "diagnostic_rgb"
    depth_dump_dir = output_dir / "diagnostic_depth"
    if save_all_frames:
        rgb_dump_dir.mkdir(parents=True, exist_ok=True)
        depth_dump_dir.mkdir(parents=True, exist_ok=True)

    print(f"Sampling {sample_frames} frames for analysis:")
    
    for i, frame_idx in enumerate(sample_indices):
        # Read RGB frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame_bgr = cap.read()
        if not ok:
            print(f"   Frame {frame_idx}: Could not read RGB")
            continue
            
        # Check depth
        depth = _load_depth(depth_dir, frame_idx)
        if depth is None:
            print(f"   Frame {frame_idx}: No valid depth data")
            continue
        else:
            depth_success += 1
            
        # Check depth statistics
        depth_min, depth_max = depth.min(), depth.max()
        depth_mean = depth.mean()
        
        # Check hand detection
        wrists = _wrist_from_frame(frame_bgr, depth, cpm)
        hands_detected = sum(1 for w in wrists.values() if w is not None)
        
        if hands_detected > 0:
            detection_success += 1
            status = f"{hands_detected} hands detected"
        else:
            status = "No hands detected"
            
        print(
            f"   Frame {frame_idx}: {status}, Depth: {depth_min:.2f}-{depth_max:.2f}m (avg: {depth_mean:.2f}m)"
        )

        # Dump annotated RGB + depth images (only when asked)
        if save_all_frames:
            vis_rgb = frame_bgr.copy()
            for pt in wrists.values():
                if pt is not None:
                    u, v = map(int, pt)
                    cv2.circle(vis_rgb, (u, v), 6, (0, 0, 255), -1)  # red dot

            cv2.imwrite(str(rgb_dump_dir / f"rgb_{frame_idx:06d}.jpg"), vis_rgb)

            # Convert depth (metres) to inverse depth for better contrast then to 8-bit
            inv = 1.0 / (depth + 1e-6)
            inv_norm = cv2.normalize(inv, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            depth_col = cv2.applyColorMap(inv_norm, cv2.COLORMAP_INFERNO)
            cv2.imwrite(
                str(depth_dump_dir / f"depth_{frame_idx:06d}.jpg"), depth_col
            )
    
    cap.release()
    
    # 3. Summary
    detection_rate = (detection_success / sample_frames) * 100
    depth_rate = (depth_success / sample_frames) * 100
    
    print(f"\nSummary:")
    print(f"   • Valid depth frames: {depth_success}/{sample_frames} ({depth_rate:.1f}%)")
    print(f"   • Successful hand detection: {detection_success}/{sample_frames} ({detection_rate:.1f}%)")
    
    if detection_rate < 50:
        print(f"WARNING: Low hand detection rate ({detection_rate:.1f}%)")
        print(f"   This suggests:")
        print(f"   • Video format/encoding issues affecting hand detection")
        print(f"   • Poor depth quality from Video-Depth-Anything")
        print(f"   • Color space/compression artifacts")
        print(f"   • Hands not clearly visible in frames")
    
    if depth_rate < 90:
        print(f"WARNING: Poor depth data quality ({depth_rate:.1f}%)")
        print(f"   This suggests Video-Depth-Anything is struggling with this video")
    
    # 4. Save sample debug frame
    if sample_indices.size > 0 and not save_all_frames:
        mid_idx = sample_indices[len(sample_indices)//2]
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_idx)
        ok, frame = cap.read()
        if ok:
            debug_path = output_dir / f"debug_frame_{mid_idx}.jpg"
            cv2.imwrite(str(debug_path), frame)
            print(f"Saved debug frame: {debug_path}")
        cap.release()
    
    print("=" * 60)

# CLI entry point (with feedback loop)
def main():
    parser = argparse.ArgumentParser("EgoLoc 3‑D Demo (bimanual edition)")
    parser.add_argument("--video_path", required=True, help="Input video")
    parser.add_argument(
        "--action", default="Grasping the object", help="Action label shown to GPT-4o"
    )
    parser.add_argument(
        "--hand",
        default="both",
        choices=["both", "either", "right", "left"],
        help="Which hand(s) to analyze in temporal localization"
    )
    parser.add_argument(
        "--grid_size", type=int, default=3, help="Grid size for numbered frame grids"
    )
    parser.add_argument("--output_dir", default="output", help="Output folder")
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu", "auto"],
        help="Computation device",
    )
    parser.add_argument(
        "--credentials", required=True, help="Path to .env with OpenAI / Azure keys"
    )
    parser.add_argument(
        "--encoder",
        default="vits",
        choices=["vits", "vitl"],
        help="Video-Depth-Anything backbone: 'vits' (small) or 'vitl' (large)",
    )
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Run diagnostic mode to analyze video and hand detection issues",
    )
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch and torch.cuda.is_available() else "cpu"
    elif args.device == "cuda" and (not torch or not torch.cuda.is_available()):
        print("[WARN] CUDA requested but unavailable – falling back to CPU")
        device = "cpu"
    else:
        device = args.device

    if args.diagnose:
        print("\nDIAGNOSTIC MODE - Analyzing video and hand detection...")
        
        # Run minimal depth generation for diagnostics
        output_dir = Path(args.output_dir)
        depth_dir = output_dir / "depth"
        if not (depth_dir / "pred_depth_000000.npy").exists():
            print("Generating depth data for diagnosis...")
            generate_depth_video_vda(args.video_path, depth_dir, device=device, encoder=args.encoder)
        
        # Run diagnostics (dumping per-frame RGB+depth)
        diagnose_video_and_detection(
            args.video_path,
            depth_dir,
            output_dir,
            device=device,
            save_all_frames=True,
        )
        print("Diagnostic complete. Exiting.")
        return

    print("\n [1/3] Extracting 3‑D hand speed (bimanual) and visualizing ...")
    speed_dict, speed_json, speed_vis, depth_vid = extract_3d_speed_and_visualize(
        args.video_path,
        args.output_dir,
        device=device,
        encoder=args.encoder,
    )

    print("\n [2/3] Locating contact/separation frames and visualizing ...")
    credentials = dotenv.dotenv_values(args.credentials)

    # Validate speed data before proceeding
    video_name = Path(args.video_path).stem
    def _validate(path: Path, label: str):
        if path.exists():
            with open(path, 'r') as f:
                data = json.load(f)
            valid = [v for v in data.values() if v != 0 and not math.isnan(v)]
            print(f"[INFO] {label} speed frames valid: {len(valid)}/{len(data)}")
    _validate(Path(args.output_dir) / f"{video_name}_speed_right.json", "Right")
    _validate(Path(args.output_dir) / f"{video_name}_speed_left.json", "Left")
    _validate(Path(args.output_dir) / f"{video_name}_speed.json", "Combined")

    results: Dict[str, Dict[str, int]] = {}
    def _run_localize(tag: str):
        c_idx, s_idx = convert_video(
            args.video_path,
            args.action,
            credentials,
            args.grid_size,
            args.output_dir,
            max_feedbacks=1,
            repeat_times=1,
            hand=tag,
        )
        results[tag] = {"contact_frame": c_idx, "separation_frame": s_idx}
        # Visualize
        contact_vis = Path(args.output_dir) / f"{video_name}_contact_frame_{tag}.png"
        separation_vis = Path(args.output_dir) / f"{video_name}_separation_frame_{tag}.png"
        visualize_frame(args.video_path, c_idx, str(contact_vis), f"Contact ({tag})")
        visualize_frame(args.video_path, s_idx, str(separation_vis), f"Separation ({tag})")
        return contact_vis, separation_vis

    if args.hand in ("both", "right"):
        r_contact_vis, r_sep_vis = _run_localize("right")
        print(f"Resolved (right) ➜ contact: {results['right']['contact_frame']}, separation: {results['right']['separation_frame']}")
    if args.hand in ("both", "left"):
        l_contact_vis, l_sep_vis = _run_localize("left")
        print(f"Resolved (left)  ➜ contact: {results['left']['contact_frame']}, separation: {results['left']['separation_frame']}")
    if args.hand == "either":
        c_contact_vis, c_sep_vis = _run_localize("either")
        print(f"Resolved (combined) ➜ contact: {results['either']['contact_frame']}, separation: {results['either']['separation_frame']}")

    print("\n [3/3] Saving results ...")
    results_path = Path(args.output_dir) / f"{video_name}_result_bimanual.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print("EgoLoc bimanual output:", results)
    print(f"Results saved to: {results_path}")
    
    # Speed data files
    right_speed = Path(args.output_dir) / f"{video_name}_speed_right.json"
    left_speed = Path(args.output_dir) / f"{video_name}_speed_left.json" 
    combined_speed = Path(args.output_dir) / f"{video_name}_speed.json"
    print(f"Speed data - Right: {right_speed}")
    print(f"Speed data - Left: {left_speed}")
    print(f"Speed data - Combined: {combined_speed}")
    
    # Visualization files
    right_vis = Path(args.output_dir) / f"{video_name}_speed_vis_right.png"
    left_vis = Path(args.output_dir) / f"{video_name}_speed_vis_left.png"
    combined_vis = Path(args.output_dir) / f"{video_name}_speed_vis.png"
    print(f"Visualizations - Right: {right_vis}")
    print(f"Visualizations - Left: {left_vis}")
    print(f"Visualizations - Combined: {combined_vis}")
    
    print(f"Depth video: {depth_vid}")


if __name__ == "__main__":
    main()