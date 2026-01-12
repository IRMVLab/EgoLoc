"""
Vision-Language Model inference for frame selection and feedback
"""
import cv2
import base64
import time
import re
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from openai import OpenAI, AzureOpenAI
import openai

from .config import Config


def image_resize_for_vlm(frame: np.ndarray, inter: int = cv2.INTER_AREA) -> np.ndarray:
    """Resize image for VLM (short side ≤ 768px, long side ≤ 2000px)"""
    height, width = frame.shape[:2]
    aspect_ratio = width / height
    max_short_side = 768
    max_long_side = 2000
    
    if aspect_ratio > 1:
        new_width = min(width, max_long_side)
        new_height = int(new_width / aspect_ratio)
        if new_height > max_short_side:
            new_height = max_short_side
            new_width = int(new_height * aspect_ratio)
    else:
        new_height = min(height, max_long_side)
        new_width = int(new_height * aspect_ratio)
        if new_width > max_short_side:
            new_width = max_short_side
            new_height = int(new_width / aspect_ratio)
    
    return cv2.resize(frame, (new_width, new_height), interpolation=inter)


def extract_event_info(response: str) -> Optional[str]:
    """Extract event type from VLM response"""
    event_match = re.search(r"Event:\s*(Contact|Separation|Neither|None)", response, re.IGNORECASE)
    if event_match:
        return event_match.group(1).capitalize()
    return None


def extract_frame_info(response: str) -> int:
    """Extract frame number from VLM response"""
    frame_match = re.search(r"Frame:\s*(-?\d+)", response)
    if not frame_match:
        return -1
    try:
        return int(frame_match.group(1))
    except ValueError:
        return -1


def scene_understanding(
    credentials: Dict[str, Any],
    frame: np.ndarray,
    prompt_message: str,
    principle: Optional[str] = None,
    negative_example: Optional[np.ndarray] = None,
    negative_examples: Optional[List[np.ndarray]] = None
) -> Any:
    """
    Use VLM for scene understanding.
    
    Args:
        credentials: API credentials dict
        frame: Input frame/image
        prompt_message: Prompt text
        principle: 'state', 'feedback', or None (for frame selection)
        negative_example: Optional single negative example image for in-context learning (deprecated, use negative_examples)
        negative_examples: Optional list of negative example images for in-context learning
    
    Returns:
        Result based on principle:
        - 'state': Event type string ('Contact', 'Separation', 'Neither', None)
        - 'feedback': Response string ('1' for positive, '0' for negative)
        - None: Frame index (int) or -1
    """
    frame = image_resize_for_vlm(frame)
    _, buffer = cv2.imencode(".jpg", frame)
    base64_frame = base64.b64encode(buffer).decode("utf-8")
    
    content = [
        {"type": "text", "text": prompt_message},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_frame}", "detail": "high"}
        }
    ]
    
    if negative_examples is not None and len(negative_examples) > 0:
        for neg_example in negative_examples:
            neg_frame = image_resize_for_vlm(neg_example)
            _, neg_buffer = cv2.imencode(".jpg", neg_frame)
            base64_neg = base64.b64encode(neg_buffer).decode("utf-8")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_neg}", "detail": "high"}
            })
    elif negative_example is not None:
        neg_frame = image_resize_for_vlm(negative_example)
        _, neg_buffer = cv2.imencode(".jpg", neg_frame)
        base64_neg = base64.b64encode(neg_buffer).decode("utf-8")
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_neg}", "detail": "high"}
        })
    
    messages = [{"role": "user", "content": content}]
    
    # Setup client
    if credentials.get("AZURE_OPENAI_API_KEY"):
        client = AzureOpenAI(
            api_version="2024-02-01",
            azure_endpoint=credentials["AZURE_OPENAI_ENDPOINT"],
            api_key=credentials["AZURE_OPENAI_API_KEY"]
        )
        params = {
            "model": credentials.get("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o"),
            "messages": messages,
            "max_tokens": Config.MAX_TOKENS,
            "temperature": Config.TEMPERATURE,
            "top_p": Config.TOP_P
        }
    else:
        client = OpenAI(
            api_key=credentials["OPENAI_API_KEY"],
            base_url="https://api.chatanywhere.tech/v1"
        )
        params = {
            "model": "gpt-4o",
            "messages": messages,
            "max_tokens": Config.MAX_TOKENS,
            "temperature": Config.TEMPERATURE,
            "top_p": Config.TOP_P
        }
    
    # Call API with retries
    count = 0
    while count < Config.MAX_RETRIES:
        try:
            result = client.chat.completions.create(**params)
            response_content = result.choices[0].message.content
            finish_reason = result.choices[0].finish_reason
            
            if finish_reason == 'length':
                if count < Config.MAX_RETRIES - 1:
                    params["max_tokens"] = params.get("max_tokens", Config.MAX_TOKENS) * 2
                    count += 1
                    time.sleep(1)
                    continue
            
            break
        except (openai.RateLimitError, openai.APIStatusError) as e:
            print(f"[VLM] API error: {e}, retrying...")
            time.sleep(2)
            count += 1
        except Exception as e:
            print(f"[VLM] Error: {e}")
            if principle == "state" or principle == "feedback":
                return None
            return -1
    else:
        raise RuntimeError(f"Failed to get VLM response after {Config.MAX_RETRIES} retries")
    
    # Parse response based on principle
    if principle == "state":
        return extract_event_info(response_content)
    elif principle == "feedback":
        return response_content.strip()
    else:
        return extract_frame_info(response_content)


# Prompt templates
PROMPT_CONTACT = """
Instruction:
You will be given a time-ordered image grid containing a hand and a target object. Your task is to find the earliest contact moment, where the hand first contacts with the object, and return its index.

Reasoning Steps:
1. Analyze each frame to observe the relationship between the hand and the object.
2. Identify the earliest transition from separation (hand not touching the object) → contact (hand touching the object).
3. If no contact is detected within the grid, return -1.

Strict Output Format:
If a contact moment is detected:
"Frame: X"  
(Where X is the index of the earliest contact frame)
If no contact is detected:
"Frame: -1"
"""

PROMPT_SEPARATION = """
Instruction:
You will be given a time-ordered image grid containing a hand and a target object. Your task is to find the earliest separation moment, where the hand first separates with the object, and return its index.

Reasoning Steps:
1. Analyze each frame to observe the relationship between the hand and the object.
2. Identify the earliest transition from contact (hand touching the object) → separation (hand moving away from the object).
3. If no separation is detected within the grid, return -1.

Strict Output Format:
If a separation moment is detected:
"Frame: X"  
(Where X is the index of the earliest separation frame)
If no separation is detected:
"Frame: -1"
"""

PROMPT_STATE = """
Instruction:
- You are given two image frames: a previous frame (Frame Left) and a future frame (Frame Right).
- Your task is to detect a **Contact Moment** or **Separation Moment** by analyzing the change in interaction between the hand and a target object of the two frames.
- Avoid outputting "Event: Neither" unless both frames clearly and confidently show the **same state with no visible change**.

Definitions:
- Contact: The hand is visibly touching or making physical contact with the object (e.g., fingers pressed against the surface, grasping).
- Separation: The hand is clearly not in contact with the object (e.g., fingers hovering, obvious gap, no overlap).

Steps:
1. Ensure the same target object is being observed in both frames.
2. For each frame:
- Check whether the hand is **clearly touching** or **clearly not touching** the object.
- If it's ambiguous, lean toward detecting **Contact** if the hand is near or aligned for grasp; lean toward **Separation** if the hand is retreating or open.
3. Decide the event based on the transition:
- If Frame Left = Separation and Frame Right = Contact → output: **Event: Contact**
- If Frame Left = Contact and Frame Right = Separation → output: **Event: Separation**
- If both frames show **no significant change**, and you are confident the contact state stayed the same → output: **Event: Neither**
- In cases of uncertainty or partial motion, prefer to output **Contact** or **Separation** over "Neither".

Output Format (Strict):
Output exactly one of the following as the final line:
- "Event: Contact"
- "Event: Separation"
- "Event: Neither"
"""

PROMPT_FEEDBACK_CONTACT = (
    "I will show an image of hand-object interaction. "
    "You need to help me determine whether the hand and the object in the current image are in contact rather than just appearing to be in contact. "
    "If yes, answer 1. If not, answer 0."
)

PROMPT_FEEDBACK_SEPARATION = (
    "I will show an image of hand-object interaction. "
    "You need to help me determine whether the hand and the object in the current image are in separate rather than just appearing to be in separate. "
    "If yes, answer 1. If not, answer 0."
)

PROMPT_CONTACT_WITH_NEGATIVE = """
Instruction:
You will be given two images. The first one is a time-ordered image grid containing a hand and a target object. Your task is to find the earliest contact moment, where the hand first contacts with the object, and return its index. The second image is a negative example of error localization, which has no actual hand-object contact.

Reasoning Steps:
1. First, examine the second image (negative example) to understand what does NOT constitute a valid contact moment - this frame was incorrectly identified as a contact moment, but it actually shows no real hand-object contact.
2. Analyze each frame in the first time-ordered image grid to observe the relationship between the hand and the object.
3. Compare each frame in the grid with the negative example to ensure you avoid similar false positives.
4. Identify the earliest transition from separation (hand not touching the object) → contact (hand touching the object), making sure it is a genuine contact moment, not just appearing to be in contact.
5. If no genuine contact is detected within the grid, return -1.

Strict Output Format:
If a contact moment is detected:
"Frame: X"  
(Where X is the index of the earliest contact frame)
If no contact is detected:
"Frame: -1"
"""

PROMPT_SEPARATION_WITH_NEGATIVE = """
Instruction:
You will be given two images. The first one is a time-ordered image grid containing a hand and a target object. Your task is to find the earliest separation moment, where the hand first separates with the object, and return its index. The second image is a negative example of error localization, which has no actual hand-object separation.

Reasoning Steps:
1. First, examine the second image (negative example) to understand what does NOT constitute a valid separation moment - this frame was incorrectly identified as a separation moment, but it actually shows no real hand-object separation.
2. Analyze each frame in the first time-ordered image grid to observe the relationship between the hand and the object.
3. Compare each frame in the grid with the negative example to ensure you avoid similar false positives.
4. Identify the earliest transition from contact (hand touching the object) → separation (hand moving away from the object), making sure it is a genuine separation moment, not just appearing to be separated.
5. If no genuine separation is detected within the grid, return -1.

Strict Output Format:
If a separation moment is detected:
"Frame: X"  
(Where X is the index of the earliest separation frame)
If no separation is detected:
"Frame: -1"
"""

