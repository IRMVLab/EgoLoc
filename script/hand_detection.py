"""
HaMeR-based hand detection and 3D wrist position extraction
"""
import os
import sys
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import cv2
import numpy as np
import torch

try:
    
    # Add hamer directory to path for vitpose_model import
    # Get the absolute path to the EgoLoc directory (parent of egoloc3d)
    current_file = Path(__file__).resolve()
    egoloc_dir = current_file.parent.parent
    hamer_dir = egoloc_dir / "hamer"
    
    if hamer_dir.exists() and str(hamer_dir) not in sys.path:
        sys.path.insert(0, str(hamer_dir))

    from hamer.configs import CACHE_DIR_HAMER
    from hamer.models import download_models, load_hamer, DEFAULT_CHECKPOINT
    from hamer.utils import recursive_to
    from hamer.datasets.vitdet_dataset import ViTDetDataset
    from hamer.utils.renderer import cam_crop_to_full
    from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
    # from vitpose_model import ViTPoseModel

    HAMER_AVAILABLE = True
except ImportError as e:
    HAMER_AVAILABLE = False
    print(f"Warning: HaMeR dependencies not available: {e}")

from .config import Config


class HaMeRHandDetector:
    """Hand detector using HaMeR for 3D wrist position estimation"""

    def __init__(
        self,
        device: str = "cuda",
        checkpoint: Optional[str] = None,
        body_detector: str = "vitdet",
        rescale_factor: float = 2.0
    ):
        """
        Initialize HaMeR hand detector
        
        Args:
            device: Computation device ('cuda' or 'cpu')
            checkpoint: Path to HaMeR checkpoint (None for default)
            body_detector: Body detector type ('vitdet' or 'regnety')
            rescale_factor: Factor for padding the bbox
        """
        if not HAMER_AVAILABLE:
            raise ImportError("HaMeR dependencies not available. Please install hamer.")
        
        self.device = torch.device(device) if torch.cuda.is_available() else torch.device('cpu')
        self.rescale_factor = rescale_factor

        current_file = Path(__file__).resolve()
        egoloc_dir = current_file.parent.parent
        hamer_dir = egoloc_dir / "hamer"

        checkpoint = checkpoint or DEFAULT_CHECKPOINT

        if not Path(checkpoint).is_absolute():
            checkpoint = str(hamer_dir / checkpoint.lstrip('./'))
        
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.exists():
            print(f"[HaMeR] Checkpoint not found at {checkpoint}, downloading models...")
            try:
                original_cwd = os.getcwd()
                os.chdir(str(hamer_dir))
                try:
                    download_models(CACHE_DIR_HAMER)
                finally:
                    os.chdir(original_cwd)
            except Exception as e:
                print(f"[HaMeR] Warning: Failed to download models: {e}")
                print(f"[HaMeR] Please run: cd hamer && bash fetch_demo_data.sh")
                raise RuntimeError(
                    "HaMeR model not found. Please download it manually using: "
                    "cd hamer && bash fetch_demo_data.sh"
                )
        else:
            print(f"[HaMeR] Using existing checkpoint: {checkpoint}")

        original_cwd = os.getcwd()
        try:
            os.chdir(str(hamer_dir))
            self.model, self.model_cfg = load_hamer(checkpoint)
        finally:
            os.chdir(original_cwd)
        self.model = self.model.to(self.device)
        self.model.eval()

        self._setup_body_detector(body_detector)

        if str(hamer_dir) not in sys.path:
            sys.path.insert(0, str(hamer_dir))
        import vitpose_model as _vpm
        _vpm.ROOT_DIR = str(hamer_dir)
        _vpm.VIT_DIR = str(hamer_dir / "third-party" / "ViTPose")
        cfg_rel = Path(
            "configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/"
            "coco-wholebody/ViTPose_huge_wholebody_256x192.py"
        )
        ckpt_rel = Path("_DATA/vitpose_ckpts/vitpose+_huge/wholebody.pth")
        for _name, _dic in _vpm.ViTPoseModel.MODEL_DICT.items():
            _dic["config"] = str(hamer_dir / "third-party" / "ViTPose" / cfg_rel)
            _dic["model"] = str(hamer_dir / ckpt_rel)

        self.vitpose = _vpm.ViTPoseModel(self.device)
    
    def _setup_body_detector(self, detector_type: str):
        """Setup Detectron2-based body detector"""
        if detector_type == "vitdet":
            from detectron2.config import LazyConfig
            import hamer
            
            cfg_path = Path(hamer.__file__).parent / 'configs' / 'cascade_mask_rcnn_vitdet_h_75ep.py'
            detectron2_cfg = LazyConfig.load(str(cfg_path))

            checkpoint_paths = [
                "/home/hamer/models/model_final_f05665.pkl",
                Path(hamer.__file__).parent.parent / "models" / "model_final_f05665.pkl",
            ]
            checkpoint_found = False
            for cp in checkpoint_paths:
                if Path(cp).exists():
                    detectron2_cfg.train.init_checkpoint = str(cp)
                    checkpoint_found = True
                    break

            if not checkpoint_found:
                detectron2_cfg.train.init_checkpoint = (
                    "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/"
                    "cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
                )
            
            for i in range(3):
                detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
            self.detector = DefaultPredictor_Lazy(detectron2_cfg)
        
        elif detector_type == "regnety":
            from detectron2 import model_zoo
            detectron2_cfg = model_zoo.get_config(
                'new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py', trained=True
            )
            detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
            detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh = 0.4
            self.detector = DefaultPredictor_Lazy(detectron2_cfg)
        else:
            raise ValueError(f"Unknown detector type: {detector_type}")
    
    def detect_wrist_3d(self, frame_bgr: np.ndarray) -> Optional[Tuple[float, float, float]]:
        """
        Detect 3D wrist position from a frame
        
        Args:
            frame_bgr: Input frame in BGR format (H, W, 3)
        
        Returns:
            (X, Y, Z) in camera coordinates, or None if not detected
        """
        img_cv2 = frame_bgr.copy()
        img = img_cv2.copy()[:, :, ::-1]

        det_out = self.detector(img_cv2)
        det_instances = det_out['instances']
        valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
        pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        pred_scores = det_instances.scores[valid_idx].cpu().numpy()

        if len(pred_bboxes) == 0:
            return None

        vitposes_out = self.vitpose.predict_pose(
            img,
            [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
        )

        if len(vitposes_out) == 0:
            return None

        vitposes = vitposes_out[0]
        right_hand_keyp = vitposes['keypoints'][-21:]

        valid = right_hand_keyp[:, 2] > 0.5
        if valid.sum() <= 3:
            return None

        valid_keyp = right_hand_keyp[valid]
        bbox = [
            valid_keyp[:, 0].min(),
            valid_keyp[:, 1].min(),
            valid_keyp[:, 0].max(),
            valid_keyp[:, 1].max()
        ]
        bboxes = np.array([bbox])
        is_right = np.array([1])

        dataset = ViTDetDataset(
            self.model_cfg, img_cv2, bboxes, is_right,
            rescale_factor=self.rescale_factor
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=0
        )
        
        with torch.no_grad():
            for batch in dataloader:
                batch = recursive_to(batch, self.device)
                out = self.model(batch)

                pred_cam = out['pred_cam']
                multiplier = (2 * batch['right'] - 1)
                pred_cam[:, 1] = multiplier * pred_cam[:, 1]

                box_center = batch["box_center"].float()
                box_size = batch["box_size"].float()
                img_size = batch["img_size"].float()
                scaled_focal_length = (
                    self.model_cfg.EXTRA.FOCAL_LENGTH / self.model_cfg.MODEL.IMAGE_SIZE *
                    img_size.max()
                )
                pred_cam_t_full = cam_crop_to_full(
                    pred_cam, box_center, box_size, img_size, scaled_focal_length
                ).detach().cpu().numpy()

                cam_t = pred_cam_t_full[0]
                return tuple(cam_t.tolist())

        return None
    
    def process_video(self, video_path: str, output_json: Optional[str] = None,
                     fps: Optional[float] = None) -> Dict[str, List[float]]:
        """
        Process video and extract 3D wrist positions for all frames
        
        Args:
            video_path: Path to input video
            output_json: Optional path to save results as JSON
            fps: Video FPS (if None, reads from video)
        
        Returns:
            Dict mapping frame_id (as string) to [X, Y, Z] camera coordinates
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        
        if fps is None:
            fps = cap.get(cv2.CAP_PROP_FPS)
        
        hand_coords = {}
        frame_idx = 0
        
        print(f"[HaMeR] Processing video: {video_path}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            wrist_3d = self.detect_wrist_3d(frame)
            if wrist_3d is not None:
                hand_coords[str(frame_idx + 1)] = list(wrist_3d)
            
            frame_idx += 1
            if (frame_idx + 1) % 100 == 0:
                print(f"[HaMeR] Processed {frame_idx + 1} frames")
        
        cap.release()
        print(f"[HaMeR] Completed: detected {len(hand_coords)} frames with valid wrist positions")
        
        if output_json:
            Path(output_json).parent.mkdir(parents=True, exist_ok=True)
            with open(output_json, 'w') as f:
                json.dump(hand_coords, f, indent=2)
            print(f"[HaMeR] Saved results to {output_json}")
        
        return hand_coords

