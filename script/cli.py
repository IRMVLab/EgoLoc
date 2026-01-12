"""
Command-line interface for EgoLoc3D
"""
import argparse
import json
from pathlib import Path

from .config import Config
from .pipeline import extract_3d_speed_and_visualize, convert_video
from .utils import visualize_frame


def main():
    parser = argparse.ArgumentParser(
        description="EgoLoc3D: 3D Temporal Interaction Localization"
    )
    
    # Required arguments
    parser.add_argument(
        "--video_path",
        required=True,
        type=str,
        help="Path to input video file"
    )
    parser.add_argument(
        "--credentials",
        required=False,
        default="auth.env",
        type=str,
        help="Path to .env file with OpenAI/Azure API credentials"
    )
    parser.add_argument(
        "--output_dir",
        default="output_long",
        type=str,
        help="Output directory for results"
    )
    
    # Optional arguments
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu", "auto"],
        help="Computation device"
    )
    parser.add_argument(
        "--encoder",
        default="vits",
        choices=["vits", "vitl"],
        help="Video-Depth-Anything encoder: 'vits' (small) or 'vitl' (large)"
    )
    parser.add_argument(
        "--speed_json",
        type=str,
        default=None,
        help="Path to pre-computed speed JSON file. If provided, skips VDA, HaMeR, ICP, and speed computation steps"
    )
    parser.add_argument(
        "--grid_size",
        type=int,
        default=2,
        help="Grid size for frame selection (grid_size^2 frames)"
    )
    parser.add_argument(
        "--max_feedback",
        type=int,
        default=1,
        help="Maximum feedback iterations for refinement"
    )
    parser.add_argument(
        "--use_feedback",
        action="store_true",
        default=True,
        help="Enable feedback mechanism"
    )
    parser.add_argument(
        "--video_type",
        default="short",
        choices=["short", "long"],
        help="Video type: 'short' (single action) or 'long' (multiple actions)"
    )
    
    args = parser.parse_args()
    
    # Auto device selection
    if args.device == "auto":
        try:
            import torch  # type: ignore
            args.device = "cuda" if torch and torch.cuda.is_available() else "cpu"
        except ImportError:
            args.device = "cpu"
    
    print("=" * 60)
    print("EgoLoc3D - 3D Temporal Interaction Localization")
    print("=" * 60)
    print(f"Video: {args.video_path}")
    print(f"Output: {args.output_dir}")
    print(f"Device: {args.device}")
    print(f"Encoder: {args.encoder}")
    if args.speed_json:
        print(f"Speed JSON: {args.speed_json} (using pre-computed)")
    print("=" * 60)
    
    # Load credentials
    credentials = Config.load_credentials(args.credentials)
    
    # Setup paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_name = Path(args.video_path).stem
    
    # Step 1: Extract 3D speed or use pre-computed speed
    if args.speed_json:
        # Use pre-computed speed JSON directly
        speed_json_path = Path(args.speed_json)
        if not speed_json_path.exists():
            raise ValueError(f"Speed JSON file not found: {args.speed_json}")
        print(f"\n[Step 1/3] Using pre-computed speed JSON: {speed_json_path}")
        print("  Skipping: VDA depth generation, HaMeR detection, point cloud generation, ICP registration")
        speed_json = str(speed_json_path)
    else:
        # Extract speed from video using VDA + HaMeR + ICP
        print("\n[Step 1/3] Extracting 3D hand speed...")
        speed_dict, speed_json, speed_vis = extract_3d_speed_and_visualize(
            args.video_path,
            str(output_dir),
            credentials,
            device=args.device,
            encoder=args.encoder
        )
        print("[Step 1] Speed extraction complete")
        print(f"  Speed JSON: {speed_json}")
        print(f"  Speed visualization: {speed_vis}")
    
    # Step 2: Temporal localization
    print("\n[Step 2/3] Locating contact/separation frames...")
    # Use VDA-optimized extraction only if speed was generated from video (not pre-computed)
    use_vda = (not args.speed_json) and (args.video_type == "long")
    pairs = convert_video(
        args.video_path,
        credentials,
        str(output_dir),
        speed_json,
        grid_size=args.grid_size,
        max_feedback=args.max_feedback,
        use_feedback=args.use_feedback,
        video_type=args.video_type,
        use_vda=use_vda
    )
    
    print("\n[Step 2] Temporal localization complete")
    print(f"  Detected {len(pairs)} contact/separation pair(s)")
    for i, (contact, separation) in enumerate(pairs):
        print(f"  Pair {i+1}: Contact @ frame {contact}, Separation @ frame {separation}")
    
    # Step 3: Save results and visualize
    print("\n[Step 3/3] Saving results...")
    result = {
        "video": video_name,
        "pairs": [{"contact": c, "separation": s} for c, s in pairs]
    }
    
    # For short videos, keep only first pair
    if args.video_type == "short" and len(pairs) > 0:
        contact_idx, separation_idx = pairs[0]
        result["contact_frame"] = contact_idx
        result["separation_frame"] = separation_idx
        
        # Visualize keyframes
        contact_vis = output_dir / f"{video_name}_contact_frame.png"
        separation_vis = output_dir / f"{video_name}_separation_frame.png"
        visualize_frame(args.video_path, contact_idx, str(contact_vis), "Contact")
        visualize_frame(args.video_path, separation_idx, str(separation_vis), "Separation")
    
    result_path = output_dir / f"{video_name}_result.json"
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print("\n[Step 3] Results saved")
    print(f"  Result JSON: {result_path}")
    
    # Final summary
    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)
    print(f"Result JSON: {result_path}")
    if args.video_type == "short" and len(pairs) > 0:
        print(f"Contact frame: {pairs[0][0]}")
        print(f"Separation frame: {pairs[0][1]}")
    print("=" * 60)


if __name__ == "__main__":
    main()

