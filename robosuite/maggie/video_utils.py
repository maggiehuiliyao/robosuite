"""
Video recording utilities for Robosuite evaluation.
"""

import os
import numpy as np


def save_episode_video(frames, episode_idx, output_dir, fps=30, prefix="episode"):
    """
    Save frames as MP4 video.

    Args:
        frames: List of numpy arrays (H, W, 3) in RGB format
        episode_idx: Episode number
        output_dir: Directory to save video
        fps: Frames per second
        prefix: Filename prefix

    Returns:
        video_path: Path to saved video
    """
    try:
        import imageio
    except ImportError:
        print("Warning: imageio not installed. Cannot save video.")
        return None

    os.makedirs(output_dir, exist_ok=True)
    video_path = os.path.join(output_dir, f"{prefix}_{episode_idx:03d}.mp4")

    # Convert frames to uint8 if needed
    frames_uint8 = []
    for frame in frames:
        if frame.dtype != np.uint8:
            # Assume frame is in [0, 255] range but float
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        frames_uint8.append(frame)

    imageio.mimsave(video_path, frames_uint8, fps=fps)
    print(f"Saved video: {video_path}")

    return video_path


def save_multi_camera_video(frame_dict, episode_idx, output_dir, fps=30, prefix="episode"):
    """
    Save multi-camera video with side-by-side views.

    Args:
        frame_dict: Dict mapping camera names to list of frames
                   e.g., {'agentview': [frame1, frame2, ...], 'frontview': [...]}
        episode_idx: Episode number
        output_dir: Directory to save video
        fps: Frames per second
        prefix: Filename prefix

    Returns:
        video_path: Path to saved video
    """
    # Get camera names and verify all have same number of frames
    camera_names = sorted(frame_dict.keys())
    num_frames = len(frame_dict[camera_names[0]])

    for cam_name in camera_names:
        assert len(frame_dict[cam_name]) == num_frames, \
            f"Camera {cam_name} has {len(frame_dict[cam_name])} frames, expected {num_frames}"

    # Concatenate camera views horizontally
    combined_frames = []
    for i in range(num_frames):
        # Get frames from all cameras at this timestep
        frames_at_t = [frame_dict[cam_name][i] for cam_name in camera_names]

        # Ensure all frames are uint8
        frames_at_t = [
            np.clip(f, 0, 255).astype(np.uint8) if f.dtype != np.uint8 else f
            for f in frames_at_t
        ]

        # Concatenate horizontally
        combined = np.hstack(frames_at_t)
        combined_frames.append(combined)

    return save_episode_video(combined_frames, episode_idx, output_dir, fps, prefix)


def add_text_to_frame(frame, text, position=(10, 30), font_scale=0.7, color=(255, 255, 255), thickness=2):
    """
    Add text overlay to a frame using OpenCV.

    Args:
        frame: Numpy array (H, W, 3)
        text: Text to add
        position: (x, y) position for text
        font_scale: Font size
        color: RGB color tuple
        thickness: Text thickness

    Returns:
        frame: Frame with text overlay
    """
    try:
        import cv2
    except ImportError:
        print("Warning: OpenCV not installed. Cannot add text to frame.")
        return frame

    frame = frame.copy()
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA
    )
    return frame
