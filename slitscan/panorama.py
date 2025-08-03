"""Panorama construction using optical flow and robust motion estimation.

The module estimates camera motion between consecutive frames using
``cv2.calcOpticalFlowFarneback`` followed by a RANSAC-based affine transform
fit.  RANSAC is employed to reject outlier flow vectors, such as those
originating from independently moving objects, providing a more stable
translation estimate for panorama generation.
"""

import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class MotionResult:
    """Estimated motion parameters."""
    speed: float  # pixels per second
    direction: str  # 'left' or 'right'


def process_video(
    video_path: str,
    strip_width: int = 2,
    strip_spacing: float = 20.0,
    output: str = "panorama.png",
) -> MotionResult:
    """Process ``video_path`` to build a panorama image.

    Parameters
    ----------
    video_path:
        Path to input video file.
    strip_width:
        Width of each vertical strip in pixels.
    strip_spacing:
        Required horizontal displacement (in pixels) between strips.
    output:
        Filename where the panorama image will be written.

    Returns
    -------
    MotionResult
        Estimated speed (pixels/second) and direction.
    """

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    ret, prev_frame = cap.read()
    if not ret:
        raise ValueError("Empty video")

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    height, width = prev_frame.shape[:2]

    strips = []
    distance_acc = 0.0
    flow_x_total = 0.0
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )

        h, w = prev_gray.shape
        y, x = np.mgrid[0:h, 0:w]
        prev_pts = np.stack((x, y), axis=-1).reshape(-1, 2).astype(np.float32)
        next_pts = prev_pts + flow.reshape(-1, 2)
        M, _ = cv2.estimateAffinePartial2D(prev_pts, next_pts, method=cv2.RANSAC)
        if M is None:
            prev_gray = gray
            continue

        dx = float(M[0, 2])
        distance_acc += abs(dx)
        flow_x_total += dx
        frame_count += 1
        print(f"Processed frame {frame_count}")

        if distance_acc >= strip_spacing:
            x = width // 2
            strip = frame[:, x : x + strip_width]
            strips.append(strip)
            distance_acc = 0.0

        prev_gray = gray

    cap.release()

    if strips:
        panorama = np.hstack(strips)
        cv2.imwrite(output, panorama)

    avg_flow_x = flow_x_total / frame_count if frame_count else 0.0
    speed = abs(avg_flow_x) * fps
    direction = "right" if avg_flow_x > 0 else "left"
    return MotionResult(speed=speed, direction=direction)


__all__ = ["process_video", "MotionResult"]
