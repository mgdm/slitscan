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
        Width of each vertical strip in pixels. Retained for backward
        compatibility; the current implementation does not use it.
    strip_spacing:
        Required horizontal displacement (in pixels) between strips.
        Retained for backward compatibility and ignored.
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

    # start with a modest canvas (3x the frame size) centred on the first frame
    canvas_height = height * 3
    canvas_width = width * 3
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    offset_x = (canvas_width - width) // 2
    offset_y = (canvas_height - height) // 2
    canvas[offset_y : offset_y + height, offset_x : offset_x + width] = prev_frame

    # track observed bounds of all warped frames
    obs_min_x = offset_x
    obs_min_y = offset_y
    obs_max_x = offset_x + width
    obs_max_y = offset_y + height

    cum_dx = 0.0
    cum_dy = 0.0
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
        dy = float(M[1, 2])
        cum_dx += dx
        cum_dy += dy
        flow_x_total += dx
        frame_count += 1

        # project frame corners into canvas coordinates
        corners = np.array(
            [[0, 0], [width, 0], [0, height], [width, height]], dtype=np.float32
        )
        trans_corners = corners + np.array([cum_dx + offset_x, cum_dy + offset_y])
        min_x, min_y = trans_corners.min(axis=0)
        max_x, max_y = trans_corners.max(axis=0)

        # expand canvas if projected corners exceed current bounds
        pad_left = int(max(0, -min_x))
        pad_top = int(max(0, -min_y))
        pad_right = int(max(0, max_x - canvas_width))
        pad_bottom = int(max(0, max_y - canvas_height))
        if pad_left or pad_top or pad_right or pad_bottom:
            canvas = cv2.copyMakeBorder(
                canvas,
                pad_top,
                pad_bottom,
                pad_left,
                pad_right,
                borderType=cv2.BORDER_CONSTANT,
            )
            canvas_height, canvas_width = canvas.shape[:2]
            offset_x += pad_left
            offset_y += pad_top
            min_x += pad_left
            max_x += pad_left
            min_y += pad_top
            max_y += pad_top

        # warp the frame into the canvas coordinate system
        M_total = np.array(
            [[1, 0, cum_dx + offset_x], [0, 1, cum_dy + offset_y]], dtype=np.float32
        )
        warped = cv2.warpAffine(frame, M_total, (canvas_width, canvas_height))
        mask = np.any(warped != 0, axis=2)
        canvas[mask] = warped[mask]

        obs_min_x = min(obs_min_x, min_x)
        obs_min_y = min(obs_min_y, min_y)
        obs_max_x = max(obs_max_x, max_x)
        obs_max_y = max(obs_max_y, max_y)

        prev_gray = gray

    cap.release()

    # crop to the area that actually received image data
    cropped = canvas[int(obs_min_y) : int(obs_max_y), int(obs_min_x) : int(obs_max_x)]
    cv2.imwrite(output, cropped)

    avg_flow_x = flow_x_total / frame_count if frame_count else 0.0
    speed = abs(avg_flow_x) * fps
    direction = "right" if avg_flow_x > 0 else "left"
    return MotionResult(speed=speed, direction=direction)


__all__ = ["process_video", "MotionResult"]
