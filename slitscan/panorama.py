"""Feature-based panorama stitching pipeline.

This module replaces the original slitscan approach with a more robust
feature-based pipeline.  Keypoints and descriptors are detected for each
video frame using ORB and matched against the previous frame.  RANSAC based
homography estimation recovers the inter-frame transformation while masking
out regions with moving foreground objects.  A cumulative transform warps
each frame into a growing canvas and simple feathering blends overlapping
regions.  Frames are first projected onto a cylinder so that panoramas with a
wide field of view are less distorted.
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class MotionResult:
    """Estimated motion parameters returned by :func:`process_video`."""

    speed: float  # pixels per second
    direction: str  # "left" or "right"


def _cylindrical_project(img: np.ndarray, f: float) -> np.ndarray:
    """Project ``img`` onto a cylinder with focal length ``f``.

    Parameters
    ----------
    img:
        Input image.
    f:
        Focal length in pixels.
    """

    h, w = img.shape[:2]
    y_i, x_i = np.indices((h, w))
    x = (x_i - w / 2.0) / f
    y = (y_i - h / 2.0) / f

    X = np.sin(x)
    Y = y
    Z = np.cos(x)

    map_x = f * X / Z + w / 2.0
    map_y = f * Y / Z + h / 2.0

    return cv2.remap(
        img,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )


def process_video(
    video_path: str,
    strip_width: int = 2,  # kept for API compatibility, not used
    strip_spacing: float = 20.0,  # kept for API compatibility, not used
    output: str = "panorama.png",
) -> MotionResult:
    """Process ``video_path`` and create a panorama image.

    Frames are stitched using ORB features and homographies.  The final
    panorama is written to ``output`` and a :class:`MotionResult` is returned
    describing the average camera motion.
    """

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    ret, frame = cap.read()
    if not ret:
        raise ValueError("Empty video")

    height, width = frame.shape[:2]
    focal = width / np.pi  # simple approximation

    frame = _cylindrical_project(frame, focal)
    gray_prev = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    def detect_and_compute(gray: np.ndarray):
        pts = cv2.goodFeaturesToTrack(gray, maxCorners=1000, qualityLevel=0.01, minDistance=3)
        if pts is None:
            return [], None
        kp = [cv2.KeyPoint(float(x), float(y), 31) for x, y in pts.reshape(-1, 2)]
        kp, desc = sift.compute(gray, kp)
        coords = [k.pt for k in kp]
        return coords, desc

    kp_prev, desc_prev = detect_and_compute(gray_prev)

    canvas_width = width * (total_frames if total_frames > 0 else 100)
    canvas_height = height
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.float32)
    mask_canvas = np.zeros((canvas_height, canvas_width), dtype=np.float32)
    offset = canvas_width // 2 - width // 2

    H_cumulative = np.eye(3)
    T_offset = np.array([[1.0, 0.0, offset], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    # place first frame
    warped = cv2.warpPerspective(frame, T_offset @ H_cumulative, (canvas_width, canvas_height))
    canvas = warped.astype(np.float32)
    mask = cv2.warpPerspective(
        np.ones((height, width), dtype=np.float32),
        T_offset @ H_cumulative,
        (canvas_width, canvas_height),
    )
    mask_canvas = mask

    flow_x_total = 0.0
    frame_count = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = _cylindrical_project(frame, focal)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        kp, desc = detect_and_compute(gray)
        if desc_prev is None or desc is None or len(kp_prev) < 4 or len(kp) < 4:
            kp_prev, desc_prev, gray_prev = kp, desc, gray
            continue

        matches = matcher.match(desc_prev, desc)
        # simple foreground masking using frame difference
        diff = cv2.absdiff(gray_prev, gray)
        _, fg_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        bg_mask = cv2.bitwise_not(fg_mask)

        good = []
        for m in matches:
            x_prev, y_prev = kp_prev[m.queryIdx]
            if bg_mask[int(y_prev), int(x_prev)] > 0:
                good.append(m)

        if len(good) < 4:
            kp_prev, desc_prev, gray_prev = kp, desc, gray
            continue

        src_pts = np.float32([kp_prev[m.queryIdx] for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp[m.trainIdx] for m in good]).reshape(-1, 1, 2)

        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is None:
            kp_prev, desc_prev, gray_prev = kp, desc, gray
            continue

        flow_x_total += float(H[0, 2])
        frame_count += 1

        # transform current frame into coordinates of the first frame
        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            kp_prev, desc_prev, gray_prev = kp, desc, gray
            continue
        H_cumulative = H_cumulative @ H_inv
        warp_mat = T_offset @ H_cumulative

        warped = cv2.warpPerspective(frame, warp_mat, (canvas_width, canvas_height))
        mask = cv2.warpPerspective(
            np.ones((height, width), dtype=np.float32),
            warp_mat,
            (canvas_width, canvas_height),
        )
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        alpha = mask[..., None] / 255.0
        canvas = canvas * (1.0 - alpha) + warped.astype(np.float32) * alpha
        mask_canvas = np.maximum(mask_canvas, mask)

        kp_prev, desc_prev, gray_prev = kp, desc, gray

    cap.release()

    ys, xs = np.nonzero(mask_canvas)
    if len(xs) and len(ys):
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        panorama = canvas[y_min : y_max + 1, x_min : x_max + 1]
    else:
        panorama = canvas

    cv2.imwrite(output, panorama.astype(np.uint8))

    avg_flow_x = flow_x_total / frame_count if frame_count else 0.0
    speed = abs(avg_flow_x) * fps
    direction = "left" if avg_flow_x > 0 else "right"
    return MotionResult(speed=speed, direction=direction)


__all__ = ["process_video", "MotionResult"]

