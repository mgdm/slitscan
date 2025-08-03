import cv2
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from slitscan import process_video


def create_synthetic_video(path: Path, frames: int = 20, fps: int = 10) -> None:
    width, height = 60, 40
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))

    frame = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.rectangle(frame, (10, 10), (20, 30), (255, 255, 255), -1)
    for _ in range(frames):
        writer.write(frame)
        frame = np.roll(frame, -1, axis=1)  # shift left
    writer.release()


def test_process_video(tmp_path: Path):
    video_file = tmp_path / "input.mp4"
    out_file = tmp_path / "pano.png"
    create_synthetic_video(video_file)

    result = process_video(str(video_file), strip_width=2, strip_spacing=5, output=str(out_file))

    assert result.direction == "left"
    assert result.speed > 0
    assert out_file.exists()
