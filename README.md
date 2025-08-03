# Slitscan Panorama Builder

This project processes a side-window video to estimate motion and build a panoramic image.

## Usage

```bash
python -m slitscan <video path> --output panorama.png
```

The script estimates the average horizontal speed and direction of the scene using optical flow. It extracts vertical strips at regular distances and stitches them into a panorama image.

Dependencies are listed in `requirements.txt`.
