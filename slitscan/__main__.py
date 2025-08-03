import argparse
from .panorama import process_video


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a slit-scan panorama from a video")
    parser.add_argument("video", help="Path to input video")
    parser.add_argument("--output", default="panorama.png", help="Output panorama filename")
    parser.add_argument("--strip-width", type=int, default=2, help="Width of each strip in pixels")
    parser.add_argument(
        "--strip-spacing",
        type=float,
        default=20.0,
        help="Horizontal displacement between strips in pixels",
    )
    args = parser.parse_args()

    result = process_video(
        args.video,
        strip_width=args.strip_width,
        strip_spacing=args.strip_spacing,
        output=args.output,
    )

    print(f"Estimated speed: {result.speed:.2f} px/s, direction: {result.direction}")
    print(f"Panorama saved to {args.output}")


if __name__ == "__main__":
    main()
