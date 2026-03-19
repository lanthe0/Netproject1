"""Decoder entrypoint."""

from __future__ import annotations

import argparse
import os
import sys

try:
    from _2Dcode import decode_image
    from config import RECTIFY_MODEL_PATH
    from utils.video_decode import video_to_qr_sequence
except ImportError:
    from ._2Dcode import decode_image
    from .config import RECTIFY_MODEL_PATH
    from .utils.video_decode import video_to_qr_sequence


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode captured video back to binary data")
    parser.add_argument("input_video", help="Input captured video path")
    parser.add_argument("output_bin", help="Decoded output binary path")
    parser.add_argument("output_vbin", help="Decoded validity bitmap path")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable rectification debug output (processed frames / failed frames / yolo visualization)",
    )
    parser.add_argument(
        "--debug-dir",
        default="output/rectify_debug",
        help="Directory used when --debug is enabled",
    )
    args = parser.parse_args()

    input_video_path = args.input_video
    output_bin_path = args.output_bin
    output_vbin_path = args.output_vbin

    if not os.path.exists(input_video_path):
        print(f"Error: input file does not exist: {input_video_path}")
        sys.exit(1)

    print("--- Decoding ---")
    print(f"input video: {input_video_path}")
    print(f"rectify model: {RECTIFY_MODEL_PATH}")
    print(f"output data file: {output_bin_path}")
    print(f"output validity file: {output_vbin_path}")

    try:
        prepared_frames = video_to_qr_sequence(
            input_video_path,
            debug=args.debug,
            debug_dir=args.debug_dir,
        )
    except Exception as exc:
        print(f"Error: failed to preprocess video frames: {exc}")
        sys.exit(1)

    print(f"prepared frames: {len(prepared_frames)}")

    try:
        decode_image(prepared_frames, output_bin_path, output_vbin_path)
        print("decode completed")
    except Exception as exc:
        print(f"Error: decode failed: {exc}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
