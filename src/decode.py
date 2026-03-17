"""Decoder entrypoint."""

from __future__ import annotations

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
    if len(sys.argv) != 4:
        print("Usage: python decode.py <input_video> <output_bin> <output_vbin>")
        print("Example: python decode.py output.mp4 output/decoded.bin output/vout.bin")
        sys.exit(1)

    input_video_path = sys.argv[1]
    output_bin_path = sys.argv[2]
    output_vbin_path = sys.argv[3]

    if not os.path.exists(input_video_path):
        print(f"Error: input file does not exist: {input_video_path}")
        sys.exit(1)

    print("--- Decoding ---")
    print(f"input video: {input_video_path}")
    print(f"rectify model: {RECTIFY_MODEL_PATH}")
    print(f"output data file: {output_bin_path}")
    print(f"output validity file: {output_vbin_path}")

    try:
        prepared_frames = video_to_qr_sequence(input_video_path)
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
