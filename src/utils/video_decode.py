"""Video preprocessing utilities for the decoder."""

from __future__ import annotations

import cv2
import numpy as np

try:
    from config import RECTIFY_MODEL_PATH
    from _2Dcode import BASE_GRID
    from config import BIG_FINDER_SIZE, GRID_SIZE, QUIET_WIDTH, SMALL_FINDER_SIZE
    from utils.rectify_tool import Rectifier
except ImportError:
    from ..config import RECTIFY_MODEL_PATH
    from .._2Dcode import BASE_GRID
    from ..config import BIG_FINDER_SIZE, GRID_SIZE, QUIET_WIDTH, SMALL_FINDER_SIZE
    from .rectify_tool import Rectifier


def extract_video_frames(path: str) -> list[np.ndarray]:
    """Read all frames from a video file."""

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"failed to open video: {path}")

    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)

    cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from video: {path}")
    return frames


def _to_grid(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)

    if arr.shape[0] % GRID_SIZE != 0 or arr.shape[1] % GRID_SIZE != 0:
        arr = cv2.resize(arr, (GRID_SIZE, GRID_SIZE), interpolation=cv2.INTER_NEAREST)
        return (arr < 128).astype(np.uint8)

    module = arr.shape[0] // GRID_SIZE
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            block = arr[
                row * module : (row + 1) * module,
                col * module : (col + 1) * module,
            ]
            grid[row, col] = 1 if np.mean(block) < 128 else 0
    return grid


def _finder_mask() -> np.ndarray:
    mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    q = QUIET_WIDTH
    big = BIG_FINDER_SIZE
    small = SMALL_FINDER_SIZE

    mask[q : q + big, q : q + big] = True
    mask[q : q + big, GRID_SIZE - q - big : GRID_SIZE - q] = True
    mask[GRID_SIZE - q - big : GRID_SIZE - q, q : q + big] = True
    mask[GRID_SIZE - q - small : GRID_SIZE - q, GRID_SIZE - q - small : GRID_SIZE - q] = True
    return mask


FINDER_MASK = _finder_mask()


def _finder_match_score(frame: np.ndarray) -> float:
    grid = _to_grid(frame)
    return float((grid[FINDER_MASK] == BASE_GRID[FINDER_MASK]).mean())


def preprocess_frames_for_decoder(
    frames: list[np.ndarray],
    *,
    rectifier: Rectifier | None = None,
) -> list[np.ndarray]:
    """Rectify frames for decode, choosing the higher-scoring geometry per frame."""

    worker = rectifier or Rectifier(model_path=RECTIFY_MODEL_PATH)
    processed_frames: list[np.ndarray] = []
    rectified_wins = 0
    raw_wins = 0
    rectify_failures = 0
    first_failure_reason: str | None = None

    for idx, frame in enumerate(frames):
        raw_score = _finder_match_score(frame)
        rectified = worker.rectify_for_decoder_frame(frame)

        if rectified is None:
            processed_frames.append(frame)
            raw_wins += 1
            rectify_failures += 1
            if first_failure_reason is None:
                first_failure_reason = (
                    str(worker.last_error) if worker.last_error is not None else "unknown rectify failure"
                )
        else:
            rectified_score = _finder_match_score(rectified)
            if rectified_score > raw_score:
                processed_frames.append(rectified)
                rectified_wins += 1
            else:
                processed_frames.append(frame)
                raw_wins += 1

        if (idx + 1) % 20 == 0 or idx + 1 == len(frames):
            print(
                "rectify progress: "
                f"{idx + 1}/{len(frames)} frames, "
                f"rectified_wins={rectified_wins}, "
                f"raw_wins={raw_wins}, "
                f"rectify_failures={rectify_failures}"
            )

    print(f"rectify model: {worker.model_file}")
    print(
        "rectify summary: "
        f"rectified_wins={rectified_wins}, "
        f"raw_wins={raw_wins}, "
        f"rectify_failures={rectify_failures}"
    )
    if first_failure_reason is not None:
        print(f"first rectify failure: {first_failure_reason}")
    return processed_frames


def video_to_qr_sequence(path: str) -> list[np.ndarray]:
    """Convert a video into a frame sequence suitable for `_2Dcode.decode_image`."""

    frames = extract_video_frames(path)
    return preprocess_frames_for_decoder(frames)
