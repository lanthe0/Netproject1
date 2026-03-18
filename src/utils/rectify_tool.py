from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal, Optional

import cv2
import numpy as np
from ultralytics import YOLO

try:
    from config import GRID_SIZE, RECTIFY_MODEL_PATH
except ImportError:
    try:
        from ..config import GRID_SIZE, RECTIFY_MODEL_PATH
    except ImportError:
        GRID_SIZE = 108
        RECTIFY_MODEL_PATH = "best.pt"

try:
    from .rectify import (
        assign_roles,
        assign_roles_from_four,
        build_quad_from_roles,
        detect_finders,
        pick_3_big_1_small,
        rectify_image,
        rectify_image_cropped,
        rectify_image_for_decoder,
    )
except ImportError:
    from rectify import (
        assign_roles,
        assign_roles_from_four,
        build_quad_from_roles,
        detect_finders,
        pick_3_big_1_small,
        rectify_image,
        rectify_image_cropped,
        rectify_image_for_decoder,
    )


RectifyMode = Literal["cropped", "decoder"]
IMAGE_SIZE = GRID_SIZE * 10


def _resolve_model_path(model_path: str) -> Path:
    raw = Path(model_path)
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent

    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append(Path.cwd() / raw)
        candidates.append(script_dir / raw)
        candidates.append(project_root / raw)
        if raw.name == raw.as_posix():
            candidates.append(project_root / "best.pt")
            candidates.append(script_dir / "best.pt")
            candidates.append(script_dir / "model" / "best.pt")

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return (candidates[-1] if candidates else raw).resolve()


def _normalize_image(image: np.ndarray) -> np.ndarray:
    if image is None or not isinstance(image, np.ndarray):
        raise ValueError("image must be a valid numpy.ndarray")
    if image.ndim not in (2, 3):
        raise ValueError("image must be 2D or 3D")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def _save_image(image: np.ndarray, save_path: Optional[str]) -> None:
    if not save_path:
        return

    out_file = Path(save_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out_file), image)
    if not ok:
        raise RuntimeError(f"Failed to save image: {out_file}")


def _normalize_mode(mode: str) -> RectifyMode:
    normalized = mode.strip().lower()
    if normalized not in ("cropped", "decoder"):
        raise ValueError(f"Unsupported rectify mode: {mode}")
    return normalized


def _detect_roles(
    detections,
    *,
    min_area_ratio: float,
):
    picked = pick_3_big_1_small(detections, min_area_ratio=min_area_ratio)
    if picked is None:
        if len(detections) < 4:
            raise RuntimeError("Pass1 failed: less than 4 detections.")
        top4 = sorted(detections, key=lambda item: item["score"], reverse=True)[:4]
        _, roles = assign_roles_from_four(top4)
        return roles

    big3, small1 = picked
    _, roles = assign_roles(big3, small1)
    return roles


def _build_corner_points(
    image: np.ndarray,
    roles,
    *,
    refine_corners: bool,
    corner_expand_ratio: float,
) -> np.ndarray:
    return build_quad_from_roles(
        image,
        roles,
        refine=refine_corners,
        expand_ratio=corner_expand_ratio,
    )


def _rectify_from_stage(
    image: np.ndarray,
    src_pts: np.ndarray,
    *,
    size: int,
    mode: RectifyMode,
) -> np.ndarray:
    if mode == "cropped":
        return rectify_image_cropped(image, src_pts, out_size=size)
    return rectify_image_for_decoder(image, src_pts, out_size=size)


def _rectify_with_model(
    model: YOLO,
    image: np.ndarray,
    *,
    size: int,
    conf: float,
    iou: float,
    max_det: int,
    min_area_ratio: float,
    center_margin_ratio: float,
    refine_corners: bool,
    corner_expand_ratio: float,
    mode: RectifyMode,
) -> np.ndarray:
    detections = detect_finders(model, image, conf=conf, iou=iou, max_det=max_det)
    roles = _detect_roles(detections, min_area_ratio=min_area_ratio)
    stage1_corner_pts = _build_corner_points(
        image,
        roles,
        refine_corners=refine_corners,
        corner_expand_ratio=corner_expand_ratio,
    )

    rectified_stage1, _ = rectify_image(
        image,
        stage1_corner_pts,
        out_size=size,
        center_margin_ratio=center_margin_ratio,
    )

    pass2_detections = detect_finders(
        model,
        rectified_stage1,
        conf=conf,
        iou=iou,
        max_det=max_det,
    )
    try:
        pass2_roles = _detect_roles(pass2_detections, min_area_ratio=min_area_ratio)
    except RuntimeError:
        pass2_roles = None

    if pass2_roles is not None:
        pass2_corner_pts = _build_corner_points(
            rectified_stage1,
            pass2_roles,
            refine_corners=refine_corners,
            corner_expand_ratio=corner_expand_ratio,
        )
        return _rectify_from_stage(
            rectified_stage1,
            pass2_corner_pts,
            size=size,
            mode=mode,
        )

    return _rectify_from_stage(
        image,
        stage1_corner_pts,
        size=size,
        mode=mode,
    )


class Rectifier:
    def __init__(
        self,
        *,
        model_path: str = RECTIFY_MODEL_PATH,
        size: int = IMAGE_SIZE,
        conf: float = 0.15,
        iou: float = 0.7,
        max_det: int = 20,
        min_area_ratio: float = 1.2,
        center_margin_ratio: float = 0.18,
        refine_corners: bool = True,
        corner_expand_ratio: float = 0.18,
    ) -> None:
        self.model_path = model_path
        self.size = size
        self.conf = conf
        self.iou = iou
        self.max_det = max_det
        self.min_area_ratio = min_area_ratio
        self.center_margin_ratio = center_margin_ratio
        self.refine_corners = refine_corners
        self.corner_expand_ratio = corner_expand_ratio
        self._model: YOLO | None = None
        self._model_file: Path | None = None
        self.last_error: Exception | None = None

    @property
    def model_file(self) -> Path:
        if self._model_file is None:
            self._model_file = _resolve_model_path(self.model_path)
        return self._model_file

    def _get_model(self) -> YOLO:
        if self._model is None:
            model_file = self.model_file
            if not model_file.exists():
                raise FileNotFoundError(f"Model not found: {model_file}")
            self._model = YOLO(str(model_file))
        return self._model

    def _rectify(self, frame: np.ndarray, *, mode: RectifyMode) -> np.ndarray | None:
        try:
            image = _normalize_image(frame)
            rectified = _rectify_with_model(
                self._get_model(),
                image,
                size=self.size,
                conf=self.conf,
                iou=self.iou,
                max_det=self.max_det,
                min_area_ratio=self.min_area_ratio,
                center_margin_ratio=self.center_margin_ratio,
                refine_corners=self.refine_corners,
                corner_expand_ratio=self.corner_expand_ratio,
                mode=mode,
            )
        except Exception as exc:
            self.last_error = exc
            return None

        self.last_error = None
        return rectified

    def rectify_for_decoder_frame(self, frame: np.ndarray) -> np.ndarray | None:
        return self._rectify(frame, mode="decoder")

    def rectify_cropped_frame(self, frame: np.ndarray) -> np.ndarray | None:
        return self._rectify(frame, mode="cropped")

    def rectify_frame(self, frame: np.ndarray) -> np.ndarray | None:
        return self.rectify_for_decoder_frame(frame)


def _get_rectified(
    image: np.ndarray,
    *,
    model_path: str,
    size: int,
    conf: float,
    iou: float,
    max_det: int,
    min_area_ratio: float,
    center_margin_ratio: float,
    refine_corners: bool,
    corner_expand_ratio: float,
    save_path: Optional[str],
    mode: RectifyMode,
) -> np.ndarray:
    normalized = _normalize_image(image)
    model_file = _resolve_model_path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"Model not found: {model_file}")

    model = YOLO(str(model_file))
    rectified = _rectify_with_model(
        model,
        normalized,
        size=size,
        conf=conf,
        iou=iou,
        max_det=max_det,
        min_area_ratio=min_area_ratio,
        center_margin_ratio=center_margin_ratio,
        refine_corners=refine_corners,
        corner_expand_ratio=corner_expand_ratio,
        mode=mode,
    )
    _save_image(rectified, save_path)
    return rectified


def get_rectified_cropped(
    image: np.ndarray,
    model_path: str = RECTIFY_MODEL_PATH,
    size: int = IMAGE_SIZE,
    conf: float = 0.15,
    iou: float = 0.7,
    max_det: int = 20,
    min_area_ratio: float = 1.2,
    center_margin_ratio: float = 0.18,
    refine_corners: bool = True,
    corner_expand_ratio: float = 0.18,
    save_path: Optional[str] = None,
) -> np.ndarray:
    return _get_rectified(
        image,
        model_path=model_path,
        size=size,
        conf=conf,
        iou=iou,
        max_det=max_det,
        min_area_ratio=min_area_ratio,
        center_margin_ratio=center_margin_ratio,
        refine_corners=refine_corners,
        corner_expand_ratio=corner_expand_ratio,
        save_path=save_path,
        mode="cropped",
    )


def get_rectified_for_decoder(
    image: np.ndarray,
    model_path: str = RECTIFY_MODEL_PATH,
    size: int = IMAGE_SIZE,
    conf: float = 0.15,
    iou: float = 0.7,
    max_det: int = 20,
    min_area_ratio: float = 1.2,
    center_margin_ratio: float = 0.18,
    refine_corners: bool = True,
    corner_expand_ratio: float = 0.18,
    save_path: Optional[str] = None,
) -> np.ndarray:
    return _get_rectified(
        image,
        model_path=model_path,
        size=size,
        conf=conf,
        iou=iou,
        max_det=max_det,
        min_area_ratio=min_area_ratio,
        center_margin_ratio=center_margin_ratio,
        refine_corners=refine_corners,
        corner_expand_ratio=corner_expand_ratio,
        save_path=save_path,
        mode="decoder",
    )


def _get_rectified_from_path(
    image_path: str,
    *,
    model_path: str,
    size: int,
    conf: float,
    iou: float,
    max_det: int,
    min_area_ratio: float,
    center_margin_ratio: float,
    refine_corners: bool,
    corner_expand_ratio: float,
    save_path: Optional[str],
    mode: RectifyMode,
) -> np.ndarray:
    image_file = Path(image_path)
    if not image_file.exists():
        raise FileNotFoundError(f"Image not found: {image_file}")

    image = cv2.imread(str(image_file))
    if image is None:
        raise RuntimeError(f"Failed to read image: {image_file}")

    return _get_rectified(
        image,
        model_path=model_path,
        size=size,
        conf=conf,
        iou=iou,
        max_det=max_det,
        min_area_ratio=min_area_ratio,
        center_margin_ratio=center_margin_ratio,
        refine_corners=refine_corners,
        corner_expand_ratio=corner_expand_ratio,
        save_path=save_path,
        mode=mode,
    )


def get_rectified_cropped_from_path(
    image_path: str,
    model_path: str = RECTIFY_MODEL_PATH,
    size: int = IMAGE_SIZE,
    conf: float = 0.15,
    iou: float = 0.7,
    max_det: int = 20,
    min_area_ratio: float = 1.2,
    center_margin_ratio: float = 0.18,
    refine_corners: bool = True,
    corner_expand_ratio: float = 0.18,
    save_path: Optional[str] = None,
) -> np.ndarray:
    return _get_rectified_from_path(
        image_path,
        model_path=model_path,
        size=size,
        conf=conf,
        iou=iou,
        max_det=max_det,
        min_area_ratio=min_area_ratio,
        center_margin_ratio=center_margin_ratio,
        refine_corners=refine_corners,
        corner_expand_ratio=corner_expand_ratio,
        save_path=save_path,
        mode="cropped",
    )


def get_rectified_for_decoder_from_path(
    image_path: str,
    model_path: str = RECTIFY_MODEL_PATH,
    size: int = IMAGE_SIZE,
    conf: float = 0.15,
    iou: float = 0.7,
    max_det: int = 20,
    min_area_ratio: float = 1.2,
    center_margin_ratio: float = 0.18,
    refine_corners: bool = True,
    corner_expand_ratio: float = 0.18,
    save_path: Optional[str] = None,
) -> np.ndarray:
    return _get_rectified_from_path(
        image_path,
        model_path=model_path,
        size=size,
        conf=conf,
        iou=iou,
        max_det=max_det,
        min_area_ratio=min_area_ratio,
        center_margin_ratio=center_margin_ratio,
        refine_corners=refine_corners,
        corner_expand_ratio=corner_expand_ratio,
        save_path=save_path,
        mode="decoder",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Rectify an image with YOLO-first logic.")
    parser.add_argument("image", type=str, help="Input image path")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output image path",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="cropped",
        choices=["cropped", "decoder"],
        help="Output mode",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=RECTIFY_MODEL_PATH,
        help="YOLO model path",
    )
    parser.add_argument("--size", type=int, default=IMAGE_SIZE, help="Rectified output size")
    parser.add_argument("--conf", type=float, default=0.15, help="Detection confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=20, help="Maximum detections")
    parser.add_argument(
        "--min-area-ratio",
        type=float,
        default=1.2,
        help="Minimum ratio between smallest big finder area and small finder area",
    )
    parser.add_argument(
        "--center-margin-ratio",
        type=float,
        default=0.18,
        help="Inset ratio for the stage-one perspective transform",
    )
    parser.add_argument(
        "--corner-expand-ratio",
        type=float,
        default=0.18,
        help="ROI expand ratio when refining corners inside detection boxes",
    )
    args = parser.parse_args()

    mode = _normalize_mode(args.mode)
    output_path = args.output
    if output_path is None:
        suffix = "cropped" if mode == "cropped" else "decoder"
        output_path = f"output/interface_rectified_{suffix}.jpg"

    result = _get_rectified_from_path(
        args.image,
        model_path=args.model_path,
        size=args.size,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        min_area_ratio=args.min_area_ratio,
        center_margin_ratio=args.center_margin_ratio,
        refine_corners=True,
        corner_expand_ratio=args.corner_expand_ratio,
        save_path=output_path,
        mode=mode,
    )
    print(
        "Rectified image saved to "
        f"{Path(output_path).resolve()} with shape={result.shape} mode={mode}"
    )


__all__ = [
    "Rectifier",
    "get_rectified_cropped",
    "get_rectified_cropped_from_path",
    "get_rectified_for_decoder",
    "get_rectified_for_decoder_from_path",
]


if __name__ == "__main__":
    main()
