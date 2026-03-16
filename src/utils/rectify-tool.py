from pathlib import Path
from typing import Optional
import sys

import cv2
import numpy as np
from ultralytics import YOLO

IMAGE_SIZE = 1080

from rectify import (
    assign_roles,
    assign_roles_from_four,
    build_quad_from_roles,
    detect_finders,
    pick_3_big_1_small,
    rectify_image,
    rectify_image_cropped,
)


def _resolve_model_path(model_path: str) -> Path:
    """Resolve model path robustly across common launch locations."""
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
            # If only file name is given, also try these common locations.
            candidates.append(project_root / "best.pt")
            candidates.append(script_dir / "best.pt")
            candidates.append(script_dir / "model" / "best.pt")

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return (candidates[-1] if candidates else raw).resolve()


def get_rectified_cropped(
    image: np.ndarray,
    model_path: str = "best.pt",
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
    """
    传入图像使用yolo检测+二次矫正策略返回矫正截取图像（numpy.ndarray）。

    策略摘要:
    1) 第一轮检测 finder；优先 3大+1小，失败则用4点几何兜底。
    2) 基于角色框在 ROI 内细化四角点，完成第一轮透视矫正。
    3) 第二轮在矫正图上再次检测并细化角点。
    4) 输出“基于细化角点的截取矫正图”（cropped）。
    """
    if image is None or not isinstance(image, np.ndarray):
        raise ValueError("image must be a valid numpy.ndarray")
    if image.ndim not in (2, 3):
        raise ValueError("image must be 2D (grayscale) or 3D (color)")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    model_file = _resolve_model_path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"Model not found: {model_file}")

    model = YOLO(str(model_file))

    # Pass 1 detect and role assignment.
    detections = detect_finders(model, image, conf=conf, iou=iou, max_det=max_det)
    picked = pick_3_big_1_small(detections, min_area_ratio=min_area_ratio)
    if picked is None:
        if len(detections) < 4:
            raise RuntimeError("Pass1 failed: less than 4 detections.")
        top4 = sorted(detections, key=lambda d: d["score"], reverse=True)[:4]
        _, roles = assign_roles_from_four(top4)
    else:
        big3, small1 = picked
        _, roles = assign_roles(big3, small1)

    stage1_corner_pts = build_quad_from_roles(
        image,
        roles,
        refine=refine_corners,
        expand_ratio=corner_expand_ratio,
    )

    rectified_stage1, _ = rectify_image(
        image,
        stage1_corner_pts,
        out_size=size,
        center_margin_ratio=center_margin_ratio,
    )

    # Pass 2 detect and corner refinement, then cropped rectification.
    pass2_detections = detect_finders(model, rectified_stage1, conf=conf, iou=iou, max_det=max_det)
    pass2_picked = pick_3_big_1_small(pass2_detections, min_area_ratio=min_area_ratio)

    if pass2_picked is not None:
        pass2_big3, pass2_small1 = pass2_picked
        _, pass2_roles = assign_roles(pass2_big3, pass2_small1)
        pass2_corner_pts = build_quad_from_roles(
            rectified_stage1,
            pass2_roles,
            refine=refine_corners,
            expand_ratio=corner_expand_ratio,
        )
        cropped = rectify_image_cropped(rectified_stage1, pass2_corner_pts, out_size=size)
    elif len(pass2_detections) >= 4:
        pass2_top4 = sorted(pass2_detections, key=lambda d: d["score"], reverse=True)[:4]
        _, pass2_roles = assign_roles_from_four(pass2_top4)
        pass2_corner_pts = build_quad_from_roles(
            rectified_stage1,
            pass2_roles,
            refine=refine_corners,
            expand_ratio=corner_expand_ratio,
        )
        cropped = rectify_image_cropped(rectified_stage1, pass2_corner_pts, out_size=size)
    else:
        # Fallback: return pass1 cropped result.
        cropped = rectify_image_cropped(image, stage1_corner_pts, out_size=size)

    if save_path:
        out_file = Path(save_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(out_file), cropped)
        if not ok:
            raise RuntimeError(f"Failed to save image: {out_file}")

    return cropped


def get_rectified_cropped_from_path(
    image_path: str,
    model_path: str = "best.pt",
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
    image_file = Path(image_path)
    if not image_file.exists():
        raise FileNotFoundError(f"Image not found: {image_file}")

    image = cv2.imread(str(image_file))
    if image is None:
        raise RuntimeError(f"Failed to read image: {image_file}")

    return get_rectified_cropped(
        image=image,
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
    )


if __name__ == "__main__":
    # 示例：从路径读取后传入图片对象。
    INPUT_IMAGE = "./input/test3.jpg"
    OUTPUT_IMAGE = "./output/interface_rectified_cropped.jpg"
    img = cv2.imread(INPUT_IMAGE)
    if img is None:
        raise RuntimeError(f"Failed to read demo image: {INPUT_IMAGE}")

    result = get_rectified_cropped(
        image=img,
        size=IMAGE_SIZE,
        conf=0.15,
        save_path=OUTPUT_IMAGE,
    )
    print(f"Interface done, result shape={result.shape}, saved to {OUTPUT_IMAGE}")
