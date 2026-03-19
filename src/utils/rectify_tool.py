from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Literal, Optional

import cv2
import numpy as np

os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).resolve().parents[2]))

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
        detect_finders_opencv,
        pick_3_big_1_small,
        select_roles_with_prediction,
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
        detect_finders_opencv,
        pick_3_big_1_small,
        select_roles_with_prediction,
        rectify_image,
        rectify_image_cropped,
        rectify_image_for_decoder,
    )


RectifyMode = Literal["cropped", "decoder"]
IMAGE_SIZE = GRID_SIZE * 10
DECODER_EXPAND_CANDIDATES = (0.0, 0.1, 0.2)
DEFAULT_YOLO_CONF = 0.03
DEFAULT_YOLO_IOU = 0.5
DEFAULT_YOLO_MAX_DET = 10


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
    """根据检测框恢复四个 finder 角色，优先使用三大角补小角策略。

    输入:
    - detections: 检测框列表，格式与 YOLO 输出兼容。
    - min_area_ratio: 大 finder 与小 finder 的最小面积比先验。

    输出:
    - `roles` 字典，包含 `tl/tr/br/bl` 四个角色对应的检测框。

    原理/流程:
    - 先尝试使用三个大 finder 推算右下小 finder 的位置并补位。
    - 若当前检测框满足明显的 `3大1小` 结构，则直接沿用该分组。
    - 只有在补位策略也失败时，才退回旧的 top4 角色分配逻辑。
    """

    if len(detections) >= 3:
        try:
            return select_roles_with_prediction(detections, min_area_ratio=min_area_ratio)
        except RuntimeError:
            pass

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
    expand_modules: float = 0.0,
) -> np.ndarray:
    if mode == "cropped":
        return rectify_image_cropped(image, src_pts, out_size=size)
    return rectify_image_for_decoder(image, src_pts, out_size=size, expand_modules=expand_modules)


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


def _rectify_with_opencv(
    image: np.ndarray,
    *,
    size: int,
    min_area_ratio: float,
    center_margin_ratio: float,
    refine_corners: bool,
    corner_expand_ratio: float,
    mode: RectifyMode,
) -> np.ndarray:
    """使用传统视觉轮廓检测执行兜底矫正。

    输入:
    - image: BGR 输入帧。
    - size: 目标输出尺寸。
    - min_area_ratio: 三大一小 finder 的面积先验。
    - center_margin_ratio: 第一阶段透视变换的内缩比例。
    - refine_corners: 是否在 finder ROI 内细化角点。
    - corner_expand_ratio: 角点细化时的 ROI 扩展比例。
    - mode: 输出模式，`cropped` 或 `decoder`。

    输出:
    - 满足协议几何关系的矫正图像。

    原理/流程:
    - 先用轮廓层级和方形几何筛选 finder 候选。
    - 再复用 YOLO 路径的角色分配和透视变换逻辑。
    - 对第一阶段结果做一次二次检测，尽量减少残余斜切。
    """

    detections = detect_finders_opencv(image)
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

    pass2_detections = detect_finders_opencv(rectified_stage1)
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
            expand_modules=0.0,
        )

    return _rectify_from_stage(
        image,
        stage1_corner_pts,
        size=size,
        mode=mode,
        expand_modules=0.0,
    )


def _rectify_with_model_candidates(
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
) -> list[np.ndarray]:
    """为单帧生成多个 decoder 几何候选，供后续按 CRC 选优。

    输入:
    - model: 已加载的 YOLO 模型。
    - image: 原始 BGR 输入帧。
    - size: 矫正后输出尺寸。
    - conf/iou/max_det: YOLO 检测参数。
    - min_area_ratio: 三大一小 finder 面积先验。
    - center_margin_ratio: 第一阶段透视矫正的内缩比例。
    - refine_corners: 是否细化 finder 角点。
    - corner_expand_ratio: 角点细化 ROI 扩展比例。

    输出:
    - 同一帧对应的多个 decoder 候选图像列表。

    原理/流程:
    - 先按主路径完成 YOLO 两阶段 finder 定位。
    - 复用同一套第二阶段角点。
    - 仅改变 decoder 目标点的轻微外扩量，生成少量候选几何。
    """

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
        base_image = rectified_stage1
        base_pts = pass2_corner_pts
    else:
        base_image = image
        base_pts = stage1_corner_pts

    return [
        _rectify_from_stage(
            base_image,
            base_pts,
            size=size,
            mode="decoder",
            expand_modules=expand_modules,
        )
        for expand_modules in DECODER_EXPAND_CANDIDATES
    ]


class Rectifier:
    def __init__(
        self,
        *,
        model_path: str = RECTIFY_MODEL_PATH,
        size: int = IMAGE_SIZE,
        conf: float = DEFAULT_YOLO_CONF,
        iou: float = DEFAULT_YOLO_IOU,
        max_det: int = DEFAULT_YOLO_MAX_DET,
        min_area_ratio: float = 1.2,
        center_margin_ratio: float = 0.18,
        refine_corners: bool = True,
        corner_expand_ratio: float = 0.18,
        enable_opencv_fallback: bool = False,
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
        self.enable_opencv_fallback = enable_opencv_fallback
        self._model: YOLO | None = None
        self._model_file: Path | None = None
        self.last_error: Exception | None = None
        self.last_method: str | None = None

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
        """对单帧执行矫正，默认仅使用 YOLO 路径。

        输入:
        - frame: 视频中的原始图像帧。
        - mode: 输出模式，供解码或裁切显示使用。

        输出:
        - 矫正成功时返回图像，失败时返回 `None`。

        原理/流程:
        - 先将输入统一归一化成 BGR 图像。
        - 优先走 YOLO finder 检测与二阶段透视矫正。
        - 仅当 `enable_opencv_fallback=True` 时，才尝试轮廓检测兜底。
        - 记录最后一次成功的方法与最近一次错误，便于调试。
        """

        image = None
        yolo_error: Exception | None = None
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
            self.last_error = None
            self.last_method = "yolo"
            return rectified
        except Exception as exc:
            yolo_error = exc

        if not self.enable_opencv_fallback:
            self.last_error = yolo_error
            self.last_method = None
            return None

        try:
            if image is None:
                image = _normalize_image(frame)
            rectified = _rectify_with_opencv(
                image,
                size=self.size,
                min_area_ratio=self.min_area_ratio,
                center_margin_ratio=self.center_margin_ratio,
                refine_corners=self.refine_corners,
                corner_expand_ratio=self.corner_expand_ratio,
                mode=mode,
            )
            self.last_error = yolo_error
            self.last_method = "opencv"
            return rectified
        except Exception as fallback_exc:
            if yolo_error is None:
                self.last_error = fallback_exc
            else:
                self.last_error = RuntimeError(f"yolo failed: {yolo_error}; opencv failed: {fallback_exc}")
            self.last_method = None
            return None

    def rectify_for_decoder_frame(self, frame: np.ndarray) -> np.ndarray | None:
        return self._rectify(frame, mode="decoder")

    def rectify_for_decoder_candidates(self, frame: np.ndarray) -> list[np.ndarray]:
        """生成单帧的多个 decoder 候选图像。

        输入:
        - frame: 视频中的原始图像帧。

        输出:
        - 候选 decoder 图像列表；若主路径失败则返回空列表。

        原理/流程:
        - 复用 YOLO 两阶段检测结果。
        - 基于不同的 decoder 目标边界外扩量生成少量候选。
        - 供上层把同一帧的多个 grid 一并交给解码器，用 CRC 自动筛选。
        """

        try:
            image = _normalize_image(frame)
            candidates = _rectify_with_model_candidates(
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
            )
        except Exception as exc:
            self.last_error = exc
            self.last_method = None
            return []

        self.last_error = None
        self.last_method = "yolo"
        return candidates

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
    conf: float = DEFAULT_YOLO_CONF,
    iou: float = DEFAULT_YOLO_IOU,
    max_det: int = DEFAULT_YOLO_MAX_DET,
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
    conf: float = DEFAULT_YOLO_CONF,
    iou: float = DEFAULT_YOLO_IOU,
    max_det: int = DEFAULT_YOLO_MAX_DET,
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
    conf: float = DEFAULT_YOLO_CONF,
    iou: float = DEFAULT_YOLO_IOU,
    max_det: int = DEFAULT_YOLO_MAX_DET,
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
    conf: float = DEFAULT_YOLO_CONF,
    iou: float = DEFAULT_YOLO_IOU,
    max_det: int = DEFAULT_YOLO_MAX_DET,
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
    parser.add_argument("--conf", type=float, default=DEFAULT_YOLO_CONF, help="Detection confidence threshold")
    parser.add_argument("--iou", type=float, default=DEFAULT_YOLO_IOU, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=DEFAULT_YOLO_MAX_DET, help="Maximum detections")
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
