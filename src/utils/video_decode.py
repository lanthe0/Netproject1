"""解码端的视频预处理工具。

当前策略:
- 手机实拍帧必须先经过矫正，原始帧不再作为可解码候选。
- 视频按流式逐帧读取，避免整段视频一次性进内存。
- 默认只使用 YOLO 矫正链路；OpenCV 兜底仅保留作实验能力。
- 矫正成功后立即压成协议网格，降低内存占用。
"""

from __future__ import annotations

import binascii
import cv2
import numpy as np
from pathlib import Path
import reedsolo
from typing import Iterator

try:
    from config import RECTIFY_MODEL_PATH
    from _2Dcode import BASE_GRID, DATA_ITER, INFO_ITER, bytes_to_bits
    from config import BIG_FINDER_SIZE, GRID_SIZE, QUIET_WIDTH, SMALL_FINDER_SIZE
    from utils.rectify_tool import COMPLEX_DECODER_EXPAND_CANDIDATES, Rectifier
    from utils.rectify import draw_detection_only, detect_finders
    from config import ECC_BYTES
except ImportError:
    from ..config import RECTIFY_MODEL_PATH
    from .._2Dcode import BASE_GRID, DATA_ITER, INFO_ITER, bytes_to_bits
    from ..config import BIG_FINDER_SIZE, GRID_SIZE, QUIET_WIDTH, SMALL_FINDER_SIZE
    from .rectify_tool import COMPLEX_DECODER_EXPAND_CANDIDATES, Rectifier
    from .rectify import draw_detection_only, detect_finders
    from ..config import ECC_BYTES


def iter_video_frames(path: str, *, max_frames: int = 0) -> Iterator[np.ndarray]:
    """从磁盘按顺序逐帧读取视频。

    输入:
    - path: 视频文件路径。
    - max_frames: 最多读取多少帧，`0` 表示读取全部。

    输出:
    - 逐帧产出的 OpenCV 图像迭代器，顺序与视频一致。

    原理/流程:
    - 用 `cv2.VideoCapture` 打开视频。
    - 每次只读取一帧并立即产出。
    - 迭代结束后统一释放句柄。
    """

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"failed to open video: {path}")

    emitted = 0
    try:
        while max_frames <= 0 or emitted < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            emitted += 1
            yield frame
    finally:
        cap.release()


def extract_video_frames(path: str, *, max_frames: int = 0) -> list[np.ndarray]:
    """为旧脚本兼容性保留的全量取帧接口。

    输入:
    - path: 视频文件路径。
    - max_frames: 最多读取多少帧。

    输出:
    - 原始帧列表。

    原理/流程:
    - 复用流式读帧接口。
    - 仅在仍然需要列表的旧调试脚本中物化到内存。
    """

    frames = list(iter_video_frames(path, max_frames=max_frames))
    if not frames:
        raise RuntimeError(f"no frames decoded from video: {path}")
    return frames


def _to_grid(frame: np.ndarray) -> np.ndarray:
    """将矫正后的图像采样成协议使用的二值网格。

    输入:
    - frame: 已矫正图像，或已经接近网格尺寸的灰度/二值图。

    输出:
    - `GRID_SIZE x GRID_SIZE` 的二值矩阵，其中 `1` 表示黑色模块。

    原理/流程:
    - 先统一转为灰度图。
    - 若图像不是整倍数模块尺寸，则先缩放到网格大小再阈值化。
    - 若是整倍数尺寸，则对每个模块块做均值采样。
    - 当前实拍样本上，块均值方案比中心窗口投票更稳定。
    """

    arr = np.asarray(frame)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)

    if arr.shape == (GRID_SIZE, GRID_SIZE):
        if arr.max() <= 1:
            return (arr > 0).astype(np.uint8)
        return (arr < 128).astype(np.uint8)

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
    """生成只覆盖 finder 区域的掩码。

    输入:
    - 无，直接使用协议常量。

    输出:
    - 布尔矩阵，只在三大一小 finder 区域上为真。

    原理/流程:
    - 按协议中的固定 finder 位置进行标记。
    - 后续只用这些区域衡量矫正结果是否接近理想模板。
    """

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
RS_CODEC = reedsolo.RSCodec(ECC_BYTES)


def _finder_match_score(frame: np.ndarray) -> float:
    """计算候选图像在 finder 区域上的模板匹配分数。

    输入:
    - frame: 矫正后的候选图像，或已经采样成网格的候选。

    输出:
    - `[0, 1]` 之间的匹配率，越大表示 finder 区域越像标准模板。

    原理/流程:
    - 先将输入转成协议网格。
    - 只比较 finder 对应的固定区域，避免 payload 区域干扰评分。
    """

    grid = _to_grid(frame)
    return float((grid[FINDER_MASK] == BASE_GRID[FINDER_MASK]).mean())


def _bits_to_int(bits: np.ndarray) -> int:
    """将 bit 序列按大端顺序转换为整数。

    输入:
    - bits: 一维 bit 数组。

    输出:
    - 还原后的无符号整数。

    原理/流程:
    - 按位左移累加，将 bit 序列解释成整数值。
    """

    value = 0
    for bit in bits.astype(np.uint8):
        value = (value << 1) | int(bit)
    return value


def _evaluate_grid_candidate(grid: np.ndarray) -> dict[str, object]:
    """快速评估单个候选网格的头部与 RS/CRC 质量。

    输入:
    - grid: 已经采样完成的协议二值网格。

    输出:
    - 包含 `bit_len`、`frame_idx`、`rs_ok`、`crc_ok` 等字段的评估字典。

    原理/流程:
    - 先读取头部里的 `CRC + bit_len + frame_idx`。
    - 按头部声明长度截取数据区，尝试执行 RS 解码。
    - 若 RS 成功，再继续做 CRC 校验，为候选排序提供依据。
    """

    header_bits = np.array([grid[r, c] for (r, c) in INFO_ITER], dtype=np.uint8)
    expected_crc = _bits_to_int(header_bits[:32])
    bit_len = _bits_to_int(header_bits[32:48])
    frame_idx = _bits_to_int(header_bits[48:64])
    byte_len = (bit_len + 7) // 8
    encoded_len = len(RS_CODEC.encode(b"\x00" * byte_len)) if byte_len > 0 else 0
    encoded_bits = min(encoded_len * 8, len(DATA_ITER))
    payload_bits = np.array([grid[r, c] for (r, c) in DATA_ITER[:encoded_bits]], dtype=np.uint8)

    rs_ok = False
    crc_ok = False
    try:
        corrected_raw_bytes, _, _ = RS_CODEC.decode(np.packbits(payload_bits).tobytes())
        corrected_bytes = bytes(corrected_raw_bytes)
        rs_ok = True
        actual_crc = binascii.crc32(corrected_bytes[:byte_len]) & 0xFFFFFFFF
        crc_ok = actual_crc == expected_crc
    except Exception:
        pass

    return {
        "bit_len": bit_len,
        "frame_idx": frame_idx,
        "rs_ok": rs_ok,
        "crc_ok": crc_ok,
    }


def _select_best_candidate(
    candidate_grids: list[np.ndarray],
    last_frame_idx: int | None,
) -> tuple[np.ndarray, dict[str, object]]:
    """在同一原始帧的多个候选网格中选出最可信的一个。

    输入:
    - candidate_grids: 同一原始帧生成的多个候选网格。
    - last_frame_idx: 上一帧已选候选的 frame_idx，可为空。

    输出:
    - 最终选中的网格，以及该网格对应的评估信息。

    原理/流程:
    - 优先选择 CRC 正确的候选。
    - 若没有 CRC 正确候选，则退而选择 RS 成功候选。
    - 在同等级候选中，优先 frame_idx 与前一已选帧更连续的结果。
    """

    scored: list[tuple[tuple[int, int, int, float], np.ndarray, dict[str, object]]] = []
    for grid in candidate_grids:
        metrics = _evaluate_grid_candidate(grid)
        continuity = 0
        if last_frame_idx is not None:
            delta = int(metrics["frame_idx"]) - last_frame_idx
            if delta == 0:
                continuity = 2
            elif 0 < delta <= 2:
                continuity = 3
            elif delta > 2:
                continuity = 1
        score = (
            int(metrics["crc_ok"]),
            int(metrics["rs_ok"]),
            continuity,
            -abs(int(metrics["bit_len"]) - len(DATA_ITER)),
        )
        scored.append((score, grid, metrics))

    scored.sort(key=lambda item: item[0], reverse=True)
    _, best_grid, best_metrics = scored[0]
    return best_grid, best_metrics


def _prepare_debug_dirs(debug: bool, debug_dir: str) -> dict[str, Path]:
    """在调试模式下准备输出目录。

    输入:
    - debug: 是否开启调试产物输出。
    - debug_dir: 调试根目录。

    输出:
    - 一个目录映射表，供后续统一写入调试图像。

    原理/流程:
    - 先构造逻辑目录名到路径的映射。
    - 只有在调试开启时才真正创建这些目录。
    """

    debug_root = Path(debug_dir)
    dirs = {
        "root": debug_root,
        "processed": debug_root / "processed",
        "failed": debug_root / "failed",
        "yolo": debug_root / "yolo",
    }
    if debug:
        for directory in dirs.values():
            directory.mkdir(parents=True, exist_ok=True)
    return dirs


def _save_debug_images(
    *,
    index: int,
    raw_frame: np.ndarray,
    chosen_frame: np.ndarray | None,
    failed: bool,
    debug: bool,
    dirs: dict[str, Path],
    worker: Rectifier,
    model,
) -> None:
    """保存逐帧调试图像，便于定位矫正问题。

    输入:
    - index: 视频流中的帧序号。
    - raw_frame: 原始手机拍摄帧。
    - chosen_frame: 最终采用的矫正图像，可能为空。
    - failed: 当前帧是否完全矫正失败。
    - debug: 全局调试开关。
    - dirs: 预先准备好的调试目录。
    - worker: 共享的矫正器实例。
    - model: 仅用于可视化 YOLO 检测框的模型对象。

    输出:
    - 无。开启调试时会直接写文件。

    原理/流程:
    - 保存最终送入解码的处理结果。
    - 对失败帧额外保存原始图。
    - 若模型可用，则保存 YOLO 检测框可视化。
    """

    if not debug:
        return

    if chosen_frame is not None:
        cv2.imwrite(str(dirs["processed"] / f"frame_{index:05d}.png"), chosen_frame)
    if failed:
        cv2.imwrite(str(dirs["failed"] / f"frame_{index:05d}.png"), raw_frame)

    if model is not None:
        try:
            detections = detect_finders(model, raw_frame, conf=worker.conf, iou=worker.iou, max_det=worker.max_det)
            yolo_vis = draw_detection_only(raw_frame, detections, title=f"frame={index} boxes={len(detections)}")
            cv2.imwrite(str(dirs["yolo"] / f"frame_{index:05d}.png"), yolo_vis)
        except Exception:
            pass

def preprocess_frames_for_decoder(
    frames: list[np.ndarray] | Iterator[np.ndarray],
    *,
    rectifier: Rectifier | None = None,
    debug: bool = False,
    debug_dir: str = "output/rectify_debug",
) -> list[np.ndarray]:
    """对一串视频帧做矫正，并立即压缩成协议网格。

    输入:
    - frames: 原始视频帧迭代器或列表。
    - rectifier: 可选的共享矫正器实例。
    - debug: 是否保存调试图像。
    - debug_dir: 调试图像输出目录。

    输出:
    - 成功矫正后的 `GRID_SIZE x GRID_SIZE` 二值网格列表。

    原理/流程:
    - 逐帧调用矫正器，当前默认只走 YOLO 路径。
    - 失败帧直接丢弃，不再回退 raw。
    - 成功帧立即采样成紧凑网格，以降低内存占用。
    """

    worker = rectifier or Rectifier(
        model_path=RECTIFY_MODEL_PATH,
        enable_opencv_fallback=True,
    )
    processed_frames: list[np.ndarray] = []
    rectified_wins = 0
    rectify_failures = 0
    first_failure_reason: str | None = None
    method_counter: dict[str, int] = {"yolo": 0, "opencv": 0}

    dirs = _prepare_debug_dirs(debug, debug_dir)
    model = None
    if debug:
        try:
            model = worker._get_model()
        except Exception:
            model = None

    seen_frames = 0
    last_selected_frame_idx: int | None = None
    for idx, frame in enumerate(frames):
        seen_frames += 1
        rectified_candidates = worker.rectify_for_decoder_candidates(frame)

        if not rectified_candidates:
            rectify_failures += 1
            _save_debug_images(
                index=idx,
                raw_frame=frame,
                chosen_frame=None,
                failed=True,
                debug=debug,
                dirs=dirs,
                worker=worker,
                model=model,
            )
            if first_failure_reason is None:
                first_failure_reason = str(worker.last_error) if worker.last_error is not None else "unknown rectify failure"
        else:
            candidate_grids = [_to_grid(candidate) for candidate in rectified_candidates]
            best_grid, best_metrics = _select_best_candidate(candidate_grids, last_selected_frame_idx)
            processed_frames.append(best_grid)
            rectified_wins += 1
            last_selected_frame_idx = int(best_metrics["frame_idx"])
            if worker.last_method in method_counter:
                method_counter[worker.last_method] += 1
            _save_debug_images(
                index=idx,
                raw_frame=frame,
                chosen_frame=rectified_candidates[0],
                failed=False,
                debug=debug,
                dirs=dirs,
                worker=worker,
                model=model,
            )

        if (idx + 1) % 100 == 0:
            print(
                "rectify progress: "
                f"{idx + 1} frames, "
                f"kept={rectified_wins}, "
                f"failures={rectify_failures}, "
                f"yolo={method_counter['yolo']}, "
                f"opencv={method_counter['opencv']}"
            )

    print(f"rectify model: {worker.model_file}")
    print(
        "rectify summary: "
        f"seen={seen_frames}, "
        f"kept={rectified_wins}, "
        f"failures={rectify_failures}, "
        f"yolo={method_counter['yolo']}, "
        f"opencv={method_counter['opencv']}"
    )
    if first_failure_reason is not None:
        print(f"first rectify failure: {first_failure_reason}")
    if debug:
        print(f"rectify debug saved: {dirs['root'].resolve()}")
    return processed_frames


def video_to_qr_sequence(path: str, *, debug: bool = False, debug_dir: str = "output/rectify_debug") -> list[np.ndarray]:
    """将视频转换成适合 `_2Dcode.decode_image` 的网格序列。

    输入:
    - path: 实拍视频路径。
    - debug: 是否保存调试图像。
    - debug_dir: 调试图像输出目录。

    输出:
    - 由二值协议网格构成的列表。

    原理/流程:
    - 直接按流式读取视频帧。
    - 每帧在矫正成功后立即二值化采样。
    - 最终只返回紧凑网格，不保留整段原始视频帧。
    """

    frames = iter_video_frames(path)
    return preprocess_frames_for_decoder(frames, debug=debug, debug_dir=debug_dir)


def preprocess_frames_for_color_decoder(
    frames: list[np.ndarray] | Iterator[np.ndarray],
    *,
    rectifier: Rectifier | None = None,
    debug: bool = False,
    debug_dir: str = "output/rectify_debug_color",
) -> list[np.ndarray]:
    """对视频帧做 WRGB 主链路所需的彩色矫正预处理。

    输入：
    - frames: 原始视频帧列表或逐帧迭代器。
    - rectifier: 可选的共享矫正器；为空时默认启用复杂矫正。
    - debug: 是否保存调试图像。
    - debug_dir: 调试图像输出目录。

    输出：
    - 彩色 rectified 帧列表；后续直接交给 `_color2Dcode.decode_image()`。

    原理/流程：
    - 与黑白主链路不同，这里不再把图像压成二值 grid；
    - 逐帧调用复杂矫正，保留彩色 decoder 图；
    - 每个 raw frame 只保留一个候选，避免继续堆三候选拖慢主链路；
    - 调试时同步保存 raw / yolo / rectified 图，便于后续定位 WRGB 问题。
    """

    worker = rectifier or Rectifier(
        model_path=RECTIFY_MODEL_PATH,
        enable_opencv_fallback=True,
        use_second_stage_refine=True,
        decoder_expand_candidates=COMPLEX_DECODER_EXPAND_CANDIDATES,
        enable_center_anchor_refine=True,
        decoder_interpolation=cv2.INTER_LINEAR,
    )
    processed_frames: list[np.ndarray] = []
    rectify_failures = 0
    method_counter: dict[str, int] = {"yolo": 0, "opencv": 0}
    first_failure_reason: str | None = None

    dirs = _prepare_debug_dirs(debug, debug_dir)
    model = None
    if debug:
        try:
            model = worker._get_model()
        except Exception:
            model = None

    seen_frames = 0
    for idx, frame in enumerate(frames):
        seen_frames += 1
        rectified_candidates = worker.rectify_for_decoder_candidates(frame)

        if not rectified_candidates:
            rectify_failures += 1
            _save_debug_images(
                index=idx,
                raw_frame=frame,
                chosen_frame=None,
                failed=True,
                debug=debug,
                dirs=dirs,
                worker=worker,
                model=model,
            )
            if first_failure_reason is None:
                first_failure_reason = str(worker.last_error) if worker.last_error is not None else "unknown rectify failure"
            continue

        chosen_frame = rectified_candidates[0]
        processed_frames.append(chosen_frame)
        if worker.last_method in method_counter:
            method_counter[worker.last_method] += 1
        _save_debug_images(
            index=idx,
            raw_frame=frame,
            chosen_frame=chosen_frame,
            failed=False,
            debug=debug,
            dirs=dirs,
            worker=worker,
            model=model,
        )

        if (idx + 1) % 100 == 0:
            print(
                "wrgb rectify progress: "
                f"{idx + 1} frames, "
                f"kept={len(processed_frames)}, "
                f"failures={rectify_failures}, "
                f"yolo={method_counter['yolo']}, "
                f"opencv={method_counter['opencv']}"
            )

    print(f"rectify model: {worker.model_file}")
    print(
        "wrgb rectify summary: "
        f"seen={seen_frames}, "
        f"kept={len(processed_frames)}, "
        f"failures={rectify_failures}, "
        f"yolo={method_counter['yolo']}, "
        f"opencv={method_counter['opencv']}"
    )
    if first_failure_reason is not None:
        print(f"first wrgb rectify failure: {first_failure_reason}")
    if debug:
        print(f"wrgb rectify debug saved: {dirs['root'].resolve()}")
    return processed_frames


def video_to_color_sequence(
    path: str,
    *,
    debug: bool = False,
    debug_dir: str = "output/rectify_debug_color",
) -> list[np.ndarray]:
    """将实拍视频转换成 WRGB 主协议可直接解码的彩色序列。

    输入：
    - path: 实拍视频路径。
    - debug: 是否保存调试图像。
    - debug_dir: 调试图像输出目录。

    输出：
    - 彩色 rectified 帧列表。

    原理/流程：
    - 流式读取视频；
    - 逐帧做复杂矫正与中心锚点微调；
    - 保留彩色 rectified 图，供 WRGB 协议主解码使用。
    """

    frames = iter_video_frames(path)
    return preprocess_frames_for_color_decoder(frames, debug=debug, debug_dir=debug_dir)
