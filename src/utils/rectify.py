import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


MODEL_PATH = "best.pt"


def resize_for_display(image: np.ndarray, max_side: int = 1000) -> np.ndarray:
    h, w = image.shape[:2]
    cur_max = max(h, w)
    if cur_max <= max_side:
        return image

    scale = max_side / float(cur_max)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    return cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)


def detect_finders(model: YOLO, image: np.ndarray, conf: float, iou: float, max_det: int):
    results = model(image, conf=conf, iou=iou, max_det=max_det, verbose=False)
    if not results:
        return []

    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return []

    xyxy = boxes.xyxy.detach().cpu().numpy()
    scores = boxes.conf.detach().cpu().numpy()

    out = []
    for box, score in zip(xyxy, scores):
        x1, y1, x2, y2 = box.tolist()
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        out.append(
            {
                "xyxy": (float(x1), float(y1), float(x2), float(y2)),
                "score": float(score),
                "area": float(area),
            }
        )
    return out


def center_of(box_xyxy):
    x1, y1, x2, y2 = box_xyxy
    return np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)


def pick_point_by_role(points: np.ndarray, role: str) -> np.ndarray:
    if role == "tl":
        idx = int(np.argmin(points[:, 0] + points[:, 1]))
    elif role == "tr":
        idx = int(np.argmax(points[:, 0] - points[:, 1]))
    elif role == "br":
        idx = int(np.argmax(points[:, 0] + points[:, 1]))
    elif role == "bl":
        idx = int(np.argmax(points[:, 1] - points[:, 0]))
    else:
        raise ValueError(f"Unknown role: {role}")
    return points[idx].astype(np.float32)


def bbox_corner_for_role(box_xyxy, role: str) -> np.ndarray:
    x1, y1, x2, y2 = box_xyxy
    pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
    return pick_point_by_role(pts, role)


def refine_corner_from_box(image: np.ndarray, box_xyxy, role: str, expand_ratio: float = 0.18) -> np.ndarray:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box_xyxy
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    ex = bw * max(0.0, expand_ratio)
    ey = bh * max(0.0, expand_ratio)

    rx1 = int(max(0, np.floor(x1 - ex)))
    ry1 = int(max(0, np.floor(y1 - ey)))
    rx2 = int(min(w - 1, np.ceil(x2 + ex)))
    ry2 = int(min(h - 1, np.ceil(y2 + ey)))

    if rx2 <= rx1 or ry2 <= ry1:
        return bbox_corner_for_role(box_xyxy, role)

    roi = image[ry1 : ry2 + 1, rx1 : rx2 + 1]
    if roi.size == 0:
        return bbox_corner_for_role(box_xyxy, role)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, bw_img = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(bw_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return bbox_corner_for_role(box_xyxy, role)

    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 16:
        return bbox_corner_for_role(box_xyxy, role)

    rect = cv2.minAreaRect(cnt)
    pts = cv2.boxPoints(rect).astype(np.float32)
    pts[:, 0] += rx1
    pts[:, 1] += ry1
    return pick_point_by_role(pts, role)


def build_quad_from_roles(image: np.ndarray, roles, refine: bool, expand_ratio: float) -> np.ndarray:
    order = ["tl", "tr", "br", "bl"]
    pts = []
    for role in order:
        box = roles[role]["xyxy"]
        if refine:
            p = refine_corner_from_box(image, box, role, expand_ratio=expand_ratio)
        else:
            p = bbox_corner_for_role(box, role)
        pts.append(p)
    return np.array(pts, dtype=np.float32)


def pick_3_big_1_small(detections, min_area_ratio=1.2):
    if len(detections) < 4:
        return None

    top4 = sorted(detections, key=lambda d: d["area"], reverse=True)[:4]
    top4_sorted = sorted(top4, key=lambda d: d["area"], reverse=True)

    big3 = top4_sorted[:3]
    small1 = top4_sorted[3]

    area_ratio = big3[2]["area"] / max(small1["area"], 1e-6)
    if area_ratio < min_area_ratio:
        return None

    return big3, small1


def assign_roles(big3, small1):
    big_pts = [center_of(d["xyxy"]) for d in big3]

    # TL: smallest x+y
    sums = [p[0] + p[1] for p in big_pts]
    tl_idx = int(np.argmin(sums))
    tl_det = big3[tl_idx]
    tl_center = big_pts[tl_idx]

    rem = [(big3[i], big_pts[i]) for i in range(3) if i != tl_idx]
    if rem[0][1][0] >= rem[1][1][0]:
        tr_det = rem[0][0]
        bl_det = rem[1][0]
        tr_center = rem[0][1]
        bl_center = rem[1][1]
    else:
        tr_det = rem[1][0]
        bl_det = rem[0][0]
        tr_center = rem[1][1]
        bl_center = rem[0][1]

    br_det = small1
    br_center = center_of(small1["xyxy"])

    centers = np.array([tl_center, tr_center, br_center, bl_center], dtype=np.float32)
    roles = {
        "tl": tl_det,
        "tr": tr_det,
        "br": br_det,
        "bl": bl_det,
    }
    return centers, roles


def assign_roles_from_four(dets4):
    # Fallback when size-based 3-big-1-small split is unstable.
    pts = [center_of(d["xyxy"]) for d in dets4]

    tl_idx = int(np.argmin([p[0] + p[1] for p in pts]))
    br_idx = int(np.argmax([p[0] + p[1] for p in pts]))

    rem = [i for i in range(4) if i not in (tl_idx, br_idx)]
    i1, i2 = rem[0], rem[1]
    # x-y larger tends to be TR, the other is BL.
    if (pts[i1][0] - pts[i1][1]) >= (pts[i2][0] - pts[i2][1]):
        tr_idx, bl_idx = i1, i2
    else:
        tr_idx, bl_idx = i2, i1

    centers = np.array([pts[tl_idx], pts[tr_idx], pts[br_idx], pts[bl_idx]], dtype=np.float32)
    roles = {
        "tl": dets4[tl_idx],
        "tr": dets4[tr_idx],
        "br": dets4[br_idx],
        "bl": dets4[bl_idx],
    }
    return centers, roles


def build_dst_points(out_size: int, center_margin_ratio: float) -> np.ndarray:
    margin = int(round(out_size * max(0.0, min(0.45, center_margin_ratio))))
    return np.array(
        [
            [margin, margin],
            [out_size - 1 - margin, margin],
            [out_size - 1 - margin, out_size - 1 - margin],
            [margin, out_size - 1 - margin],
        ],
        dtype=np.float32,
    )


def rectify_image(image: np.ndarray, src_pts: np.ndarray, out_size: int, center_margin_ratio: float):
    # Map center points to an inset quad (not image corners) to keep more outer context.
    dst_pts = build_dst_points(out_size, center_margin_ratio)
    mat = cv2.getPerspectiveTransform(src_pts, dst_pts)
    rectified = cv2.warpPerspective(image, mat, (out_size, out_size))
    return rectified, mat


def rectify_image_cropped(image: np.ndarray, src_pts: np.ndarray, out_size: int) -> np.ndarray:
    # Direct crop-style rectification from four refined corners.
    dst_pts = np.array(
        [[0, 0], [out_size - 1, 0], [out_size - 1, out_size - 1], [0, out_size - 1]],
        dtype=np.float32,
    )
    mat = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(image, mat, (out_size, out_size))


def warp_keep_all(image: np.ndarray, homography: np.ndarray, out_size: int) -> np.ndarray:
    # Apply homography while fitting the whole transformed image into output canvas.
    h, w = image.shape[:2]
    corners = np.array(
        [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
        dtype=np.float32,
    )
    tc = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), homography).reshape(-1, 2)

    min_x = float(np.min(tc[:, 0]))
    min_y = float(np.min(tc[:, 1]))
    max_x = float(np.max(tc[:, 0]))
    max_y = float(np.max(tc[:, 1]))

    span_w = max(max_x - min_x, 1e-6)
    span_h = max(max_y - min_y, 1e-6)
    scale = min((out_size - 1) / span_w, (out_size - 1) / span_h)

    tx = -min_x * scale + ((out_size - 1) - span_w * scale) * 0.5
    ty = -min_y * scale + ((out_size - 1) - span_h * scale) * 0.5

    fit = np.array(
        [
            [scale, 0.0, tx],
            [0.0, scale, ty],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    composed = fit @ homography
    return cv2.warpPerspective(image, composed, (out_size, out_size), borderMode=cv2.BORDER_REPLICATE)


def draw_debug(image: np.ndarray, detections, src_pts: np.ndarray):
    vis = image.copy()
    for d in detections:
        x1, y1, x2, y2 = map(int, d["xyxy"])
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(
            vis,
            f"{d['score']:.2f}",
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

    colors = [(0, 255, 0), (0, 255, 255), (0, 128, 255), (255, 255, 0)]
    names = ["TL(center)", "TR(center)", "BR(center)", "BL(center)"]
    for i, p in enumerate(src_pts):
        x, y = int(p[0]), int(p[1])
        cv2.circle(vis, (x, y), 6, colors[i], -1)
        cv2.putText(
            vis,
            names[i],
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            colors[i],
            2,
            cv2.LINE_AA,
        )
    return vis


def draw_role_points(image: np.ndarray, role_pts: np.ndarray, label: str):
    vis = image.copy()
    colors = [(0, 255, 0), (0, 255, 255), (0, 128, 255), (255, 255, 0)]
    names = ["TL", "TR", "BR", "BL"]
    for i, p in enumerate(role_pts):
        x, y = int(p[0]), int(p[1])
        cv2.circle(vis, (x, y), 6, colors[i], -1)
        cv2.putText(
            vis,
            names[i],
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            colors[i],
            2,
            cv2.LINE_AA,
        )
    cv2.putText(
        vis,
        label,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return vis


def draw_detection_only(image: np.ndarray, detections, title: str = "pass2"):
    vis = image.copy()
    for d in detections:
        x1, y1, x2, y2 = map(int, d["xyxy"])
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 180, 255), 2)
        cv2.putText(
            vis,
            f"{d['score']:.2f}",
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 180, 255),
            2,
            cv2.LINE_AA,
        )
    cv2.putText(
        vis,
        f"{title}: {len(detections)} boxes",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 180, 255),
        2,
        cv2.LINE_AA,
    )
    return vis


def main():
    parser = argparse.ArgumentParser(description="Detect 3 big + 1 small finder and rectify perspective.")
    parser.add_argument("image", type=str, help="Input image path")
    parser.add_argument("--output-dir", type=str, default="output", help="Directory to save all four output images")
    parser.add_argument("--size", type=int, default=1024, help="Rectified output size")
    parser.add_argument(
        "--center-margin-ratio",
        type=float,
        default=0.18,
        help="Margin ratio for mapping pass1 center points in rectified image (higher keeps more outer area)",
    )
    parser.add_argument(
        "--stage2-margin-ratio",
        type=float,
        default=0.0,
        help="Target margin ratio for second rectification. Uses keep-all warp to avoid extra crop.",
    )
    parser.add_argument(
        "--refine-corners",
        action="store_true",
        default=True,
        help="Refine QR corner points from finder ROI geometry (default on)",
    )
    parser.add_argument(
        "--corner-expand-ratio",
        type=float,
        default=0.18,
        help="ROI expand ratio when refining corners inside detection boxes",
    )
    parser.add_argument("--conf", type=float, default=0.15, help="Detection confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=20, help="Max detections")
    parser.add_argument(
        "--min-area-ratio",
        type=float,
        default=1.2,
        help="Minimum ratio between smallest big finder area and small finder area",
    )
    parser.add_argument("--show", action="store_true", help="Display debug and rectified windows")
    parser.add_argument(
        "--show-max-side",
        type=int,
        default=1000,
        help="Max side length for display windows only",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}. Put best.pt in current directory or update MODEL_PATH in rectify.py"
        )

    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(MODEL_PATH)
    detections = detect_finders(model, image, conf=args.conf, iou=args.iou, max_det=args.max_det)

    picked = pick_3_big_1_small(detections, min_area_ratio=args.min_area_ratio)
    used_fallback = False
    if picked is None:
        if len(detections) < 4:
            raise RuntimeError("Failed to find enough finder detections for rectification.")
        top4 = sorted(detections, key=lambda d: d["score"], reverse=True)[:4]
        center_pts, roles = assign_roles_from_four(top4)
        used_fallback = True
    else:
        big3, small1 = picked
        center_pts, roles = assign_roles(big3, small1)
    stage1_corner_pts = build_quad_from_roles(
        image,
        roles,
        refine=args.refine_corners,
        expand_ratio=args.corner_expand_ratio,
    )

    rectified_stage1, _ = rectify_image(
        image,
        stage1_corner_pts,
        out_size=args.size,
        center_margin_ratio=args.center_margin_ratio,
    )
    debug_vis = draw_debug(image, detections, center_pts)
    debug_vis = draw_role_points(debug_vis, stage1_corner_pts, "stage1 corner points")

    pass2_detections = detect_finders(model, rectified_stage1, conf=args.conf, iou=args.iou, max_det=args.max_det)
    pass2_vis = draw_detection_only(rectified_stage1, pass2_detections, title="pass2")

    # Use second-pass centers for one more homography to reduce residual skew.
    rectified_stage2 = rectified_stage1
    pass2_refined = False
    pass2_picked = pick_3_big_1_small(pass2_detections, min_area_ratio=args.min_area_ratio)
    pass2_corner_pts = None
    if pass2_picked is not None:
        pass2_big3, pass2_small1 = pass2_picked
        _, pass2_roles = assign_roles(pass2_big3, pass2_small1)
        pass2_corner_pts = build_quad_from_roles(
            rectified_stage1,
            pass2_roles,
            refine=args.refine_corners,
            expand_ratio=args.corner_expand_ratio,
        )
        stage2_dst = build_dst_points(args.size, args.stage2_margin_ratio)
        stage2_h = cv2.getPerspectiveTransform(pass2_corner_pts, stage2_dst)
        rectified_stage2 = warp_keep_all(rectified_stage1, stage2_h, out_size=args.size)
        pass2_refined = True
    elif len(pass2_detections) >= 4:
        pass2_top4 = sorted(pass2_detections, key=lambda d: d["score"], reverse=True)[:4]
        _, pass2_roles = assign_roles_from_four(pass2_top4)
        pass2_corner_pts = build_quad_from_roles(
            rectified_stage1,
            pass2_roles,
            refine=args.refine_corners,
            expand_ratio=args.corner_expand_ratio,
        )
        stage2_dst = build_dst_points(args.size, args.stage2_margin_ratio)
        stage2_h = cv2.getPerspectiveTransform(pass2_corner_pts, stage2_dst)
        rectified_stage2 = warp_keep_all(rectified_stage1, stage2_h, out_size=args.size)
        pass2_refined = True

    # Extra output: cropped rectification from refined corner points.
    if pass2_corner_pts is not None:
        refined_corner_cropped = rectify_image_cropped(rectified_stage1, pass2_corner_pts, out_size=args.size)
    else:
        refined_corner_cropped = rectify_image_cropped(image, stage1_corner_pts, out_size=args.size)

    first_detect_path = output_dir / "01_first_detection.jpg"
    first_rectified_path = output_dir / "02_first_rectified.jpg"
    second_detect_path = output_dir / "03_second_detection.jpg"
    second_rectified_path = output_dir / "04_second_rectified.jpg"
    refined_corner_cropped_path = output_dir / "05_refined_corner_cropped.jpg"

    cv2.imwrite(str(first_detect_path), debug_vis)
    cv2.imwrite(str(first_rectified_path), rectified_stage1)
    cv2.imwrite(str(second_detect_path), pass2_vis)
    cv2.imwrite(str(second_rectified_path), rectified_stage2)
    cv2.imwrite(str(refined_corner_cropped_path), refined_corner_cropped)

    print(f"Model: {MODEL_PATH}")
    print(f"Input: {image_path.resolve()}")
    print(f"Output dir: {output_dir.resolve()}")
    print(f"Pass1 boxes: {len(detections)}")
    print(f"Pass1 fallback-4pt used: {used_fallback}")
    print(f"Pass2 boxes: {len(pass2_detections)}")
    print(f"Pass2 refine applied: {pass2_refined}")
    print(f"Saved: {first_detect_path.resolve()}")
    print(f"Saved: {first_rectified_path.resolve()}")
    print(f"Saved: {second_detect_path.resolve()}")
    print(f"Saved: {second_rectified_path.resolve()}")
    print(f"Saved: {refined_corner_cropped_path.resolve()}")

    if args.show:
        debug_show = resize_for_display(debug_vis, max_side=args.show_max_side)
        rectified1_show = resize_for_display(rectified_stage1, max_side=args.show_max_side)
        pass2_show = resize_for_display(pass2_vis, max_side=args.show_max_side)
        rectified2_show = resize_for_display(rectified_stage2, max_side=args.show_max_side)
        refined_cropped_show = resize_for_display(refined_corner_cropped, max_side=args.show_max_side)
        cv2.imshow("rectify_debug", debug_show)
        cv2.imshow("rectified_pass1", rectified1_show)
        cv2.imshow("rectified_pass2", pass2_show)
        cv2.imshow("rectified_pass2_refined", rectified2_show)
        cv2.imshow("rectified_refined_corner_cropped", refined_cropped_show)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
