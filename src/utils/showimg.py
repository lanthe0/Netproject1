import numpy as np

def matrix_to_bw_image(matrix: np.ndarray, pixel_per_cell: int = 1) -> np.ndarray:
    """Convert a 2D binary matrix to a grayscale image.

    Args:
        matrix: 2D ndarray, values should be 0/1 (or bool).
        pixel_per_cell: Output scale factor per matrix cell.

    Returns:
        Grayscale uint8 image where 1 -> black(0), 0 -> white(255).
    """
    arr = np.asarray(matrix)
    if arr.ndim != 2:
        raise ValueError("matrix_to_bw_image expects a 2D matrix")
    if pixel_per_cell <= 0:
        raise ValueError("pixel_per_cell must be positive")

    # Normalize to binary mask (non-zero is treated as 1).
    bin_arr = (arr != 0).astype(np.uint8)
    img = ((1 - bin_arr) * 255).astype(np.uint8)

    if pixel_per_cell > 1:
        img = np.repeat(np.repeat(img, pixel_per_cell, axis=0), pixel_per_cell, axis=1)
    return img


def show_binary_matrix(
    matrix: np.ndarray,
    pixel_per_cell: int = 1,
    window_name: str = "Binary Matrix",
    wait_ms: int = 0,
) -> np.ndarray:
    """Display a binary matrix as an image and return the rendered image.

    Note: requires OpenCV (cv2) installed for display.
    """
    img = matrix_to_bw_image(matrix, pixel_per_cell=pixel_per_cell)
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("show_binary_matrix requires OpenCV (cv2)") from exc

    cv2.imshow(window_name, img)
    cv2.waitKey(wait_ms)
    if wait_ms == 0:
        cv2.destroyWindow(window_name)
    return img
