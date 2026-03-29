"""Evaluation metrics for stroke prediction.

Three complementary metrics:
1. DTW (Dynamic Time Warping) — trajectory similarity accounting for warping
2. Chamfer distance — point-set distance (order-agnostic)
3. Visual SSIM — render predicted strokes and compare to input image
"""

import numpy as np
import torch
from PIL import Image, ImageDraw


# ---------- DTW ----------

def dtw_distance(pred: np.ndarray, target: np.ndarray) -> float:
    """Dynamic Time Warping distance between two trajectories.

    Compares only the (x, y) positions (ignoring pen state).
    Uses the absolute positions (cumulative sum of deltas).

    Args:
        pred:   (N, 3) predicted deltas (dx, dy, pen)
        target: (M, 3) ground truth deltas (dx, dy, pen)

    Returns:
        Normalized DTW distance.
    """
    # Convert deltas to absolute positions
    pred_xy = np.cumsum(pred[:, :2], axis=0)
    target_xy = np.cumsum(target[:, :2], axis=0)

    n, m = len(pred_xy), len(target_xy)
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = np.sqrt(np.sum((pred_xy[i - 1] - target_xy[j - 1]) ** 2))
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])

    return dtw[n, m] / (n + m)


def dtw_distance_fast(pred: np.ndarray, target: np.ndarray, radius: int = 20) -> float:
    """DTW with Sakoe-Chiba band constraint for speed.

    Same interface as dtw_distance but O(n * radius) instead of O(n * m).
    """
    pred_xy = np.cumsum(pred[:, :2], axis=0)
    target_xy = np.cumsum(target[:, :2], axis=0)

    n, m = len(pred_xy), len(target_xy)
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0.0

    for i in range(1, n + 1):
        j_start = max(1, i - radius)
        j_end = min(m, i + radius)
        for j in range(j_start, j_end + 1):
            cost = np.sqrt(np.sum((pred_xy[i - 1] - target_xy[j - 1]) ** 2))
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])

    return dtw[n, m] / (n + m)


# ---------- Chamfer Distance ----------

def chamfer_distance(pred: np.ndarray, target: np.ndarray) -> float:
    """Chamfer distance between predicted and target point sets.

    Symmetric: average of nearest-neighbor distances in both directions.

    Args:
        pred:   (N, 3) predicted deltas
        target: (M, 3) ground truth deltas

    Returns:
        Chamfer distance (lower is better).
    """
    pred_xy = np.cumsum(pred[:, :2], axis=0)
    target_xy = np.cumsum(target[:, :2], axis=0)

    # pred → target
    dists_p2t = np.min(
        np.sqrt(np.sum((pred_xy[:, None, :] - target_xy[None, :, :]) ** 2, axis=-1)),
        axis=1,
    )
    # target → pred
    dists_t2p = np.min(
        np.sqrt(np.sum((target_xy[:, None, :] - pred_xy[None, :, :]) ** 2, axis=-1)),
        axis=1,
    )

    return (dists_p2t.mean() + dists_t2p.mean()) / 2


# ---------- Visual SSIM ----------

def render_trajectory(deltas: np.ndarray, width: int = 192, height: int = 64) -> np.ndarray:
    """Render a trajectory (deltas) to a grayscale image.

    Args:
        deltas: (N, 3) normalized deltas (dx, dy, pen_state)
        width, height: output image size

    Returns:
        (H, W) uint8 grayscale image.
    """
    xy = np.cumsum(deltas[:, :2], axis=0)
    pen = deltas[:, 2]

    # Fit to image
    xy_min = xy.min(axis=0)
    xy_max = xy.max(axis=0)
    xy_range = xy_max - xy_min
    xy_range[xy_range < 1e-6] = 1.0

    pad = 6
    scale = min((width - 2 * pad) / xy_range[0], (height - 2 * pad) / xy_range[1])
    scaled = (xy - xy_min) * scale
    offset_x = (width - xy_range[0] * scale) / 2
    offset_y = (height - xy_range[1] * scale) / 2
    img_xy = scaled + np.array([offset_x, offset_y])

    img = Image.new("L", (width, height), 255)
    dr = ImageDraw.Draw(img)
    for i in range(1, len(img_xy)):
        if pen[i - 1] < 0.5:  # pen down
            dr.line(
                [tuple(img_xy[i - 1]), tuple(img_xy[i])],
                fill=0, width=2,
            )
    return np.array(img)


def ssim_1d(img1: np.ndarray, img2: np.ndarray) -> float:
    """Simplified SSIM between two grayscale images.

    Args:
        img1, img2: (H, W) uint8 arrays

    Returns:
        SSIM value in [-1, 1] (higher is better).
    """
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    mu1 = img1.mean()
    mu2 = img2.mean()
    sigma1_sq = img1.var()
    sigma2_sq = img2.var()
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return float(ssim)


def visual_ssim(pred_deltas: np.ndarray, target_deltas: np.ndarray, **kwargs) -> float:
    """Render both trajectories and compute SSIM between them.

    Args:
        pred_deltas:   (N, 3) predicted deltas
        target_deltas: (M, 3) ground truth deltas

    Returns:
        SSIM value (higher is better).
    """
    img_pred = render_trajectory(pred_deltas, **kwargs)
    img_target = render_trajectory(target_deltas, **kwargs)
    return ssim_1d(img_pred, img_target)


def pen_accuracy(pred_deltas: np.ndarray, target_deltas: np.ndarray) -> float:
    """Pen up/down classification accuracy.

    Truncates to the shorter of the two sequences.

    Returns:
        Accuracy in [0, 1].
    """
    min_len = min(len(pred_deltas), len(target_deltas))
    pred_pen = (pred_deltas[:min_len, 2] > 0.5).astype(int)
    target_pen = (target_deltas[:min_len, 2] > 0.5).astype(int)
    return float((pred_pen == target_pen).mean())


def compute_all_metrics(pred_deltas: np.ndarray, target_deltas: np.ndarray) -> dict:
    """Compute all metrics for a single sample.

    Args:
        pred_deltas:   (N, 3) predicted (dx, dy, pen)
        target_deltas: (M, 3) ground truth (dx, dy, pen)

    Returns:
        dict with dtw, chamfer, ssim, pen_accuracy
    """
    return {
        "dtw": dtw_distance_fast(pred_deltas, target_deltas),
        "chamfer": chamfer_distance(pred_deltas, target_deltas),
        "ssim": visual_ssim(pred_deltas, target_deltas),
        "pen_accuracy": pen_accuracy(pred_deltas, target_deltas),
    }
