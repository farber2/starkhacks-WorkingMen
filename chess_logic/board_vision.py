"""Board normalization helpers for camera-to-board vision MVP.

This module focuses on a fixed-camera setup:
- detect an approximate board quadrilateral with OpenCV
- warp to a stable top-down square
- return metadata useful for debugging
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class BoardVisionResult:
    """Normalized board image plus debug metadata."""

    warped_board: np.ndarray
    corners: list[tuple[float, float]]
    source_size: tuple[int, int]
    warped_size: int


def _order_points(points: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = points.sum(axis=1)
    rect[0] = points[np.argmin(s)]
    rect[2] = points[np.argmax(s)]

    diff = np.diff(points, axis=1)
    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]
    return rect


def _largest_quad(contours: list[np.ndarray]) -> np.ndarray | None:
    best = None
    best_area = 0.0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 2000:
            continue
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4 and area > best_area:
            best_area = area
            best = approx.reshape(4, 2).astype("float32")
    return best


def detect_board_corners(frame_bgr: np.ndarray) -> list[tuple[float, float]]:
    """Detect board corners or fall back to full-frame corners.

    Returns corners in order: TL, TR, BR, BL.
    """
    h, w = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 60, 160)
    edges = cv2.dilate(edges, np.ones((3, 3), dtype=np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    quad = _largest_quad(contours)

    if quad is None:
        quad = np.array(
            [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
            dtype="float32",
        )
    ordered = _order_points(quad)
    return [(float(x), float(y)) for x, y in ordered]


def warp_board(frame_bgr: np.ndarray, corners: list[tuple[float, float]], size: int = 800) -> np.ndarray:
    """Warp board image to a top-down square view of fixed size."""
    src = np.array(corners, dtype="float32")
    dst = np.array(
        [[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(frame_bgr, matrix, (size, size))


def normalize_board_image(frame_bgr: np.ndarray, size: int = 800) -> BoardVisionResult:
    """Detect corners and return normalized top-down board image."""
    if frame_bgr is None or frame_bgr.size == 0:
        raise ValueError("Input frame is empty.")
    h, w = frame_bgr.shape[:2]
    corners = detect_board_corners(frame_bgr)
    warped = warp_board(frame_bgr, corners, size=size)
    return BoardVisionResult(
        warped_board=warped,
        corners=corners,
        source_size=(w, h),
        warped_size=size,
    )

