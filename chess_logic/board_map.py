"""Square map + detection-to-square mapping utilities for vision MVP."""

from __future__ import annotations

from dataclasses import dataclass

import chess
import cv2
import numpy as np


PIECE_NAME_TO_SYMBOL = {
    "white-pawn": "P",
    "white-knight": "N",
    "white-bishop": "B",
    "white-rook": "R",
    "white-queen": "Q",
    "white-king": "K",
    "black-pawn": "p",
    "black-knight": "n",
    "black-bishop": "b",
    "black-rook": "r",
    "black-queen": "q",
    "black-king": "k",
}


@dataclass
class SquareRegion:
    square: str
    row: int
    col: int
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    center: tuple[int, int]
    polygon: list[tuple[int, int]]


def get_square_map(warped_board: np.ndarray) -> dict[str, SquareRegion]:
    """Return all 64 square regions for a normalized board image.

    Stable order is chessboard order from a8..h8 down to a1..h1.
    """
    h, w = warped_board.shape[:2]
    step_x = w / 8.0
    step_y = h / 8.0
    mapping: dict[str, SquareRegion] = {}

    for row in range(8):  # 0 = rank 8
        for col in range(8):  # 0 = file a
            x1 = int(round(col * step_x))
            y1 = int(round(row * step_y))
            x2 = int(round((col + 1) * step_x))
            y2 = int(round((row + 1) * step_y))

            file_char = chr(ord("a") + col)
            rank_char = str(8 - row)
            square = f"{file_char}{rank_char}"

            mapping[square] = SquareRegion(
                square=square,
                row=row,
                col=col,
                bbox=(x1, y1, x2, y2),
                center=((x1 + x2) // 2, (y1 + y2) // 2),
                polygon=[(x1, y1), (x2, y1), (x2, y2), (x1, y2)],
            )
    return mapping


def draw_square_overlay(image: np.ndarray, square_map: dict[str, SquareRegion]) -> np.ndarray:
    """Draw all 64 square boundaries and labels for debugging."""
    out = image.copy()
    for square in sorted(square_map.keys(), key=lambda s: (8 - int(s[1]), ord(s[0]))):
        region = square_map[square]
        x1, y1, x2, y2 = region.bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), (60, 180, 255), 1)
        cv2.putText(
            out,
            square,
            (x1 + 3, y1 + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return out


def _piece_symbol_from_class_name(class_name: str) -> str | None:
    return PIECE_NAME_TO_SYMBOL.get(class_name)


def map_detections_to_squares(
    detections: list[dict],
    square_map: dict[str, SquareRegion],
) -> tuple[dict[str, str | None], dict[str, float], dict[str, dict]]:
    """Map YOLO detections onto squares with conflict resolution.

    - square occupancy: square -> piece symbol | None
    - confidence map: square -> confidence
    - assignment debug: square -> raw detection summary
    """
    occupancy = {sq: None for sq in square_map}
    confidences = {sq: 0.0 for sq in square_map}
    assignments: dict[str, dict] = {}

    for det in detections:
        bbox = det.get("bbox", {})
        x1, y1 = int(bbox.get("x1", 0)), int(bbox.get("y1", 0))
        x2, y2 = int(bbox.get("x2", 0)), int(bbox.get("y2", 0))
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        conf = float(det.get("confidence", 0.0))
        class_name = str(det.get("class_name", ""))
        piece_symbol = _piece_symbol_from_class_name(class_name)
        if piece_symbol is None:
            continue

        chosen_square = None
        for square, region in square_map.items():
            sx1, sy1, sx2, sy2 = region.bbox
            if sx1 <= cx < sx2 and sy1 <= cy < sy2:
                chosen_square = square
                break
        if chosen_square is None:
            continue

        if conf > confidences[chosen_square]:
            occupancy[chosen_square] = piece_symbol
            confidences[chosen_square] = conf
            assignments[chosen_square] = {
                "class_name": class_name,
                "piece_symbol": piece_symbol,
                "confidence": conf,
                "center": [cx, cy],
                "bbox": [x1, y1, x2, y2],
            }
    return occupancy, confidences, assignments


def board_from_occupancy(occupancy: dict[str, str | None]) -> list[list[str]]:
    """Convert occupancy mapping into 8x8 printable matrix."""
    board = []
    for rank in range(8, 0, -1):
        row = []
        for file_char in "abcdefgh":
            sq = f"{file_char}{rank}"
            row.append(occupancy.get(sq) or ".")
        board.append(row)
    return board


def format_board_map(board: list[list[str]]) -> str:
    """Return text board map for debugging logs."""
    lines = []
    for rank_idx, row in enumerate(board):
        rank = 8 - rank_idx
        lines.append(f"{rank} " + " ".join(row))
    lines.append("  a b c d e f g h")
    return "\n".join(lines)


def build_board_map(
    detections: list[dict],
    board_bbox: tuple[int, int, int, int],
) -> list[list[str]]:
    """Compatibility helper expected by existing detector code."""
    x1, y1, x2, y2 = board_bbox
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    mock_board_img = np.zeros((height, width, 3), dtype=np.uint8)
    square_map = get_square_map(mock_board_img)
    occupancy, _, _ = map_detections_to_squares(detections, square_map)
    return board_from_occupancy(occupancy)


def build_board_map_with_confidences(
    detections: list[dict],
    board_bbox: tuple[int, int, int, int],
) -> tuple[list[list[str]], list[list[float]]]:
    """Return board matrix plus confidence matrix."""
    x1, y1, x2, y2 = board_bbox
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    mock_board_img = np.zeros((height, width, 3), dtype=np.uint8)
    square_map = get_square_map(mock_board_img)
    occupancy, conf_map, _ = map_detections_to_squares(detections, square_map)

    board = board_from_occupancy(occupancy)
    conf_rows = []
    for rank in range(8, 0, -1):
        row = []
        for file_char in "abcdefgh":
            row.append(float(conf_map.get(f"{file_char}{rank}", 0.0)))
        conf_rows.append(row)
    return board, conf_rows


def board_to_square_occupancy(board: chess.Board) -> dict[str, str | None]:
    """Convert python-chess board to square occupancy dict."""
    occ: dict[str, str | None] = {}
    for sq in chess.SQUARES:
        name = chess.square_name(sq)
        piece = board.piece_at(sq)
        occ[name] = piece.symbol() if piece else None
    return occ

