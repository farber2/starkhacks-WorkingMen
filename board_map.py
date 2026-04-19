"""Compatibility wrapper for vision board-map helpers.

Existing detector code imports `board_map` at repo root.
Core implementation lives in `chess_logic.board_map`.
"""

from chess_logic.board_map import (  # noqa: F401
    build_board_map,
    build_board_map_with_confidences,
    board_from_occupancy,
    board_to_square_occupancy,
    draw_square_overlay,
    format_board_map,
    get_square_map,
    map_detections_to_squares,
)

