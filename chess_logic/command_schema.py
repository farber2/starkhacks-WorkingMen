"""Schemas for robot-executable chess commands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(slots=True)
class RobotMoveCommand:
    """Represents a robot-ready command for a single chess move."""

    action_type: str
    from_square: Optional[str] = None
    to_square: Optional[str] = None
    remove_square: Optional[str] = None
    king_from: Optional[str] = None
    king_to: Optional[str] = None
    rook_from: Optional[str] = None
    rook_to: Optional[str] = None
    promote_to: Optional[str] = None
