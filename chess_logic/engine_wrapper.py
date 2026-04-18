"""UCI engine wrapper for requesting best moves from Stockfish."""

from __future__ import annotations

import os
from typing import Optional

import chess
import chess.engine


class ChessEngine:
    """Small wrapper around a UCI-compatible engine process."""

    def __init__(self, engine_path: Optional[str] = None) -> None:
        self.engine_path = engine_path or os.getenv("STOCKFISH_PATH", "stockfish")
        self._engine = chess.engine.SimpleEngine.popen_uci(self.engine_path)

    def get_best_move(self, board: chess.Board, time_limit: float = 0.5) -> str:
        """Return the engine's best move in UCI format."""
        result = self._engine.play(board, chess.engine.Limit(time=time_limit))
        if result.move is None:
            raise RuntimeError("Engine did not return a move.")
        return result.move.uci()

    def close(self) -> None:
        """Gracefully close the engine process."""
        self._engine.quit()
