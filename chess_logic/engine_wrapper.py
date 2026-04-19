"""UCI engine wrapper for requesting best moves from Stockfish."""

from __future__ import annotations

import os
import random
from typing import Literal, Optional

import chess
import chess.engine

Difficulty = Literal["easy", "medium", "hard"]

# Keep demo tuning knobs in one place.
DIFFICULTY_SETTINGS: dict[Difficulty, dict[str, float | int]] = {
    # Easy: very quick, sometimes picks non-best among top moves.
    "easy": {"time_limit": 0.12, "top_n": 3},
    # Medium: quick but usually sensible.
    "medium": {"time_limit": 0.22, "top_n": 2},
    # Hard: strongest current behavior.
    "hard": {"time_limit": 0.50, "top_n": 1},
}


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

    def get_top_moves(
        self,
        board: chess.Board,
        time_limit: float = 0.5,
        multipv: int = 3,
    ) -> list[dict[str, str]]:
        """Return top candidate moves with simple score metadata.

        Notes:
            - Uses Stockfish multipv analysis.
            - Scores are from the side to move in centipawns (or mate string).
            - Keeps output string-only so callers can pass data directly to LLM prompts.
        """
        analysis = self._engine.analyse(
            board,
            chess.engine.Limit(time=time_limit),
            multipv=max(1, multipv),
        )
        if isinstance(analysis, dict):
            analysis = [analysis]

        candidates: list[dict[str, str]] = []
        for entry in analysis:
            pv = entry.get("pv") or []
            if not pv:
                continue
            move = pv[0]
            if move not in board.legal_moves:
                continue

            score_obj = entry.get("score")
            score_text = "unknown"
            score_cp: int | None = None
            score_mate: int | None = None
            if score_obj is not None:
                pov = score_obj.pov(board.turn)
                mate = pov.mate()
                if mate is not None:
                    score_mate = int(mate)
                    score_text = f"mate {mate}"
                else:
                    cp = pov.score(mate_score=100000)
                    if cp is not None:
                        score_cp = int(cp)
                        score_text = f"{cp} cp"

            candidates.append(
                {
                    "uci": move.uci(),
                    "san": board.san(move),
                    "score": score_text,
                    "score_cp": str(score_cp) if score_cp is not None else "",
                    "score_mate": str(score_mate) if score_mate is not None else "",
                }
            )

        return candidates

    def get_opponent_move(
        self,
        board: chess.Board,
        difficulty: Difficulty = "medium",
    ) -> str:
        """Return a move tuned for demo difficulty.

        Easy/medium modes sample from top candidates for a more human/forgiving feel.
        """
        if difficulty not in DIFFICULTY_SETTINGS:
            difficulty = "medium"

        settings = DIFFICULTY_SETTINGS[difficulty]
        time_limit = float(settings["time_limit"])
        top_n = int(settings["top_n"])

        if top_n <= 1:
            return self.get_best_move(board, time_limit=time_limit)

        candidates = self.get_top_moves(board, time_limit=time_limit, multipv=top_n)
        if not candidates:
            return self.get_best_move(board, time_limit=time_limit)

        # Weighted pick favoring stronger options while allowing variety.
        # Example for 3: weights [3, 2, 1]
        weights = [max(1, top_n - i) for i in range(len(candidates))]
        picked = random.choices(candidates, weights=weights, k=1)[0]
        return str(picked["uci"])

    def close(self) -> None:
        """Gracefully close the engine process."""
        self._engine.quit()
