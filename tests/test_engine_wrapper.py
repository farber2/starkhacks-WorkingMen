"""Tests for UCI engine wrapper integration."""

from __future__ import annotations

import os
import shutil

import chess
import pytest

from chess_logic.engine_wrapper import ChessEngine


def test_get_best_move_returns_uci() -> None:
    configured = os.getenv("STOCKFISH_PATH")
    candidate = configured or shutil.which("stockfish")

    if candidate is None:
        pytest.skip("Stockfish binary not found; set STOCKFISH_PATH to run this test.")

    engine = ChessEngine(engine_path=candidate)
    board = chess.Board()

    try:
        best_move = engine.get_best_move(board, time_limit=0.05)
        parsed = chess.Move.from_uci(best_move)
        assert parsed in board.legal_moves
    finally:
        engine.close()
