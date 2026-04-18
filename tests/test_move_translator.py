"""Tests for standard and capture move translations."""

from __future__ import annotations

import chess

from chess_logic.move_translator import translate_move


def test_translate_normal_move() -> None:
    board = chess.Board()
    cmd = translate_move(board, "e2e4")

    assert cmd.action_type == "move_piece"
    assert cmd.from_square == "e2"
    assert cmd.to_square == "e4"


def test_translate_capture_move() -> None:
    board = chess.Board()
    board.push_uci("e2e4")
    board.push_uci("d7d5")

    cmd = translate_move(board, "e4d5")

    assert cmd.action_type == "capture_piece"
    assert cmd.from_square == "e4"
    assert cmd.to_square == "d5"
    assert cmd.remove_square == "d5"
