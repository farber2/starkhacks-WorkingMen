"""Tests for castling, en passant, and promotion translations."""

from __future__ import annotations

import chess

from chess_logic.move_translator import translate_move


def test_translate_kingside_castle() -> None:
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")

    cmd = translate_move(board, "e1g1")

    assert cmd.action_type == "castle_kingside"
    assert cmd.king_from == "e1"
    assert cmd.king_to == "g1"
    assert cmd.rook_from == "h1"
    assert cmd.rook_to == "f1"


def test_translate_en_passant() -> None:
    board = chess.Board("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1")

    cmd = translate_move(board, "e5d6")

    assert cmd.action_type == "en_passant"
    assert cmd.from_square == "e5"
    assert cmd.to_square == "d6"
    assert cmd.remove_square == "d5"


def test_translate_promotion() -> None:
    board = chess.Board("7k/4P3/8/8/8/8/8/4K3 w - - 0 1")

    cmd = translate_move(board, "e7e8q")

    assert cmd.action_type == "promotion"
    assert cmd.from_square == "e7"
    assert cmd.to_square == "e8"
    assert cmd.promote_to == "queen"
