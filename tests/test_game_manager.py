"""Tests for ChessGame state and legal move handling."""

from __future__ import annotations

import chess

from chess_logic.game_manager import ChessGame


def test_is_legal_move_handles_valid_invalid_and_malformed_input() -> None:
    game = ChessGame()

    assert game.is_legal_move("e2e4") is True
    assert game.is_legal_move("e2e5") is False
    assert game.is_legal_move("badmove") is False


def test_apply_move_updates_board_for_legal_move() -> None:
    game = ChessGame()

    assert game.apply_move("e2e4") is True
    piece = game.get_board().piece_at(chess.parse_square("e4"))
    assert piece is not None
    assert piece.piece_type == chess.PAWN
    assert piece.color == chess.WHITE


def test_apply_move_rejects_illegal_move() -> None:
    game = ChessGame()

    original_fen = game.get_fen()
    assert game.apply_move("e2e5") is False
    assert game.get_fen() == original_fen


def test_reset_restores_initial_state() -> None:
    game = ChessGame()
    initial_fen = game.get_fen()

    game.apply_move("e2e4")
    assert game.get_fen() != initial_fen

    game.reset()
    assert game.get_fen() == initial_fen


def test_get_board_returns_board_instance() -> None:
    game = ChessGame()

    board = game.get_board()
    assert isinstance(board, chess.Board)


def test_is_game_over_detects_terminal_position() -> None:
    game = ChessGame()

    assert game.is_game_over() is False
    game.apply_move("f2f3")
    game.apply_move("e7e5")
    game.apply_move("g2g4")
    game.apply_move("d8h4")
    assert game.is_game_over() is True


def test_get_board_with_coordinates_includes_files_and_ranks() -> None:
    game = ChessGame()

    rendered = game.get_board_with_coordinates()
    assert "8 " in rendered
    assert "1 " in rendered
    assert "a b c d e f g h" in rendered
