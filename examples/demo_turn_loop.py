"""Example script showing one automated player-engine turn flow."""

from __future__ import annotations

from chess_logic.coach import explain_move
from chess_logic.engine_wrapper import ChessEngine
from chess_logic.game_manager import ChessGame
from chess_logic.move_translator import translate_move


def run_demo() -> None:
    """Play a few scripted user moves against the engine and print commands."""
    game = ChessGame()
    engine = ChessEngine()
    scripted_user_moves = ["e2e4", "g1f3", "f1c4"]

    try:
        for user_move in scripted_user_moves:
            if game.is_game_over() or not game.is_legal_move(user_move):
                break

            print("\nCurrent board:")
            print(game.get_board_with_coordinates())
            print(f"User move: {user_move}")
            user_fen_before = game.get_fen()
            user_explanation = explain_move(user_fen_before, user_move)
            print(f"User command: {translate_move(game.get_board(), user_move)}")
            print(
                "Coach (your move): "
                f"{user_explanation}"
            )
            game.apply_move(user_move)

            if game.is_game_over():
                break

            ai_move = engine.get_best_move(game.get_board(), time_limit=0.3)
            print(f"Engine move: {ai_move}")
            print(f"Engine command: {translate_move(game.get_board(), ai_move)}")
            game.apply_move(ai_move)
    finally:
        engine.close()


if __name__ == "__main__":
    run_demo()
