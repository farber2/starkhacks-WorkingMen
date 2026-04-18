"""CLI loop for user-vs-engine move generation and translation."""

from __future__ import annotations

from chess_logic.engine_wrapper import ChessEngine
from chess_logic.game_manager import ChessGame
from chess_logic.move_translator import translate_move


def main() -> None:
    """Run an interactive command-line chess loop using UCI move input."""
    game = ChessGame()
    engine = ChessEngine()

    print("Physical Chess Robot Controller (logic only)")
    print("Enter UCI moves like e2e4. Type 'quit' to exit.")

    try:
        while not game.is_game_over():
            print("\nCurrent board:")
            print(game.get_board_with_coordinates())

            user_move = input("Your move (UCI): ").strip().lower()
            if user_move in {"quit", "exit"}:
                print("Session ended by user.")
                break

            if not game.is_legal_move(user_move):
                print("Illegal move. Try again.")
                continue

            user_command = translate_move(game.get_board(), user_move)
            print(f"Robot command (player): {user_command}")
            game.apply_move(user_move)

            if game.is_game_over():
                break

            ai_move = engine.get_best_move(game.get_board(), time_limit=0.5)
            ai_command = translate_move(game.get_board(), ai_move)
            print(f"Engine move: {ai_move}")
            print(f"Robot command (engine): {ai_command}")
            game.apply_move(ai_move)

    finally:
        engine.close()

    print(f"Game over status: {game.get_board().result(claim_draw=True)}")


if __name__ == "__main__":
    main()
