"""CLI loop for user-vs-engine move generation and translation."""

from __future__ import annotations

import os

import chess

from chess_logic.coach import explain_help_recommendation, get_move_summary
from chess_logic.engine_wrapper import ChessEngine
from chess_logic.game_manager import ChessGame
from chess_logic.move_translator import translate_move
from chess_logic.tts import speak_text

ENABLE_TTS = os.getenv("CHESS_TTS", "0").strip() == "1"


def get_help_recommendation(
    board: chess.Board,
    engine: ChessEngine,
) -> tuple[dict[str, str], list[dict[str, str]]]:
    """Return best move summary plus top candidates for help mode.

    Adaptation note:
    - If your engine wrapper already returns richer analysis, map that output into
      the same fields used below: uci/san/score + python-computed move facts.
    """
    candidates = engine.get_top_moves(board, time_limit=0.6, multipv=3)
    if not candidates:
        best_move = engine.get_best_move(board, time_limit=0.5)
        candidates = [{"uci": best_move, "san": board.san(chess.Move.from_uci(best_move)), "score": "unknown"}]

    best = candidates[0]
    summary = get_move_summary(board, best["uci"])
    summary.update(
        {
            "uci": best["uci"],
            "san": best.get("san", best["uci"]),
            "score": best.get("score", "unknown"),
        }
    )
    return summary, candidates


def main() -> None:
    """Run an interactive command-line chess loop using UCI move input."""
    game = ChessGame()
    engine = ChessEngine()

    print("Physical Chess Robot Controller (logic only)")
    print("Enter UCI moves like e2e4. Type 'help' for coaching, 'quit' to exit.")

    try:
        while not game.is_game_over():
            print("\nCurrent board:")
            print(game.get_board_with_coordinates())

            user_move = input("Your move (UCI): ").strip().lower()
            if user_move in {"quit", "exit"}:
                print("Session ended by user.")
                break

            if user_move == "help":
                best_summary, candidates = get_help_recommendation(game.get_board(), engine)
                advice = explain_help_recommendation(
                    game.get_fen(),
                    best_summary,
                    candidates,
                )
                print(f"Coach (help): {advice}")
                if ENABLE_TTS:
                    speak_text(advice)
                continue

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
