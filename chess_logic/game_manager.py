"""Game state management built on top of python-chess."""

from __future__ import annotations

import chess


class ChessGame:
    """Manages chess board state and legal move application."""

    def __init__(self) -> None:
        self._board = chess.Board()

    def reset(self) -> None:
        """Reset the game to the initial starting position."""
        self._board.reset()

    def is_legal_move(self, uci_move: str) -> bool:
        """Return True if the given UCI move is legal in the current position."""
        try:
            move = chess.Move.from_uci(uci_move)
        except ValueError:
            return False
        return move in self._board.legal_moves

    def apply_move(self, uci_move: str) -> bool:
        """Apply a legal UCI move and return True on success."""
        if not self.is_legal_move(uci_move):
            return False
        self._board.push(chess.Move.from_uci(uci_move))
        return True

    def get_fen(self) -> str:
        """Get the board state as a FEN string."""
        return self._board.fen()

    def is_game_over(self) -> bool:
        """Return True when the game has reached a terminal state."""
        return self._board.is_game_over()

    def get_board(self) -> chess.Board:
        """Expose the underlying board object for advanced integrations."""
        return self._board

    def get_board_with_coordinates(self) -> str:
        """Return an ASCII board annotated with rank and file coordinates."""
        board_lines = str(self._board).splitlines()
        numbered_lines = [f"{8 - idx} {line}" for idx, line in enumerate(board_lines)]
        files = "  a b c d e f g h"
        return "\n".join(numbered_lines + [files])
