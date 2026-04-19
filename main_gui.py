"""Tkinter GUI for the chess project using existing backend logic classes."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox

import chess

from chess_logic.engine_wrapper import ChessEngine
from chess_logic.game_manager import ChessGame


class ChessGUI:
    """Simple click-to-move chess GUI (human as White, engine as Black)."""

    LIGHT_SQUARE = "#F0D9B5"
    DARK_SQUARE = "#B58863"
    SELECTED_SQUARE = "#F6F669"

    PIECE_UNICODE = {
        "P": "♙",
        "N": "♘",
        "B": "♗",
        "R": "♖",
        "Q": "♕",
        "K": "♔",
        "p": "♟",
        "n": "♞",
        "b": "♝",
        "r": "♜",
        "q": "♛",
        "k": "♚",
    }

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Chess Robot GUI")

        self.game = ChessGame()
        self.engine = ChessEngine()

        # Tracks currently selected square for click-to-move.
        self.selected_square: str | None = None

        self.status_var = tk.StringVar(value="Your turn (White)")
        self.buttons: dict[str, tk.Button] = {}

        self._build_ui()
        self._draw_board()

        # Ensure engine process is closed when the window exits.
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        """Create board, labels, and controls."""
        board_frame = tk.Frame(self.root, padx=10, pady=10)
        board_frame.pack()

        files = "abcdefgh"

        # Add file labels on bottom: a-h.
        tk.Label(board_frame, text=" ").grid(row=8, column=0)
        for col, file_char in enumerate(files, start=1):
            tk.Label(board_frame, text=file_char, font=("Helvetica", 11, "bold")).grid(
                row=8, column=col, padx=2, pady=2
            )

        # Add rank labels on the left and 8x8 square buttons.
        for row in range(8):
            rank = str(8 - row)
            tk.Label(board_frame, text=rank, font=("Helvetica", 11, "bold")).grid(
                row=row, column=0, padx=2, pady=2
            )

            for col in range(8):
                square = self._square_name(row, col)
                btn = tk.Button(
                    board_frame,
                    text="",
                    width=3,
                    height=1,
                    font=("Helvetica", 24),
                    relief="raised",
                    command=lambda sq=square: self._on_square_click(sq),
                )
                btn.grid(row=row, column=col + 1, sticky="nsew")
                self.buttons[square] = btn

        status_label = tk.Label(self.root, textvariable=self.status_var, font=("Helvetica", 12))
        status_label.pack(pady=(0, 8))

        reset_button = tk.Button(self.root, text="Reset Game", command=self._reset_game)
        reset_button.pack(pady=(0, 12))

    def _square_name(self, row: int, col: int) -> str:
        """Convert UI row/col to chess square name (e.g., row 0 col 0 -> a8)."""
        file_char = chr(ord("a") + col)
        rank = str(8 - row)
        return f"{file_char}{rank}"

    def _draw_board(self) -> None:
        """Redraw all board squares based on current game state."""
        board = self.game.get_board()

        for row in range(8):
            for col in range(8):
                square_name = self._square_name(row, col)
                square_index = chess.parse_square(square_name)
                piece = board.piece_at(square_index)

                base_color = self.LIGHT_SQUARE if (row + col) % 2 == 0 else self.DARK_SQUARE
                bg = self.SELECTED_SQUARE if square_name == self.selected_square else base_color

                text = ""
                if piece is not None:
                    text = self.PIECE_UNICODE[piece.symbol()]

                self.buttons[square_name].configure(text=text, bg=bg, activebackground=bg)

    def _on_square_click(self, square_name: str) -> None:
        """Handle selecting a piece and then selecting destination square."""
        if self.game.is_game_over():
            return

        board = self.game.get_board()

        # Human player is White in this first version.
        if board.turn != chess.WHITE:
            self.status_var.set("Please wait for engine move...")
            return

        if self.selected_square is None:
            piece = board.piece_at(chess.parse_square(square_name))
            if piece is None:
                self.status_var.set("Select a white piece first.")
                return
            if piece.color != chess.WHITE:
                self.status_var.set("You can only move white pieces.")
                return

            self.selected_square = square_name
            self.status_var.set(f"Selected {square_name}. Choose destination.")
            self._draw_board()
            return

        if square_name == self.selected_square:
            self.selected_square = None
            self.status_var.set("Selection cleared. Your turn (White).")
            self._draw_board()
            return

        uci_move = self._build_uci_move(self.selected_square, square_name)
        self.selected_square = None

        if not self.game.is_legal_move(uci_move):
            self.status_var.set(f"Illegal move: {uci_move}")
            self._draw_board()
            return

        self.game.apply_move(uci_move)
        self._draw_board()

        if self._handle_game_over_if_needed():
            return

        self._play_engine_turn()

    def _build_uci_move(self, from_square: str, to_square: str) -> str:
        """Create UCI move string and auto-promote pawns to queen when needed."""
        board = self.game.get_board()
        uci_move = f"{from_square}{to_square}"

        piece = board.piece_at(chess.parse_square(from_square))
        if piece and piece.piece_type == chess.PAWN:
            to_rank = int(to_square[1])
            if (piece.color == chess.WHITE and to_rank == 8) or (
                piece.color == chess.BLACK and to_rank == 1
            ):
                uci_move += "q"

        return uci_move

    def _play_engine_turn(self) -> None:
        """Ask engine for best move, apply it, and refresh UI."""
        self.status_var.set("Engine is thinking...")
        self.root.update_idletasks()

        try:
            ai_move = self.engine.get_best_move(self.game.get_board(), time_limit=0.5)
            if not self.game.apply_move(ai_move):
                self.status_var.set(f"Engine produced illegal move: {ai_move}")
                return
        except Exception as exc:  # Keep GUI alive if engine call fails.
            self.status_var.set("Engine error. See popup for details.")
            messagebox.showerror("Engine Error", str(exc))
            return

        self._draw_board()
        if self._handle_game_over_if_needed():
            return

        self.status_var.set("Your turn (White)")

    def _handle_game_over_if_needed(self) -> bool:
        """Show result when game is over. Returns True if terminal."""
        if not self.game.is_game_over():
            return False

        result = self.game.get_board().result(claim_draw=True)
        self.status_var.set(f"Game Over: {result}")
        messagebox.showinfo("Game Over", f"Result: {result}")
        return True

    def _reset_game(self) -> None:
        """Reset game state and board UI for a new game."""
        self.game.reset()
        self.selected_square = None
        self.status_var.set("Your turn (White)")
        self._draw_board()

    def _on_close(self) -> None:
        """Cleanly close engine process, then close window."""
        try:
            self.engine.close()
        except Exception:
            pass
        self.root.destroy()


def main() -> None:
    """Run the Tkinter chess GUI."""
    root = tk.Tk()

    try:
        ChessGUI(root)
    except Exception as exc:
        messagebox.showerror(
            "Startup Error",
            "Could not start Chess GUI. Make sure Stockfish is installed and available.\n\n"
            f"Details: {exc}",
        )
        root.destroy()
        return

    root.mainloop()


if __name__ == "__main__":
    main()
