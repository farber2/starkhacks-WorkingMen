"""Translation from chess UCI moves to robot-ready command schema."""

from __future__ import annotations

import chess

from chess_logic.command_schema import RobotMoveCommand
from chess_logic.move_info import get_move_info


def translate_move(board: chess.Board, uci_move: str) -> RobotMoveCommand:
    """Convert a legal UCI move into a structured robot command."""
    move = chess.Move.from_uci(uci_move)
    info = get_move_info(board, uci_move)

    if info["is_castling"]:
        if chess.square_file(move.to_square) == 6:
            rank = "1" if board.turn == chess.WHITE else "8"
            return RobotMoveCommand(
                action_type="castle_kingside",
                king_from=f"e{rank}",
                king_to=f"g{rank}",
                rook_from=f"h{rank}",
                rook_to=f"f{rank}",
            )

        rank = "1" if board.turn == chess.WHITE else "8"
        return RobotMoveCommand(
            action_type="castle_queenside",
            king_from=f"e{rank}",
            king_to=f"c{rank}",
            rook_from=f"a{rank}",
            rook_to=f"d{rank}",
        )

    if info["is_en_passant"]:
        remove_rank_delta = -1 if board.turn == chess.WHITE else 1
        remove_rank = int(info["to_square"][1]) + remove_rank_delta
        return RobotMoveCommand(
            action_type="en_passant",
            from_square=info["from_square"],
            to_square=info["to_square"],
            remove_square=f"{info['to_square'][0]}{remove_rank}",
        )

    if info["is_promotion"]:
        return RobotMoveCommand(
            action_type="promotion",
            from_square=info["from_square"],
            to_square=info["to_square"],
            promote_to=info["promotion_piece"],
        )

    if info["is_capture"]:
        return RobotMoveCommand(
            action_type="capture_piece",
            from_square=info["from_square"],
            to_square=info["to_square"],
            remove_square=info["to_square"],
        )

    return RobotMoveCommand(
        action_type="move_piece",
        from_square=info["from_square"],
        to_square=info["to_square"],
    )
