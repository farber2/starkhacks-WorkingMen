"""Infer likely legal move from noisy detected board occupancy."""

from __future__ import annotations

from dataclasses import dataclass

import chess

from chess_logic.board_map import board_to_square_occupancy


@dataclass
class CandidateMoveScore:
    uci: str
    san: str
    score: float
    change_overlap: float
    mismatch_count: int
    matched_squares: int


def _changed_squares(
    prev_occ: dict[str, str | None],
    detected_occ: dict[str, str | None],
) -> set[str]:
    changed = set()
    for sq in prev_occ:
        if prev_occ.get(sq) != detected_occ.get(sq):
            changed.add(sq)
    return changed


def _score_move(
    board: chess.Board,
    move: chess.Move,
    detected_occ: dict[str, str | None],
    detected_conf: dict[str, float] | None = None,
) -> CandidateMoveScore:
    detected_conf = detected_conf or {}
    sim = board.copy(stack=False)
    san = board.san(move)
    sim.push(move)
    sim_occ = board_to_square_occupancy(sim)
    prev_occ = board_to_square_occupancy(board)

    mismatch_count = 0
    matched_squares = 0
    score = 0.0
    for sq, sim_piece in sim_occ.items():
        seen_piece = detected_occ.get(sq)
        conf = float(detected_conf.get(sq, 0.3))
        if sim_piece == seen_piece:
            matched_squares += 1
            score += 1.0 + conf
        else:
            mismatch_count += 1
            score -= 0.8 + (0.4 * conf)

    sim_changed = _changed_squares(prev_occ, sim_occ)
    det_changed = _changed_squares(prev_occ, detected_occ)

    # Prefer candidates whose change pattern overlaps observed change pattern.
    intersection = sim_changed.intersection(det_changed)
    union = sim_changed.union(det_changed) or {move.uci()[:2], move.uci()[2:4]}
    overlap = len(intersection) / len(union)
    score += overlap * 12.0

    # Practical tactical priors for beginner-facing move detection.
    if board.is_capture(move):
        score += 2.5
    if board.gives_check(move):
        score += 2.0
    if board.is_castling(move):
        score += 1.5

    return CandidateMoveScore(
        uci=move.uci(),
        san=san,
        score=score,
        change_overlap=overlap,
        mismatch_count=mismatch_count,
        matched_squares=matched_squares,
    )


def infer_legal_move_from_detection(
    previous_board: chess.Board,
    detected_occupancy: dict[str, str | None],
    detected_confidence: dict[str, float] | None = None,
    top_k: int = 8,
) -> tuple[str | None, float, list[dict]]:
    """Rank legal moves by consistency with detected occupancy.

    Returns:
    - best move UCI or None
    - best score
    - ranked candidate list (for debug UI)
    """
    legal_moves = list(previous_board.legal_moves)
    if not legal_moves:
        return None, 0.0, []

    scored = [
        _score_move(previous_board, move, detected_occupancy, detected_confidence)
        for move in legal_moves
    ]
    scored.sort(key=lambda item: item.score, reverse=True)

    top = scored[:top_k]
    candidates = [
        {
            "uci": c.uci,
            "san": c.san,
            "score": round(c.score, 3),
            "change_overlap": round(c.change_overlap, 3),
            "mismatch_count": c.mismatch_count,
            "matched_squares": c.matched_squares,
        }
        for c in top
    ]

    best = top[0] if top else None
    return (best.uci if best else None), (best.score if best else 0.0), candidates

