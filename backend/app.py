from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import shutil
import threading
from typing import Literal

import chess
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from chess_logic.coach import OLLAMA_URL, explain_help_recommendation, get_move_summary
from chess_logic.engine_wrapper import ChessEngine
from chess_logic.game_manager import ChessGame
from chess_logic.move_translator import translate_move
from chess_logic.tts import MODEL_PATH, speak_text

Difficulty = Literal["easy", "medium", "hard"]
PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 100,
}


class MoveRequest(BaseModel):
    uci: str = Field(..., description="Move in UCI format, e.g. e2e4")


class HelpRequest(BaseModel):
    speak: bool = False


class SpeakRequest(BaseModel):
    text: str


class DifficultyRequest(BaseModel):
    level: Difficulty


class VisionBoardRequest(BaseModel):
    source: str = "mock_capture"
    image_b64: str | None = None
    simulate_fen: str | None = None


class GlassesAudioRequest(BaseModel):
    event: str = "coach_output"
    text: str | None = None
    route: str = "glasses_future"


class ChessAppService:
    """In-memory app state for hackathon-friendly local demo."""

    def __init__(self) -> None:
        self.game = ChessGame()
        self.engine = ChessEngine()
        self.lock = threading.Lock()
        self.history: list[dict[str, str | int]] = []
        self.latest_coach_text = ""
        self.last_move_uci: str | None = None
        self.opponent_difficulty: Difficulty = "easy"

    def close(self) -> None:
        self.engine.close()

    def reset(self) -> None:
        with self.lock:
            self.game.reset()
            self.history.clear()
            self.latest_coach_text = ""
            self.last_move_uci = None

    def _piece_symbol_at(self, board: chess.Board, square: chess.Square) -> str | None:
        piece = board.piece_at(square)
        return piece.symbol() if piece else None

    def _board_grid(self) -> list[list[dict[str, str | None]]]:
        board = self.game.get_board()
        grid: list[list[dict[str, str | None]]] = []
        for rank in range(7, -1, -1):
            row = []
            for file_idx in range(8):
                sq = chess.square(file_idx, rank)
                row.append(
                    {
                        "square": chess.square_name(sq),
                        "piece": self._piece_symbol_at(board, sq),
                    }
                )
            grid.append(row)
        return grid

    def _status(self) -> dict[str, str]:
        engine_status = "ready"

        ai_status = "offline"
        try:
            resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=1.5)
            if resp.ok:
                ai_status = "ready"
        except Exception:
            ai_status = "offline"

        piper_ok = shutil.which("piper") is not None
        model_ok = Path(MODEL_PATH).exists() and Path(f"{MODEL_PATH}.json").exists()
        tts_status = "ready" if piper_ok and model_ok else "unavailable"

        return {
            "engine": engine_status,
            "local_ai": ai_status,
            "tts": tts_status,
        }

    def _snapshot(self) -> dict:
        board = self.game.get_board()
        return {
            "fen": self.game.get_fen(),
            "turn": "white" if board.turn == chess.WHITE else "black",
            "is_game_over": self.game.is_game_over(),
            "result": board.result(claim_draw=True) if self.game.is_game_over() else "*",
            "board": self._board_grid(),
            "history": self.history,
            "last_move_uci": self.last_move_uci,
            "latest_coach_text": self.latest_coach_text,
            "opponent_difficulty": self.opponent_difficulty,
            "status": self._status(),
        }

    def _record_move(self, side: str, uci: str, san: str) -> None:
        self.history.append(
            {
                "ply": len(self.history) + 1,
                "side": side,
                "uci": uci,
                "san": san,
            }
        )

    def get_state(self) -> dict:
        with self.lock:
            return self._snapshot()

    def set_difficulty(self, level: Difficulty) -> dict:
        with self.lock:
            self.opponent_difficulty = level
            return self._snapshot()

    def _choose_beginner_friendly_move(
        self,
        board: chess.Board,
        candidates: list[dict[str, str]],
    ) -> dict[str, str]:
        """Fast heuristic ranking over engine candidates for help mode.

        Prefers simple tactical/value ideas (safe captures, checks, threats)
        while staying inside the engine's candidate set.
        """

        def score_candidate(entry: dict[str, str], rank_index: int) -> float:
            uci = entry.get("uci", "")
            move = chess.Move.from_uci(uci)
            if move not in board.legal_moves:
                return -10_000.0

            side = board.turn
            moved_piece = board.piece_at(move.from_square)
            moved_val = PIECE_VALUES.get(moved_piece.piece_type, 0) if moved_piece else 0

            total = 0.0
            # Keep close to engine ordering baseline.
            total += max(0, (len(candidates) - rank_index) * 3)

            # Light use of engine score when available.
            cp_raw = entry.get("score_cp", "")
            if cp_raw:
                try:
                    cp = int(cp_raw)
                    total += max(-8, min(8, cp / 60))
                except ValueError:
                    pass
            mate_raw = entry.get("score_mate", "")
            if mate_raw:
                try:
                    mate = int(mate_raw)
                    if mate > 0:
                        total += 30
                except ValueError:
                    pass

            board_after = board.copy()
            board_after.push(move)

            # 1) Captures: strongly reward safe value captures.
            if board.is_capture(move):
                captured_piece = None
                if board.is_en_passant(move):
                    ep_square = chess.square(chess.square_file(move.to_square), chess.square_rank(move.from_square))
                    captured_piece = board.piece_at(ep_square)
                else:
                    captured_piece = board.piece_at(move.to_square)

                captured_val = PIECE_VALUES.get(captured_piece.piece_type, 0) if captured_piece else 0
                dst_attacked = board_after.is_attacked_by(not side, move.to_square)
                dst_defended = board_after.is_attacked_by(side, move.to_square)

                if not dst_attacked:
                    total += 24 + (captured_val * 3)
                elif dst_defended:
                    total += 12 + max(0, captured_val * 2 - moved_val)
                else:
                    total += max(-2, captured_val - moved_val)

            # 2) Checks.
            if board.gives_check(move):
                total += 16

            # 3/4) Threats and attacks on valuable pieces.
            attacked_values = []
            for sq in board_after.attacks(move.to_square):
                target = board_after.piece_at(sq)
                if target and target.color != side:
                    attacked_values.append(PIECE_VALUES.get(target.piece_type, 0))
            if attacked_values:
                highest = max(attacked_values)
                total += highest * 2
                if highest >= PIECE_VALUES[chess.ROOK]:
                    total += 6

            # 5) Development.
            if moved_piece and moved_piece.piece_type in {chess.KNIGHT, chess.BISHOP}:
                if move.from_square in {
                    chess.B1,
                    chess.G1,
                    chess.C1,
                    chess.F1,
                    chess.B8,
                    chess.G8,
                    chess.C8,
                    chess.F8,
                }:
                    total += 8

            # 6) Center control / space.
            to_sq = chess.square_name(move.to_square)
            if to_sq in {"d4", "e4", "d5", "e5"}:
                total += 6
            elif to_sq in {"c4", "f4", "c5", "f5"}:
                total += 3

            # Encourage stable squares, but keep this light.
            dst_attacked = board_after.is_attacked_by(not side, move.to_square)
            dst_defended = board_after.is_attacked_by(side, move.to_square)
            if not dst_attacked:
                total += 2
            elif not dst_defended:
                total -= 8

            if board.is_castling(move):
                total += 10

            return total

        scored = [
            (score_candidate(entry, idx), entry)
            for idx, entry in enumerate(candidates)
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[0][1] if scored else candidates[0]

    def play_move(self, uci: str) -> dict:
        with self.lock:
            board = self.game.get_board()
            if not self.game.is_legal_move(uci):
                raise HTTPException(status_code=400, detail="Illegal move for current position.")

            user_move = chess.Move.from_uci(uci)
            user_side = "White" if board.turn == chess.WHITE else "Black"
            user_san = board.san(user_move)
            user_command = asdict(translate_move(board, uci))

            self.game.apply_move(uci)
            self.last_move_uci = uci
            self._record_move(user_side, uci, user_san)

            engine_payload: dict[str, object] | None = None
            if not self.game.is_game_over():
                board_after = self.game.get_board()
                engine_uci = self.engine.get_opponent_move(
                    board_after,
                    difficulty=self.opponent_difficulty,
                )
                engine_move = chess.Move.from_uci(engine_uci)
                engine_side = "White" if board_after.turn == chess.WHITE else "Black"
                engine_san = board_after.san(engine_move)
                engine_command = asdict(translate_move(board_after, engine_uci))

                self.game.apply_move(engine_uci)
                self.last_move_uci = engine_uci
                self._record_move(engine_side, engine_uci, engine_san)

                engine_payload = {
                    "uci": engine_uci,
                    "san": engine_san,
                    "command": engine_command,
                }

            return {
                **self._snapshot(),
                "move": {
                    "uci": uci,
                    "san": user_san,
                    "command": user_command,
                },
                "engine_move": engine_payload,
            }

    def get_help(self, speak: bool = False) -> dict:
        with self.lock:
            board = self.game.get_board()
            if self.game.is_game_over():
                raise HTTPException(status_code=400, detail="Game is already over.")

            # Keep help responsive: slightly stronger than easy opponent, still fast.
            candidates = self.engine.get_top_moves(board, time_limit=0.45, multipv=3)
            if not candidates:
                best = self.engine.get_best_move(board, time_limit=0.5)
                candidates = [{"uci": best, "san": board.san(chess.Move.from_uci(best)), "score": "unknown"}]
            best = self._choose_beginner_friendly_move(board, candidates)
            summary = get_move_summary(board, best["uci"])
            summary.update(
                {
                    "uci": best["uci"],
                    "san": best.get("san", best["uci"]),
                    "score": best.get("score", "unknown"),
                }
            )

            advice = explain_help_recommendation(self.game.get_fen(), summary, candidates)
            self.latest_coach_text = advice

            if speak:
                speak_text(advice)

            return {
                **self._snapshot(),
                "coach_recommendation": advice,
                "help": {
                    "best": best,
                    "candidates": candidates,
                },
            }


service = ChessAppService()

app = FastAPI(title="Chess Tutor Local API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/state")
def get_state() -> dict:
    return service.get_state()


@app.post("/move")
def post_move(payload: MoveRequest) -> dict:
    return service.play_move(payload.uci.strip().lower())


@app.post("/help")
def post_help(payload: HelpRequest) -> dict:
    return service.get_help(speak=payload.speak)


@app.post("/speak")
def post_speak(payload: SpeakRequest) -> dict:
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is empty.")
    speak_text(text)
    return {"ok": True}


@app.post("/reset")
def post_reset() -> dict:
    service.reset()
    return service.get_state()


@app.post("/difficulty")
def post_difficulty(payload: DifficultyRequest) -> dict:
    return service.set_difficulty(payload.level)


@app.post("/vision/board")
def post_vision_board(payload: VisionBoardRequest) -> dict:
    """Placeholder endpoint for future glasses camera frame ingestion.

    Current demo behavior:
    - keeps existing app flow unchanged
    - echoes a mock recognition result using current board FEN
    """
    recognized_fen = service.game.get_fen()
    if payload.simulate_fen:
        try:
            chess.Board(payload.simulate_fen)
            recognized_fen = payload.simulate_fen
        except ValueError:
            pass

    return {
        "ok": True,
        "mode": "meta_glasses_placeholder",
        "source": payload.source,
        "recognized_fen": recognized_fen,
        "message": (
            "Vision placeholder active. In a future Meta glasses integration, "
            "camera frames would be parsed here and converted into board state."
        ),
    }


@app.post("/glasses/audio")
def post_glasses_audio(payload: GlassesAudioRequest) -> dict:
    """Placeholder endpoint for future glasses audio routing.

    Current demo behavior:
    - does not require a real SDK
    - returns route metadata for UI status display
    """
    return {
        "ok": True,
        "event": payload.event,
        "route": payload.route,
        "preview_text": payload.text or "",
        "message": (
            "Audio placeholder active. In a future Meta glasses integration, "
            "coach speech would be routed to glasses audio output here."
        ),
    }
