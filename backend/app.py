from __future__ import annotations

import base64
from dataclasses import asdict
import glob
import os
from pathlib import Path
import shutil
import threading
import time
from typing import Literal

import chess
import cv2
import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from chess_logic.board_map import (
    board_to_square_occupancy,
    draw_square_overlay,
    format_board_map,
    get_square_map,
    map_detections_to_squares,
)
from chess_logic.board_vision import normalize_board_image, warp_board
from chess_logic.coach import OLLAMA_URL, explain_help_recommendation, get_move_summary
from chess_logic.engine_wrapper import ChessEngine
from chess_logic.game_manager import ChessGame
from chess_logic.move_reconstructor import infer_legal_move_from_detection
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


class VisionMoveRequest(BaseModel):
    image_b64: str
    previous_fen: str | None = None
    update_game_state: bool = False
    source: str = "camera_frame"


class VisionProcessRequest(BaseModel):
    image_b64: str | None = None
    source: str = "camera_frame"
    camera_index: int = 0
    confidence_threshold: float = 1.0
    activity_threshold: float = 2.5
    min_streak: int = 1
    motion_squares_threshold: int = 14
    settle_frames: int = 1
    max_changed_squares_for_move: int = 12


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
        self._vision_detector = None
        self._vision_camera = None
        self._vision_camera_index: int | None = None
        self.latest_vision_debug: dict = {}
        self.vision_last_move_uci: str | None = None
        self.vision_last_confidence: float = 0.0
        self.vision_changed_flag: bool = False
        self.vision_last_accepted_at: float = 0.0
        self.vision_last_accepted_key: str = ""
        self.vision_last_frame_occupancy: dict[str, str | None] | None = None
        self.vision_last_warped_gray: np.ndarray | None = None
        self.vision_pending_move_uci: str | None = None
        self.vision_pending_streak: int = 0
        self.vision_motion_active: bool = False
        self.vision_settle_streak: int = 0
        self.vision_locked_corners: list[tuple[float, float]] | None = None

    @property
    def vision_model_path(self) -> str:
        env_path = os.getenv("YOLO_MODEL_PATH", "").strip()
        if env_path and Path(env_path).exists():
            return env_path

        default_path = Path("runs/detect/runs/chess_pieces_white_tuned/weights/best.pt")
        if default_path.exists():
            return str(default_path)

        # Auto-fallback to newest best.pt in runs/ (hackathon-friendly).
        candidates = sorted(
            glob.glob("runs/**/weights/best.pt", recursive=True),
            key=lambda p: Path(p).stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return candidates[0]

        # Common ad-hoc location used during quick model swaps.
        if Path("bestp2.pt").exists():
            return "bestp2.pt"
        if Path("models/chess_best.pt").exists():
            return "models/chess_best.pt"
        return str(default_path)

    def _get_vision_detector(self):
        """Lazy-load YOLO detector so startup remains fast."""
        if self._vision_detector is not None:
            return self._vision_detector
        try:
            from detect import ChessDetector  # lazy import to keep optional

            print(f"[vision] loading YOLO model: {self.vision_model_path}")
            self._vision_detector = ChessDetector(
                model_path=self.vision_model_path,
                confidence_threshold=0.10,
                image_size=1280,
            )
            return self._vision_detector
        except Exception as exc:
            print(f"[vision] failed to load YOLO model: {exc}")
            return None

    def _decode_image(self, image_b64: str) -> np.ndarray:
        if not image_b64:
            raise HTTPException(status_code=400, detail="image_b64 is required.")
        payload = image_b64.split(",", 1)[1] if "," in image_b64 else image_b64
        try:
            data = base64.b64decode(payload)
            arr = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid image payload: {exc}") from exc
        if frame is None:
            raise HTTPException(status_code=400, detail="Could not decode image data.")
        return frame

    def _capture_camera_frame(self, camera_index: int) -> np.ndarray:
        try:
            from camera import Camera
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"camera.py unavailable: {exc}") from exc

        if self._vision_camera is None or self._vision_camera_index != camera_index:
            if self._vision_camera is not None:
                self._vision_camera.release()
            self._vision_camera = Camera(index=camera_index, width=1920, height=1080, fourcc="MJPG")
            self._vision_camera.open()
            self._vision_camera_index = camera_index
        return self._vision_camera.read()

    def _serialize_square_map(self, square_map):
        return {
            square: {
                "bbox": list(region.bbox),
                "center": list(region.center),
                "polygon": [list(p) for p in region.polygon],
                "row": region.row,
                "col": region.col,
            }
            for square, region in square_map.items()
        }

    def _run_vision_pipeline(self, frame: np.ndarray, include_yolo: bool = True) -> dict:
        """Run board normalization + square mapping.

        YOLO can be skipped for fast/robust frame-diff move detection.
        """
        if self.vision_locked_corners is None:
            normalized = normalize_board_image(frame, size=800)
            self.vision_locked_corners = list(normalized.corners)
        else:
            h, w = frame.shape[:2]
            warped = warp_board(frame, self.vision_locked_corners, size=800)
            normalized = type("BoardVisionShim", (), {
                "warped_board": warped,
                "corners": self.vision_locked_corners,
                "source_size": (w, h),
                "warped_size": 800,
            })()
        warped = normalized.warped_board
        square_map = get_square_map(warped)
        overlay = draw_square_overlay(warped, square_map)

        detections: list[dict] = []
        warnings: list[str] = []
        if include_yolo:
            detector = self._get_vision_detector()
            if detector is None:
                warnings.append("YOLO detector unavailable or model path missing.")
            else:
                try:
                    _, detections = detector.detect(warped)
                    # Fallback pass on full frame if warped board gave nothing.
                    if not detections:
                        _, detections = detector.detect(frame)
                except Exception as exc:
                    warnings.append(f"YOLO detection failed: {exc}")
        else:
            warnings.append("YOLO disabled for pixel-diff move detection mode.")

        occupancy, confidences, assignments = map_detections_to_squares(detections, square_map)
        mapped_piece_count = sum(1 for v in occupancy.values() if v is not None)

        return {
            "normalized": normalized,
            "warped": warped,
            "overlay": overlay,
            "square_map": square_map,
            "square_map_serialized": self._serialize_square_map(square_map),
            "detections": detections,
            "occupancy": occupancy,
            "confidences": confidences,
            "assignments": assignments,
            "mapped_piece_count": mapped_piece_count,
            "warnings": warnings,
        }

    def close(self) -> None:
        self.engine.close()
        if self._vision_camera is not None:
            self._vision_camera.release()
            self._vision_camera = None

    def reset(self) -> None:
        with self.lock:
            self.game.reset()
            self.history.clear()
            self.latest_coach_text = ""
            self.last_move_uci = None
            self.vision_last_move_uci = None
            self.vision_last_confidence = 0.0
            self.vision_changed_flag = False
            self.vision_last_accepted_at = 0.0
            self.vision_last_accepted_key = ""
            self.vision_last_frame_occupancy = None
            self.vision_last_warped_gray = None
            self.vision_pending_move_uci = None
            self.vision_pending_streak = 0
            self.vision_motion_active = False
            self.vision_settle_streak = 0
            self.vision_locked_corners = None

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
            "changed": self.vision_changed_flag,
            "vision_last_move_uci": self.vision_last_move_uci,
            "vision_last_confidence": round(float(self.vision_last_confidence), 3),
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

    def vision_board(self, payload: VisionBoardRequest) -> dict:
        """Detect board + 64-square mapping + occupancy from a frame."""
        with self.lock:
            if not payload.image_b64:
                # Keep backward compatibility with old placeholder behavior.
                recognized_fen = self.game.get_fen()
                if payload.simulate_fen:
                    try:
                        chess.Board(payload.simulate_fen)
                        recognized_fen = payload.simulate_fen
                    except ValueError:
                        pass
                return {
                    "ok": True,
                    "source": payload.source,
                    "recognized_fen": recognized_fen,
                    "message": "No image provided. Send image_b64 for real board detection.",
                }

            frame = self._decode_image(payload.image_b64)
            vision = self._run_vision_pipeline(frame)
            board_text = format_board_map(
                [
                    [vision["occupancy"].get(f"{file_char}{rank}") or "." for file_char in "abcdefgh"]
                    for rank in range(8, 0, -1)
                ]
            )

            debug = {
                "source": payload.source,
                "board_corners": [list(p) for p in vision["normalized"].corners],
                "source_size": list(vision["normalized"].source_size),
                "warped_size": vision["normalized"].warped_size,
                "square_map": vision["square_map_serialized"],
                "raw_detections": vision["detections"],
                "mapped_occupancy": vision["occupancy"],
                "square_confidences": vision["confidences"],
                "assignments": vision["assignments"],
                "board_text": board_text,
                "warnings": vision["warnings"],
            }
            self.latest_vision_debug = debug
            return {"ok": True, **debug}

    def vision_move(self, payload: VisionMoveRequest) -> dict:
        """Infer most likely legal move from current camera frame."""
        with self.lock:
            frame = self._decode_image(payload.image_b64)
            prev_board = chess.Board(payload.previous_fen) if payload.previous_fen else self.game.get_board().copy(stack=False)

            vision = self._run_vision_pipeline(frame)
            best_uci, best_score, candidates = infer_legal_move_from_detection(
                previous_board=prev_board,
                detected_occupancy=vision["occupancy"],
                detected_confidence=vision["confidences"],
                top_k=10,
            )

            applied = False
            if payload.update_game_state and best_uci:
                if self.game.get_fen() == prev_board.fen() and self.game.is_legal_move(best_uci):
                    self.game.apply_move(best_uci)
                    self.last_move_uci = best_uci
                    applied = True

            debug = {
                "source": payload.source,
                "previous_fen": prev_board.fen(),
                "detected_move_uci": best_uci,
                "move_score": round(best_score, 3),
                "candidates": candidates,
                "mapped_occupancy": vision["occupancy"],
                "square_confidences": vision["confidences"],
                "raw_detections": vision["detections"],
                "warnings": vision["warnings"],
                "applied_to_game_state": applied,
            }
            self.latest_vision_debug = debug
            return {"ok": True, **debug}

    def get_vision_debug(self) -> dict:
        with self.lock:
            return self.latest_vision_debug or {"ok": True, "message": "No vision debug artifacts yet."}

    def _should_accept_vision_move(self, prev_fen: str, move_uci: str) -> bool:
        """Debounce duplicate accepts for same detected move."""
        now = time.time()
        key = f"{prev_fen}|{move_uci}"
        if key == self.vision_last_accepted_key and (now - self.vision_last_accepted_at) < 1.2:
            return False
        self.vision_last_accepted_key = key
        self.vision_last_accepted_at = now
        return True

    def _changed_squares_from_last_frame(
        self,
        current_occ: dict[str, str | None],
    ) -> list[str]:
        """Return squares that differ from previous detected frame occupancy."""
        if self.vision_last_frame_occupancy is None:
            self.vision_last_frame_occupancy = dict(current_occ)
            return []

        changed = [
            sq for sq in sorted(current_occ.keys())
            if current_occ.get(sq) != self.vision_last_frame_occupancy.get(sq)
        ]
        self.vision_last_frame_occupancy = dict(current_occ)
        return changed

    def _changed_squares_from_pixel_activity(
        self,
        warped_board: np.ndarray,
        square_map: dict,
        activity_threshold: float,
    ) -> tuple[list[str], dict[str, float]]:
        """Compare per-square pixel activity against baseline warped frame.

        Uses the inner region of each square to reduce border/flicker noise.
        """
        gray = cv2.cvtColor(warped_board, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if self.vision_last_warped_gray is None or self.vision_last_warped_gray.shape != gray.shape:
            self.vision_last_warped_gray = gray
            return [], {sq: 0.0 for sq in square_map}

        # Compensate global auto-exposure drift before differencing.
        prev_mean = float(np.mean(self.vision_last_warped_gray))
        curr_mean = float(np.mean(gray))
        shift = prev_mean - curr_mean
        gray_aligned = np.clip(gray.astype(np.float32) + shift, 0, 255).astype(np.uint8)
        diff = cv2.absdiff(self.vision_last_warped_gray, gray_aligned)
        activity_scores: dict[str, float] = {}
        changed: list[str] = []

        h, w = diff.shape[:2]
        for square, region in square_map.items():
            x, y, bw, bh = region.bbox
            # Ignore square edges where perspective/grid aliasing is strongest.
            mx = int(max(2, bw * 0.18))
            my = int(max(2, bh * 0.18))
            x0 = max(0, min(w - 1, int(x + mx)))
            y0 = max(0, min(h - 1, int(y + my)))
            x1 = max(x0 + 1, min(w, int(x + bw - mx)))
            y1 = max(y0 + 1, min(h, int(y + bh - my)))
            roi = diff[y0:y1, x0:x1]
            score = float(np.mean(roi)) if roi.size else 0.0
            activity_scores[square] = round(score, 3)
            if score >= activity_threshold:
                changed.append(square)

        return sorted(changed), activity_scores

    def _infer_move_from_changed_squares(
        self,
        board: chess.Board,
        changed_squares: list[str],
        current_occ: dict[str, str | None] | None = None,
        top_k: int = 10,
    ) -> tuple[str | None, float, list[dict]]:
        """Infer move using frame-to-frame square changes + legal move filtering.

        Source of truth stays python-chess board state; detections only inform deltas.
        """
        changed_set = set(changed_squares)
        if not changed_set:
            return None, 0.0, []

        prev_occ = board_to_square_occupancy(board)
        candidates: list[dict] = []
        for move in board.legal_moves:
            before = board.copy(stack=False)
            after = board.copy(stack=False)
            after.push(move)
            after_occ = board_to_square_occupancy(after)

            move_changed = {
                sq for sq in prev_occ
                if prev_occ.get(sq) != after_occ.get(sq)
            }
            overlap = len(changed_set & move_changed)
            union = len(changed_set | move_changed) or 1
            jaccard = overlap / union

            uci = move.uci()
            from_sq, to_sq = uci[:2], uci[2:4]
            moved_piece = before.piece_at(chess.parse_square(from_sq))
            expected_symbol = moved_piece.symbol() if moved_piece else None

            score = jaccard * 10.0
            if move_changed == changed_set:
                score += 6.0
            elif move_changed.issuperset(changed_set):
                score += 2.0
            if from_sq in changed_set:
                score += 2.0
            if to_sq in changed_set:
                score += 2.0
            if current_occ and expected_symbol and current_occ.get(to_sq) == expected_symbol:
                score += 3.0
            if current_occ and current_occ.get(from_sq) is None:
                score += 1.0
            if board.is_capture(move):
                score += 0.8
            if board.gives_check(move):
                score += 0.5

            candidates.append(
                {
                    "uci": uci,
                    "san": board.san(move),
                    "score": round(score, 3),
                    "changed_overlap": overlap,
                    "changed_count": len(changed_set),
                    "expected_change_count": len(move_changed),
                }
            )

        candidates.sort(key=lambda item: item["score"], reverse=True)
        top = candidates[:top_k]
        if not top:
            return None, 0.0, []
        return top[0]["uci"], float(top[0]["score"]), top

    def process_vision(self, payload: VisionProcessRequest) -> dict:
        """Process one frame and update board if a confident legal move is found."""
        with self.lock:
            frame = self._decode_image(payload.image_b64) if payload.image_b64 else self._capture_camera_frame(payload.camera_index)
            prev_board = self.game.get_board().copy(stack=False)
            prev_fen = prev_board.fen()

            # Intentionally skip YOLO here: use 64-square pixel activity only.
            vision = self._run_vision_pipeline(frame, include_yolo=False)
            changed_squares_activity, activity_scores = self._changed_squares_from_pixel_activity(
                warped_board=vision["warped"],
                square_map=vision["square_map"],
                activity_threshold=float(payload.activity_threshold),
            )
            changed_squares = list(changed_squares_activity)
            # If nearly all squares changed, board warp/camera exposure likely shifted.
            # Re-lock baseline/corners instead of treating this as piece motion.
            if len(changed_squares_activity) >= 56:
                self.vision_locked_corners = None
                self.vision_last_warped_gray = cv2.GaussianBlur(
                    cv2.cvtColor(vision["warped"], cv2.COLOR_BGR2GRAY),
                    (5, 5),
                    0,
                )
                self.vision_motion_active = False
                self.vision_settle_streak = 0
                changed_squares = []
            max_changed_for_move = max(2, int(payload.max_changed_squares_for_move))
            # Aggressive mode: always infer from the most-active squares.
            # This avoids getting stuck when motion gating is too strict.
            if len(changed_squares) > max_changed_for_move:
                changed_squares = sorted(
                    changed_squares,
                    key=lambda sq: activity_scores.get(sq, 0.0),
                    reverse=True,
                )[:max_changed_for_move]
            self.vision_motion_active = False
            self.vision_settle_streak = 0
            best_uci, best_score, candidates = self._infer_move_from_changed_squares(
                board=prev_board,
                changed_squares=changed_squares,
                current_occ=None,
                top_k=10,
            )

            changed = False
            accepted = False
            threshold = float(payload.confidence_threshold)
            if best_uci and best_score >= threshold and self.game.is_legal_move(best_uci):
                # Require short streak (default 1) so demos feel responsive.
                if self.vision_pending_move_uci == best_uci:
                    self.vision_pending_streak += 1
                else:
                    self.vision_pending_move_uci = best_uci
                    self.vision_pending_streak = 1

                min_streak = max(1, int(payload.min_streak))
                if self.vision_pending_streak >= min_streak and self._should_accept_vision_move(prev_fen, best_uci):
                    side = "White" if prev_board.turn == chess.WHITE else "Black"
                    san = prev_board.san(chess.Move.from_uci(best_uci))
                    self.game.apply_move(best_uci)
                    self.last_move_uci = best_uci
                    self._record_move(side, best_uci, san)
                    self.vision_last_move_uci = best_uci
                    self.vision_last_confidence = float(best_score)
                    self.vision_changed_flag = True
                    changed = True
                    accepted = True
                    self.vision_pending_move_uci = None
                    self.vision_pending_streak = 0
                    self.vision_motion_active = False
                    self.vision_settle_streak = 0
                    # Accept the settled board as new baseline.
                    self.vision_last_warped_gray = cv2.GaussianBlur(
                        cv2.cvtColor(vision["warped"], cv2.COLOR_BGR2GRAY),
                        (5, 5),
                        0,
                    )
            else:
                self.vision_pending_move_uci = None
                self.vision_pending_streak = 0
                # Slow baseline adaptation when no move and no strong motion.
                current_gray = cv2.GaussianBlur(cv2.cvtColor(vision["warped"], cv2.COLOR_BGR2GRAY), (5, 5), 0)
                if self.vision_last_warped_gray is None:
                    self.vision_last_warped_gray = current_gray
                elif not self.vision_motion_active and not changed_squares_activity:
                    self.vision_last_warped_gray = cv2.addWeighted(
                        self.vision_last_warped_gray,
                        0.92,
                        current_gray,
                        0.08,
                        0,
                    )

            debug = {
                "source": payload.source,
                "previous_fen": prev_fen,
                "current_fen": self.game.get_fen(),
                "detected_move_uci": best_uci,
                "move_score": round(float(best_score), 3),
                "threshold": threshold,
                "accepted": accepted,
                "changed": changed,
                "changed_squares": changed_squares,
                "changed_squares_activity": changed_squares_activity,
                "changed_squares_occupancy": [],
                "activity_scores": activity_scores,
                "pending_move_uci": self.vision_pending_move_uci,
                "pending_streak": self.vision_pending_streak,
                "motion_active": self.vision_motion_active,
                "settle_streak": self.vision_settle_streak,
                "last_move_uci": self.last_move_uci,
                "candidates": candidates,
                "mapped_occupancy": vision["occupancy"],
                "mapped_piece_count": vision["mapped_piece_count"],
                "square_confidences": vision["confidences"],
                "raw_detections": vision["detections"],
                "warnings": vision["warnings"],
            }
            # Keep `changed` as a per-process event flag for frontend polling.
            self.vision_changed_flag = changed
            # Debug logs for backend terminal troubleshooting.
            print(
                "[vision] changed_squares=", changed_squares,
                " activity_changed=", changed_squares_activity,
                " occ_changed=", [],
                " best=", best_uci, "score=", round(float(best_score), 3),
                " accepted=", accepted, " pending=", self.vision_pending_move_uci,
                " streak=", self.vision_pending_streak,
                " motion_active=", self.vision_motion_active,
                " settle=", self.vision_settle_streak,
                " detections=", len(vision["detections"]),
                " mapped_pieces=", vision["mapped_piece_count"],
            )
            if candidates:
                print("[vision] top candidates:", candidates[:3])
            self.latest_vision_debug = debug
            return {"ok": True, **debug, "state": self._snapshot()}

    def get_vision_state(self) -> dict:
        with self.lock:
            board = self.game.get_board()
            changed = self.vision_changed_flag
            self.vision_changed_flag = False
            return {
                "ok": True,
                "fen": self.game.get_fen(),
                "turn": "white" if board.turn == chess.WHITE else "black",
                "last_move_uci": self.last_move_uci,
                "vision_last_move_uci": self.vision_last_move_uci,
                "vision_last_confidence": round(float(self.vision_last_confidence), 3),
                "changed": changed,
            }

    def reset_vision_state(self) -> dict:
        with self.lock:
            self.game.reset()
            self.history.clear()
            self.latest_coach_text = ""
            self.last_move_uci = None
            self.vision_last_move_uci = None
            self.vision_last_confidence = 0.0
            self.vision_changed_flag = False
            self.vision_last_accepted_at = 0.0
            self.vision_last_accepted_key = ""
            self.vision_last_frame_occupancy = None
            self.vision_last_warped_gray = None
            self.vision_pending_move_uci = None
            self.vision_pending_streak = 0
            self.vision_motion_active = False
            self.vision_settle_streak = 0
            self.vision_locked_corners = None
            board = self.game.get_board()
            return {
                "ok": True,
                "fen": self.game.get_fen(),
                "turn": "white" if board.turn == chess.WHITE else "black",
                "last_move_uci": None,
                "vision_last_move_uci": None,
                "vision_last_confidence": 0.0,
                "changed": False,
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
    return service.vision_board(payload)


@app.post("/vision/process")
def post_vision_process(payload: VisionProcessRequest) -> dict:
    return service.process_vision(payload)


@app.get("/vision/state")
def get_vision_state() -> dict:
    return service.get_vision_state()


@app.post("/vision/reset")
def post_vision_reset() -> dict:
    return service.reset_vision_state()


@app.post("/vision/move")
def post_vision_move(payload: VisionMoveRequest) -> dict:
    return service.vision_move(payload)


@app.get("/vision/debug")
def get_vision_debug() -> dict:
    return service.get_vision_debug()


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
