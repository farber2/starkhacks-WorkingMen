"""Local Ollama-powered beginner coaching helpers for the chess workflow.

This module adds natural-language tutoring on top of existing chess logic.
It does not validate or choose moves; that remains the job of the chess engine.
"""

from __future__ import annotations

import re

import chess
import requests

# Keep Ollama connection and generation details in one place.
OLLAMA_URL = "http://127.0.0.1:11434"
OLLAMA_MODEL = "phi3"
GENERATE_ENDPOINT = "/api/generate"
REQUEST_TIMEOUT_SECONDS = 20
REQUEST_TIMEOUT_HELP_SECONDS = 35
OLLAMA_TEMPERATURE = 0.2
OLLAMA_NUM_PREDICT = 64
OLLAMA_NUM_PREDICT_HELP = 160

COACH_SYSTEM_PROMPT = """You are a friendly beginner chess coach.
Tone: calm, supportive, clear, and slightly upbeat.
Style: natural spoken coaching suitable for TTS.
Use recommendation tone, not approval tone.
Do not use praise phrases like "nice move", "good move", "great move", or "nice choice".
Vary wording naturally across calls.

Grounding and safety rules:
- Use only trusted move facts and trusted themes from the prompt.
- Never invent a different move, piece, square, or side to move.
- Never invent strategic claims outside the trusted themes list.
- Treat advice as guidance for the side to move only.
- Avoid concrete follow-up square suggestions unless explicitly supported.

Forbidden output content:
- Do not mention UCI, SAN, notation, engine score, eval, centipawns, or metadata.
- Do not include labels, lists, numbering, or section headers.
"""

FALLBACK_MOVE_SECOND_SENTENCE = (
    "Nice move—this helps your development and makes your position easier to play."
)
FALLBACK_POSITION_EXPLANATION = (
    "Nice progress so far. Focus on checks, hanging pieces, king safety, and center control."
)


def get_piece_name(board: chess.Board, move_uci: str) -> str:
    """Return moved piece name for UCI move from the current board."""
    move = chess.Move.from_uci(move_uci)
    piece = board.piece_at(move.from_square)
    if piece is None:
        return "unknown piece"

    names = {
        chess.PAWN: "pawn",
        chess.KNIGHT: "knight",
        chess.BISHOP: "bishop",
        chess.ROOK: "rook",
        chess.QUEEN: "queen",
        chess.KING: "king",
    }
    return names.get(piece.piece_type, "piece")


def get_move_summary(board: chess.Board, move_uci: str) -> dict[str, str]:
    """Build factual move summary from python-chess before prompting the model."""
    move = chess.Move.from_uci(move_uci)
    side_that_moved = "White" if board.turn == chess.WHITE else "Black"
    return {
        "piece_moved": get_piece_name(board, move_uci),
        "from_square": chess.square_name(move.from_square),
        "to_square": chess.square_name(move.to_square),
        "is_capture": "yes" if board.is_capture(move) else "no",
        "side_that_moved": side_that_moved,
    }


def _build_sentence_one(move_summary: dict[str, str]) -> str:
    """Generate exact fact sentence in Python so move identity never drifts."""
    capture_text = "and it was a capture" if move_summary["is_capture"] == "yes" else "and it was not a capture"
    return (
        f"{move_summary['side_that_moved']} moved the {move_summary['piece_moved']} "
        f"from {move_summary['from_square']} to {move_summary['to_square']}, {capture_text}."
    )


def build_move_prompt(
    board_fen: str,
    move_uci: str,
    move_summary: dict[str, str],
    sentence_one: str,
    engine_suggestion: str | None = None,
) -> str:
    """Build strict move prompt for sentence-two coaching only."""
    lines = [
        "Write exactly one short sentence for sentence 2 of a beginner coaching response.",
        "Do not rewrite sentence 1 and do not restate move coordinates.",
        "Use recommendation tone, not praise tone.",
        "Do not use phrases like 'nice move' or 'good move'.",
        "Use natural wording that can vary between calls.",
        "Explain why the move helps using center control, development, king safety, or piece activity.",
        "Do not invent any facts or additional played moves.",
        "Do not propose a concrete follow-up move unless labeled exactly 'one possible next idea: ...'.",
        f"Board FEN: {board_fen}",
        f"Move UCI: {move_uci}",
        f"Side that moved: {move_summary['side_that_moved']}",
        f"Piece moved: {move_summary['piece_moved']}",
        f"From square: {move_summary['from_square']}",
        f"To square: {move_summary['to_square']}",
        f"Was capture: {move_summary['is_capture']}",
        f"Sentence 1 (already fixed): {sentence_one}",
    ]
    if engine_suggestion:
        lines.append(f"Optional context (do not treat as played): engine suggestion {engine_suggestion}")
    return "\n".join(lines)


def _build_position_prompt(board_fen: str) -> str:
    """Build compact position-level coaching prompt."""
    return "\n".join(
        [
            "Write exactly 2 short beginner-friendly coaching sentences.",
            "Be supportive and practical.",
            "Focus on checks, hanging pieces, king safety, center control, and development.",
            "Do not invent played moves.",
            f"Board FEN: {board_fen}",
        ]
    )


def _build_help_prompt(
    board_fen: str,
    best_move_summary: dict[str, str],
    candidate_moves: list[dict[str, str]],
    trusted_themes: list[str],
) -> str:
    """Build a strict help-mode prompt for exactly four short sentences."""
    candidate_san = ", ".join(move.get("san", "?") for move in candidate_moves[:3] if move.get("san"))
    recommended_san = best_move_summary.get("san", "").strip()
    if not recommended_san:
        recommended_san = best_move_summary.get("uci", "").strip()

    return "\n".join(
        [
            "Write exactly 4 short spoken beginner-coach sentences.",
            "Keep each sentence short (about 8-14 words).",
            "Use recommendation tone, not praise.",
            "Do not use approval phrases such as 'nice move', 'good move', 'great move', or 'nice choice'.",
            "Use fresh wording and avoid repetitive templates.",
            "Use only the trusted move facts and trusted themes below.",
            "Do not invent any new strategic claim outside trusted themes.",
            "Do not say the move develops a knight unless development is trusted and moved piece is knight.",
            "Do not claim king safety unless king_safety or castle is trusted.",
            "Do not claim rook connection unless improves_coordination or castle is trusted.",
            "Do not describe any move as already played except the recommendation.",
            "Do not mention notation terms, SAN/UCI names, engine score, eval, or metadata in the final output.",
            "Avoid concrete next-move square suggestions; keep watch-next guidance generic.",
            "In sentence 1, recommend the trusted move clearly.",
            "In sentence 2, explain one supported reason using trusted themes.",
            "In sentence 3, explain one beginner-friendly idea using trusted themes.",
            "In sentence 4, say one thing to watch next without unsupported tactics.",
            "Avoid generic filler unless directly relevant to the position.",
            f"Board FEN: {board_fen}",
            f"Recommended move UCI (trusted): {best_move_summary['uci']}",
            f"Recommended move SAN (trusted): {recommended_san}",
            f"Side to move: {best_move_summary['side_that_moved']}",
            f"Piece moved: {best_move_summary['piece_moved']}",
            f"From square: {best_move_summary['from_square']}",
            f"To square: {best_move_summary['to_square']}",
            f"Capture: {best_move_summary['is_capture']}",
            f"Trusted themes: {', '.join(trusted_themes) if trusted_themes else 'improves_coordination'}",
            f"Other candidate ideas (trusted, optional context): {candidate_san}",
        ]
    )


def detect_help_themes(board: chess.Board, best_move_uci: str) -> list[str]:
    """Return trusted help themes that Python can justify from board + move."""
    move = chess.Move.from_uci(best_move_uci)
    if move not in board.legal_moves:
        return ["improves_coordination"]

    piece = board.piece_at(move.from_square)
    if piece is None:
        return ["improves_coordination"]

    side = board.turn
    from_rank = chess.square_rank(move.from_square)
    to_rank = chess.square_rank(move.to_square)
    to_sq = chess.square_name(move.to_square)

    themes: set[str] = set()

    if piece.piece_type in {chess.KNIGHT, chess.BISHOP}:
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
            themes.add("development")

    if to_sq in {"d4", "e4", "d5", "e5"}:
        themes.add("center_control")

    if board.is_castling(move):
        themes.update({"castle", "king_safety", "improves_coordination"})

    if piece.piece_type == chess.BISHOP:
        if (side == chess.WHITE and to_rank < from_rank) or (
            side == chess.BLACK and to_rank > from_rank
        ):
            themes.add("bishop_retreat")

    if piece.piece_type == chess.PAWN and abs(to_rank - from_rank) == 2:
        themes.add("gains_space")
        if chess.square_file(move.from_square) in {3, 4}:  # d/e pawn
            themes.add("supports_center")

    before_attackers = board.attackers(not side, move.from_square)
    board_after = board.copy()
    board_after.push(move)
    moved_attackers = board_after.attacks(move.to_square)
    if moved_attackers:
        for sq in moved_attackers:
            attacked_piece = board_after.piece_at(sq)
            if attacked_piece and attacked_piece.color != side:
                themes.add("attacks_piece")
                if attacked_piece.piece_type in {chess.QUEEN, chess.ROOK, chess.KING}:
                    themes.add("creates_threat")
                break

    if piece.piece_type in {chess.BISHOP, chess.ROOK, chess.QUEEN}:
        if board_after.attacks(move.to_square):
            themes.add("maintains_pressure")

    if before_attackers and not board_after.is_attacked_by(not side, move.to_square):
        themes.add("avoids_threat")

    if piece.piece_type == chess.PAWN and move.from_square in {
        chess.E2,
        chess.D2,
        chess.E7,
        chess.D7,
    }:
        themes.add("opens_line")

    if piece.piece_type in {chess.KNIGHT, chess.BISHOP} and move.from_square in {
        chess.G1,
        chess.F1,
        chess.G8,
        chess.F8,
    }:
        themes.add("prepares_castling")

    if not themes:
        themes.add("improves_coordination")

    # Stable order keeps prompts predictable.
    priority = [
        "castle",
        "king_safety",
        "development",
        "center_control",
        "supports_center",
        "gains_space",
        "opens_line",
        "attacks_piece",
        "creates_threat",
        "bishop_retreat",
        "avoids_threat",
        "maintains_pressure",
        "prepares_castling",
        "improves_coordination",
    ]
    return [tag for tag in priority if tag in themes]


def _clean_single_sentence(text: str) -> str:
    """Normalize model output to a single short sentence."""
    cleaned = " ".join(text.strip().split())
    if not cleaned:
        return ""

    first = re.split(r"(?<=[.!?])\s+", cleaned)[0].strip()
    if not first:
        return ""
    if first[-1] not in ".!?":
        first = f"{first}."
    return first


def _clean_two_sentences(text: str) -> str:
    """Normalize model output to at most two short sentences."""
    cleaned = " ".join(text.strip().split())
    if not cleaned:
        return ""

    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    short = " ".join(parts[:2]).strip()
    return short or cleaned


def _clean_four_sentences(text: str) -> str:
    """Normalize model output to exactly four short sentences when possible."""
    cleaned = " ".join(text.strip().split())
    cleaned = re.sub(r"(?:^|\s)[1-4]\.\s+", " ", cleaned)
    cleaned = re.sub(r"\bSentence\s*0*\d{1,2}\s*:\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bSentence\s*0*\d{1,2}\s*role\s*:\s*[^.]*\.?", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bIn sentence\s*[1-4]\s*,\s*[^.]*\.?", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(?:using\s+)?UCI(?:\s+notation)?\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bSAN\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(?:engine\s+)?score\s*:\s*[-+]?\d+(?:\.\d+)?\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\beval(?:uation)?\s*:\s*[-+]?\d+(?:\.\d+)?\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bcentipawns?\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"[\"“”]", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,;")
    if not cleaned:
        return ""

    parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", cleaned) if part.strip()]
    if len(parts) < 4:
        return cleaned
    return " ".join(parts[:4])


def _theme_keyword_requirements() -> dict[str, set[str]]:
    """Map textual claims to required trusted themes."""
    return {
        r"\bdevelop(?:s|ment|ing)?\b": {"development"},
        r"\bcenter|central\b": {"center_control", "supports_center", "gains_space"},
        r"\bking safety|safe king|safer king|castl": {"king_safety", "castle", "prepares_castling"},
        r"\bconnect(?:s|ing)? the rooks|activate(?:s|d)? the rook": {"improves_coordination", "castle"},
        r"\battack(?:s|ing)?|threat|pressure": {"attacks_piece", "creates_threat", "maintains_pressure"},
        r"\bspace\b": {"gains_space"},
        r"\bopen(?:s|ing)? (?:a )?(?:line|diagonal|file)": {"opens_line"},
        r"\bavoid(?:s|ing)? threat|safer square|retreat": {"avoids_threat", "bishop_retreat"},
    }


def _sentence_has_forbidden_metadata(sentence: str) -> bool:
    patterns = [
        r"\bUCI\b",
        r"\bSAN\b",
        r"\bnotation\b",
        r"\bscore\b",
        r"\beval(?:uation)?\b",
        r"\bcentipawn",
    ]
    return any(re.search(p, sentence, flags=re.IGNORECASE) for p in patterns)


def _sentence_uses_unsupported_theme(sentence: str, trusted_themes: set[str]) -> bool:
    lowered = sentence.lower()
    for pattern, required in _theme_keyword_requirements().items():
        if re.search(pattern, lowered, flags=re.IGNORECASE):
            if required.isdisjoint(trusted_themes):
                return True
    return False


def _build_theme_reason_sentence(themes: set[str]) -> str:
    if "castle" in themes:
        return "This castles now and immediately improves king safety and rook coordination."
    if "attacks_piece" in themes or "creates_threat" in themes:
        return "This move creates practical pressure and asks immediate questions."
    if "development" in themes:
        return "This develops a piece to a more useful square."
    if "center_control" in themes or "supports_center" in themes:
        return "This supports central control and makes your position easier to handle."
    if "avoids_threat" in themes:
        return "This improves safety by reducing immediate tactical risk."
    if "bishop_retreat" in themes:
        return "This keeps the bishop active from a safer square."
    return "This improves coordination and supports a practical plan."


def _build_theme_idea_sentence(themes: set[str]) -> str:
    if "prepares_castling" in themes or "castle" in themes:
        return "For a beginner plan, prioritize king safety and smooth piece coordination."
    if "gains_space" in themes or "center_control" in themes:
        return "A simple idea is to claim central space and develop around it."
    if "opens_line" in themes:
        return "A useful idea is opening lines so your pieces can become active."
    if "maintains_pressure" in themes or "attacks_piece" in themes:
        return "A useful idea is keeping pressure while finishing development."
    return "A simple idea is to improve piece activity without creating weaknesses."


def _build_theme_watch_sentence(themes: set[str]) -> str:
    if "center_control" in themes or "supports_center" in themes:
        return "Next, watch how the center is challenged and keep your pieces coordinated."
    if "king_safety" in themes or "castle" in themes:
        return "Next, watch checks and keep your king and pieces well coordinated."
    if "attacks_piece" in themes or "creates_threat" in themes:
        return "Next, watch forcing replies and avoid leaving your pieces loose."
    return "Next, watch checks, hanging pieces, and your development pace."


def _call_ollama(
    user_prompt: str,
    num_predict: int | None = None,
    timeout_seconds: int | None = None,
) -> str:
    """Send a non-streaming generation request to local Ollama.

    COACH_SYSTEM_PROMPT is prepended on every request so personality is always applied.
    """
    endpoint = f"{OLLAMA_URL}{GENERATE_ENDPOINT}"
    full_prompt = f"{COACH_SYSTEM_PROMPT}\n\n{user_prompt}"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": full_prompt,
        "stream": False,
        "options": {
            "temperature": OLLAMA_TEMPERATURE,
            "num_predict": num_predict or OLLAMA_NUM_PREDICT,
        },
    }
    response = requests.post(
        endpoint,
        json=payload,
        timeout=timeout_seconds or REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    data = response.json()

    if data.get("error"):
        raise RuntimeError(str(data["error"]))

    text = data.get("response", "").strip()
    if not text:
        raise RuntimeError("Ollama returned an empty response.")

    return text


def explain_move(
    board_fen: str,
    move_uci: str,
    engine_suggestion: str | None = None,
) -> str:
    """Explain a move in beginner-friendly language as exactly two short sentences.

    Sentence 1 is generated in Python from exact move facts.
    Sentence 2 is generated by Phi-3 with a coaching tone.
    """
    try:
        board = chess.Board(board_fen)
        move = chess.Move.from_uci(move_uci)
        if move not in board.legal_moves:
            return (
                "That move was not legal in this position. "
                "Nice effort—double-check checks, piece safety, and square coordinates."
            )

        summary = get_move_summary(board, move_uci)
        sentence_one = _build_sentence_one(summary)
        prompt = build_move_prompt(
            board_fen=board_fen,
            move_uci=move_uci,
            move_summary=summary,
            sentence_one=sentence_one,
            engine_suggestion=engine_suggestion,
        )
        sentence_two = _clean_single_sentence(_call_ollama(prompt))
        if not sentence_two:
            sentence_two = FALLBACK_MOVE_SECOND_SENTENCE
        return f"{sentence_one} {sentence_two}"
    except (ValueError, requests.RequestException, RuntimeError):
        summary_fallback = {
            "side_that_moved": "White",
            "piece_moved": "piece",
            "from_square": move_uci[:2] if len(move_uci) >= 4 else "??",
            "to_square": move_uci[2:4] if len(move_uci) >= 4 else "??",
            "is_capture": "no",
        }
        sentence_one = _build_sentence_one(summary_fallback)
        return f"{sentence_one} {FALLBACK_MOVE_SECOND_SENTENCE}"


def explain_engine_move(board_fen: str, engine_move_uci: str) -> str:
    """Explain an engine move in beginner-friendly language."""
    return explain_move(board_fen=board_fen, move_uci=engine_move_uci)


def explain_help_recommendation(
    board_fen: str,
    best_move_summary: dict[str, str],
    candidate_moves: list[dict[str, str]],
) -> str:
    """Explain help-mode recommendation in exactly four short sentences."""
    recommended_uci = best_move_summary.get("uci", "").lower()
    recommended_san = best_move_summary.get("san", "")
    recommended_move_spoken = (recommended_san or best_move_summary.get("uci", "")).strip()
    recommendation_sentence = (
        f"Recommendation: {best_move_summary.get('uci', '').strip()} ({recommended_move_spoken})."
        if best_move_summary.get("uci")
        else f"Recommendation: {recommended_move_spoken}."
    )
    board = chess.Board(board_fen)
    trusted_themes = set(detect_help_themes(board, best_move_summary["uci"]))
    moved_piece = best_move_summary.get("piece_moved", "").lower()

    def _mentions_recommended_move(text: str) -> bool:
        lowered = text.lower()
        if recommended_uci and recommended_uci in lowered:
            return True
        if recommended_san and recommended_san.lower() in lowered:
            return True
        return False

    def _split_sentences(text: str) -> list[str]:
        return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]

    def _ensure_four_sentences(parts: list[str]) -> str:
        padded = parts[:]
        while len(padded) < 4:
            if len(padded) == 1:
                padded.append(_build_theme_reason_sentence(trusted_themes))
            elif len(padded) == 2:
                padded.append(_build_theme_idea_sentence(trusted_themes))
            else:
                padded.append(_build_theme_watch_sentence(trusted_themes))
        return " ".join(padded[:4])

    def _sanitize_sentence(sentence: str) -> str:
        sentence = re.sub(r"^\s*[1-4]\.\s*", "", sentence)
        sentence = re.sub(r"\bSentence\s*0*\d{1,2}\s*:\s*", "", sentence, flags=re.IGNORECASE)
        sentence = re.sub(
            r"^\s*(?:trusted theme explanation|beginner-friendly idea|reasoning behind the move(?:\s*\(trusted\))?|trusted theme reasoning)\s*:?\s*",
            "",
            sentence,
            flags=re.IGNORECASE,
        )
        sentence = re.sub(r"\b(?:using\s+)?UCI(?:\s+notation)?\b", "", sentence, flags=re.IGNORECASE)
        sentence = re.sub(r"\bSAN\b", "", sentence, flags=re.IGNORECASE)
        sentence = re.sub(r"\(trusted\)", "", sentence, flags=re.IGNORECASE)
        sentence = re.sub(r"\bscore\s*:\s*[-+]?\d+(?:\.\d+)?\b", "", sentence, flags=re.IGNORECASE)
        sentence = re.sub(r"\beval(?:uation)?\s*:\s*[-+]?\d+(?:\.\d+)?\b", "", sentence, flags=re.IGNORECASE)
        sentence = re.sub(r"\s+", " ", sentence).strip(" ,;")
        if sentence and sentence[-1] not in ".!?":
            sentence = f"{sentence}."
        return sentence

    def _repair_bad_sentence(index: int) -> str:
        if index == 1:
            return _build_theme_reason_sentence(trusted_themes)
        if index == 2:
            return _build_theme_idea_sentence(trusted_themes)
        return _build_theme_watch_sentence(trusted_themes)

    try:
        prompt = _build_help_prompt(
            board_fen=board_fen,
            best_move_summary=best_move_summary,
            candidate_moves=candidate_moves,
            trusted_themes=sorted(trusted_themes),
        )
        text = _clean_four_sentences(
            _call_ollama(
                prompt,
                num_predict=OLLAMA_NUM_PREDICT_HELP,
                timeout_seconds=REQUEST_TIMEOUT_HELP_SECONDS,
            )
        )
        parts = [_sanitize_sentence(part) for part in _split_sentences(text)]
        parts = [part for part in parts if part]
        if not parts:
            raise RuntimeError("Empty help explanation from model.")

        # Prefer model wording. Repair only the first sentence when needed.
        first_sentence = parts[0]
        move_ok = _mentions_recommended_move(first_sentence) or _mentions_recommended_move(text)
        if not move_ok:
            parts[0] = recommendation_sentence

        # Light-touch validation and repair:
        # keep model wording unless there is an obvious contradiction.
        for idx in range(1, min(len(parts), 4)):
            sentence = parts[idx]
            lowered = sentence.lower()
            is_bad = False

            if _sentence_has_forbidden_metadata(sentence):
                is_bad = True
            if "develop" in lowered and "knight" in lowered and moved_piece != "knight":
                is_bad = True
            if ("connect" in lowered and "rook" in lowered) and (
                {"improves_coordination", "castle"} & trusted_themes
            ) == set():
                is_bad = True
            if ("castles" in lowered or "castling" in lowered or "castle" in lowered) and (
                {"king_safety", "castle"} & trusted_themes
            ) == set():
                is_bad = True

            if is_bad:
                parts[idx] = _repair_bad_sentence(idx)

        return _ensure_four_sentences(parts)
    except (requests.RequestException, RuntimeError, KeyError, ValueError):
        pass

    # Theme-aware minimal fallback.
    return (
        f"{recommendation_sentence} "
        f"{_build_theme_reason_sentence(trusted_themes)} "
        f"{_build_theme_idea_sentence(trusted_themes)} "
        f"{_build_theme_watch_sentence(trusted_themes)}"
    )


def explain_position(board_fen: str) -> str:
    """Explain the current position in beginner-friendly terms."""
    prompt = _build_position_prompt(board_fen)
    try:
        return _clean_two_sentences(_call_ollama(prompt))
    except (requests.RequestException, RuntimeError):
        return FALLBACK_POSITION_EXPLANATION


def suggest_beginner_plan(board_fen: str) -> str:
    """Suggest a simple beginner plan for the current position."""
    prompt = _build_position_prompt(board_fen)
    try:
        return _clean_two_sentences(_call_ollama(prompt))
    except (requests.RequestException, RuntimeError):
        return FALLBACK_POSITION_EXPLANATION


def ollama_status_hint() -> str:
    """Return a short local setup hint when troubleshooting tutor responses."""
    return (
        "Make sure Ollama is running (`ollama serve`) and the model is available "
        f"(`ollama run {OLLAMA_MODEL}`)."
    )
