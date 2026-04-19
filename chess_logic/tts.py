"""Local Piper TTS helpers for spoken chess coaching output.

This module is intentionally simple:
- synthesizes speech with local Piper
- writes to a temporary WAV file
- plays audio on macOS with afplay
- runs playback on a background worker so the caller is non-blocking
- prints warnings instead of crashing when dependencies are missing
"""

from __future__ import annotations

import atexit
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

# Keep the model location in one place for easy configuration.
MODEL_PATH = "voice/en_US-amy-medium.onnx"
PIPER_BIN = "piper"
PLAYER_BIN = "afplay"
MAX_TEXT_CHARS = 320

_speech_queue: queue.Queue[str | None] = queue.Queue()
_worker_thread: threading.Thread | None = None
_worker_lock = threading.Lock()


def _warn(message: str) -> None:
    """Print a non-fatal warning message."""
    print(f"[TTS warning] {message}")


def _dependencies_ready() -> bool:
    """Check local TTS dependencies and print one-shot warnings."""
    model_file = Path(MODEL_PATH)
    model_config = Path(f"{MODEL_PATH}.json")

    if shutil.which(PIPER_BIN) is None:
        _warn("Piper is not installed or not on PATH. Skipping speech.")
        return False

    if not model_file.exists():
        _warn(f"Amy model not found at {model_file}. Skipping speech.")
        return False

    if not model_config.exists():
        _warn(f"Amy model config not found at {model_config}. Skipping speech.")
        return False

    if shutil.which(PLAYER_BIN) is None:
        _warn("afplay is not available on PATH. Skipping speech playback.")
        return False

    return True


def _play_wav(temp_wav_path: str) -> None:
    """Play generated speech and retry once for transient macOS audio errors."""
    try:
        subprocess.run([PLAYER_BIN, temp_wav_path], capture_output=True, check=True)
        return
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        details = stderr.strip() or str(exc)
        if "AudioQueueStart failed" not in details:
            raise

        # Audio output can fail transiently if the device is briefly unavailable.
        time.sleep(0.25)
        try:
            subprocess.run([PLAYER_BIN, temp_wav_path], capture_output=True, check=True)
            return
        except subprocess.CalledProcessError as retry_exc:
            retry_stderr = (
                retry_exc.stderr.decode()
                if isinstance(retry_exc.stderr, bytes)
                else (retry_exc.stderr or "")
            )
            retry_details = retry_stderr.strip() or str(retry_exc)
            raise RuntimeError(
                "AudioQueueStart failed repeatedly. Check macOS audio output device. "
                f"Details: {retry_details}"
            ) from retry_exc


def _speak_now(text: str) -> None:
    """Blocking speech synthesis and playback for worker usage."""
    message = text.strip()
    if not message:
        return

    model_file = Path(MODEL_PATH)
    temp_wav_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            temp_wav_path = temp_wav.name

        synth_cmd = [
            PIPER_BIN,
            "--model",
            str(model_file),
            "--output_file",
            temp_wav_path,
        ]
        subprocess.run(
            synth_cmd,
            input=message,
            text=True,
            capture_output=True,
            check=True,
        )
        _play_wav(temp_wav_path)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        details = stderr.strip() or str(exc)
        _warn(f"Piper synthesis/playback failed: {details}")
    except RuntimeError as exc:
        _warn(str(exc))
    except Exception as exc:  # pragma: no cover - defensive runtime safety
        _warn(f"Unexpected TTS error: {exc}")
    finally:
        if temp_wav_path and os.path.exists(temp_wav_path):
            try:
                os.remove(temp_wav_path)
            except OSError:
                pass


def _worker_loop() -> None:
    """Serial audio worker to avoid overlapping piper/afplay processes."""
    while True:
        item = _speech_queue.get()
        if item is None:
            _speech_queue.task_done()
            break
        _speak_now(item)
        _speech_queue.task_done()


def _ensure_worker_started() -> bool:
    """Start worker thread lazily after dependency checks."""
    global _worker_thread

    if not _dependencies_ready():
        return False

    with _worker_lock:
        if _worker_thread is None or not _worker_thread.is_alive():
            _worker_thread = threading.Thread(target=_worker_loop, daemon=True, name="tts-worker")
            _worker_thread.start()
    return True


def speak_text(text: str) -> None:
    """Queue text for non-blocking speech with local Piper and afplay."""
    message = " ".join(text.strip().split())
    if not message:
        return

    if not _ensure_worker_started():
        return

    _speech_queue.put(message[:MAX_TEXT_CHARS])


def shutdown_tts(wait: bool = False) -> None:
    """Stop background speech worker.

    Args:
        wait: If True, waits for queued items to finish before stopping.
    """
    global _worker_thread
    thread = _worker_thread
    if thread is None:
        return

    if wait:
        _speech_queue.join()
    _speech_queue.put(None)
    thread.join(timeout=3)
    _worker_thread = None


atexit.register(shutdown_tts)
