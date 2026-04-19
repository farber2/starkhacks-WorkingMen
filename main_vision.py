"""Secondary entrypoint for live chess vision + grasp feature demo.

This keeps your existing `main.py` untouched and gives you a dedicated
camera pipeline runner for hackathon demos.
"""

from __future__ import annotations

import glob
import os
import sys
from typing import Any

import cv2

# Camera and detector are part of this repo.
from camera import Camera
from detect import ChessDetector

# Optional modules (some branches may not have these yet).
try:
    from board_map import build_board_map_with_confidences, draw_square_overlay, format_board_map
except Exception:  # pragma: no cover - optional dependency
    build_board_map_with_confidences = None
    draw_square_overlay = None
    format_board_map = None

FRAME_SIZE = 1080
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080
CAMERA_FOURCC = "MJPG"
CAMERA_INDEX = 0
MODEL_PATH = "runs/detect/runs/chess_pieces_white_tuned/weights/best.pt"


def crop_to_square(frame: Any, output_size: int = FRAME_SIZE) -> Any:
    """Crop frame to centered square and resize to output size."""
    height, width = frame.shape[:2]
    side = min(width, height)
    x1 = (width - side) // 2
    y1 = (height - side) // 2
    square = frame[y1 : y1 + side, x1 : x1 + side]

    if square.shape[0] != output_size or square.shape[1] != output_size:
        square = cv2.resize(square, (output_size, output_size), interpolation=cv2.INTER_LINEAR)
    return square


def create_camera(index: int) -> Camera:
    return Camera(
        index=index,
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
        fourcc=CAMERA_FOURCC,
    )


def _open_camera_with_fallback(preferred_index: int, label: str, used_indices: set[int]) -> tuple[Camera, int]:
    """Open camera by preferred index, then fallback scan 0..6."""
    candidates = [preferred_index] + [i for i in range(0, 7) if i != preferred_index]
    for idx in candidates:
        if idx in used_indices:
            continue
        camera = create_camera(idx)
        try:
            camera.open()
            print(f"{label} camera opened at index {idx}.")
            return camera, idx
        except RuntimeError:
            continue
    raise RuntimeError(f"No available camera found for {label}. Tried indices: {candidates}")


def _fallback_format_board(board_map: Any) -> str:
    if not board_map:
        return "No board map available."
    lines = []
    for row in board_map:
        lines.append(" ".join(cell if cell else "." for cell in row))
    return "\n".join(lines)


def detect_and_map(detector: ChessDetector, frame: Any) -> tuple[Any, list[dict[str, Any]], Any]:
    """Run piece detection and best-effort board mapping."""
    frame = crop_to_square(frame)
    _, detections = detector.detect(frame)
    height, width = frame.shape[:2]

    if build_board_map_with_confidences is not None:
        board, _confidences = build_board_map_with_confidences(
            detections,
            board_bbox=(0, 0, width, height),
        )
    else:
        # Fall back to detector helper (if board_map.py supports it).
        board = detector.build_board_map(
            detections,
            board_bbox=(0, 0, width, height),
        )
    return frame, detections, board


def _resolve_model_path() -> str:
    """Resolve chess YOLO weights path for vision demo.

    Priority:
    1) YOLO_MODEL_PATH env var
    2) default MODEL_PATH
    3) latest best.pt under runs/**/weights/
    """
    env_path = os.getenv("YOLO_MODEL_PATH", "").strip()
    if env_path:
        if os.path.exists(env_path):
            return env_path
        print(f"YOLO_MODEL_PATH is set but file does not exist: {env_path}")

    if os.path.exists(MODEL_PATH):
        return MODEL_PATH

    candidates = sorted(
        glob.glob("runs/**/weights/best.pt", recursive=True),
        key=lambda p: os.path.getmtime(p),
        reverse=True,
    )
    if candidates:
        return candidates[0]

    print("No YOLO weights file found.")
    print("Expected one of:")
    print(f" - {MODEL_PATH}")
    print(" - runs/**/weights/best.pt")
    print("")
    print("Fix options:")
    print("1) Train/export weights and place best.pt under runs/.../weights/")
    print("2) Set explicit path:")
    print("   export YOLO_MODEL_PATH=/absolute/path/to/best.pt")
    sys.exit(1)


def main() -> None:
    """Run single-camera board/piece vision loop (robot pipeline disabled)."""
    model_path = _resolve_model_path()
    print(f"Using YOLO model: {model_path}")

    used_indices: set[int] = set()
    camera = None
    try:
        detector = ChessDetector(
            model_path=model_path,
            confidence_threshold=0.25,
            image_size=1088,
        )
    except Exception as exc:
        print("Failed to initialize ChessDetector with the model above.")
        print(f"Details: {exc}")
        print("If this is not your chess-piece model, verify it contains classes like 'white-pawn'.")
        sys.exit(1)

    try:
        camera, camera_index = _open_camera_with_fallback(
            CAMERA_INDEX,
            "Vision",
            used_indices,
        )
        used_indices.add(camera_index)

        width, height = camera.resolution()
        print(f"Vision camera {camera_index} opened at {width}x{height}. Press ESC to quit.")

        frame_count = 0
        read_failures = 0
        while True:
            frame_count += 1

            try:
                frame = camera.read()
                read_failures = 0
            except RuntimeError as exc:
                read_failures += 1
                if read_failures % 10 == 1:
                    print(f"[camera warning] {exc} (attempt {read_failures})")
                # Keep trying for transient camera drops.
                if read_failures >= 60:
                    raise RuntimeError("Camera read failed repeatedly. Check camera permissions/device.") from exc
                continue

            board_frame, detections, board_map = detect_and_map(detector, frame)

            if frame_count % 5 == 0:
                print("\nDetected board map:")
                if format_board_map is not None:
                    print(format_board_map(board_map))
                else:
                    print(_fallback_format_board(board_map))

            annotated = detector.draw_results(board_frame, detections)
            if draw_square_overlay is not None:
                # Draw a simple 8x8 square guide for easier tuning.
                from board_map import get_square_map

                annotated = draw_square_overlay(annotated, get_square_map(annotated))
            cv2.imshow("Chess Vision (Board + YOLO Pieces)", annotated)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        if camera is not None:
            camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
