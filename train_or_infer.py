import argparse

from ultralytics import YOLO


DATASET_PATH = "dataset_pieces.yaml"
CURRENT_WHITE_SIDE_MODEL = "runs/detect/runs/chess_pieces_white_tuned/weights/best.pt"

ROBOT_ARM_CAMERA = "robot_arm"
BIRDS_EYE_CAMERA = "birds_eye"


def _train_model(model_path, **training_args):
    model = YOLO(model_path)
    return model.train(**training_args)


def _base_training_args(name):
    return {
        "data": DATASET_PATH,
        "epochs": 100,
        "imgsz": 832,
        "batch": 8,
        "cache": True,
        "workers": 3,
        "device": 0,
        "project": "runs",
        "name": name,
        "patience": 30,
        "hsv_h": 0.01,
        "hsv_s": 0.25,
        "hsv_v": 0.25,
        "degrees": 3.0,
        "translate": 0.08,
        "scale": 0.4,
        "perspective": 0.0005,
        "mosaic": 0.7,
        "close_mosaic": 15,
    }


def train_robot_arm_camera():
    training_args = _base_training_args("chess_pieces_robot_arm_white_side")
    return _train_model(CURRENT_WHITE_SIDE_MODEL, **training_args)


def train_birds_eye_camera():
    training_args = _base_training_args("chess_pieces_birds_eye")
    training_args.update(
        {
            "epochs": 125,
            "imgsz": 1024,
            "batch": 4,
            "patience": 40,
            "hsv_s": 0.35,
            "hsv_v": 0.35,
            "degrees": 12.0,
            "translate": 0.12,
            "scale": 0.55,
            "perspective": 0.0,
            "flipud": 0.5,
            "fliplr": 0.5,
            "mosaic": 0.8,
            "close_mosaic": 20,
        }
    )
    return _train_model(CURRENT_WHITE_SIDE_MODEL, **training_args)


TRAINERS = {
    ROBOT_ARM_CAMERA: train_robot_arm_camera,
    BIRDS_EYE_CAMERA: train_birds_eye_camera,
}

CAMERA_ALIASES = {
    "robot": ROBOT_ARM_CAMERA,
    "robot-arm": ROBOT_ARM_CAMERA,
    "robot_arm": ROBOT_ARM_CAMERA,
    "arm": ROBOT_ARM_CAMERA,
    "behind-board": ROBOT_ARM_CAMERA,
    "behind_board": ROBOT_ARM_CAMERA,
    "white-side": ROBOT_ARM_CAMERA,
    "white_side": ROBOT_ARM_CAMERA,
    "bird": BIRDS_EYE_CAMERA,
    "birds-eye": BIRDS_EYE_CAMERA,
    "birds_eye": BIRDS_EYE_CAMERA,
    "bird's_eye": BIRDS_EYE_CAMERA,
    "bird-eye": BIRDS_EYE_CAMERA,
    "bird_eye": BIRDS_EYE_CAMERA,
    "overhead": BIRDS_EYE_CAMERA,
}


def _normalize_camera(camera):
    key = camera.strip().lower().replace(" ", "_")
    if key in CAMERA_ALIASES:
        return CAMERA_ALIASES[key]

    valid = ", ".join(sorted(TRAINERS))
    raise ValueError(f"Unknown camera '{camera}'. Choose one of: {valid}")


def _prompt_for_camera():
    print("Choose the camera to train for:")
    print("  1. robot_arm  - robot arm camera behind the board on white side")
    print("  2. birds_eye  - bird's-eye view camera")

    camera = input("Camera: ")
    if camera.strip() == "1":
        return ROBOT_ARM_CAMERA
    if camera.strip() == "2":
        return BIRDS_EYE_CAMERA

    return _normalize_camera(camera)


def train(camera=ROBOT_ARM_CAMERA):
    camera = _normalize_camera(camera)
    return TRAINERS[camera]()


def main():
    parser = argparse.ArgumentParser(description="Train a chess piece detector for a specific camera.")
    parser.add_argument(
        "camera",
        nargs="?",
        help="Camera to train for: robot_arm or birds_eye. Aliases like overhead and behind_board also work.",
    )
    args = parser.parse_args()

    camera = _normalize_camera(args.camera) if args.camera else _prompt_for_camera()
    print(f"Training model for {camera} camera.")
    TRAINERS[camera]()


if __name__ == "__main__":
    main()
