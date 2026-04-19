import cv2
from board_map import build_board_map_with_confidences, format_board_map
from camera import Camera
from contour_features import ContourFeatureDetector
from detect import ChessDetector

FRAME_SIZE = 1080
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080
CAMERA_FOURCC = "MJPG"
ROBOT_CAMERA_INDEX = 1
BIRDS_EYE_CAMERA_INDEX = 2

def crop_to_square(frame, output_size=FRAME_SIZE):
    height, width = frame.shape[:2]
    side = min(width, height)
    x1 = (width - side) // 2
    y1 = (height - side) // 2
    square = frame[y1:y1 + side, x1:x1 + side]

    if square.shape[0] != output_size or square.shape[1] != output_size:
        square = cv2.resize(square, (output_size, output_size), interpolation=cv2.INTER_LINEAR)

    return square

def create_camera(index):
    return Camera(
        index=index,
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
        fourcc=CAMERA_FOURCC,
    )

def detect_and_map(detector, frame):
    frame = crop_to_square(frame)
    _, detections = detector.detect(frame)
    height, width = frame.shape[:2]
    board, confidences = build_board_map_with_confidences(
        detections,
        board_bbox=(0, 0, width, height),
    )
    return frame, detections, board, confidences

def detect_grasp_features(feature_detector, frame):
    frame = crop_to_square(frame)
    curve_stacks, fallback_features = feature_detector.detect_curve_stacks(frame)
    return frame, curve_stacks, fallback_features

def format_curve_stacks(stacks, limit=5):
    lines = []
    for index, stack in enumerate(stacks[:limit], start=1):
        bbox = stack["bbox"]
        mean_horizontal_score = sum(
            feature["horizontal_score"] for feature in stack["features"]
        ) / stack["curve_count"]
        mean_compactness_score = sum(
            feature["compactness_score"] for feature in stack["features"]
        ) / stack["curve_count"]
        lines.append(
            f'{index}. center={stack["center"]} '
            f'bbox=({bbox["x"]}, {bbox["y"]}, {bbox["width"]}, {bbox["height"]}) '
            f'curves={stack["curve_count"]} '
            f'total_area={stack["total_area"]:.0f} '
            f'mean_score={stack["mean_curve_score"]:.2f} '
            f'horizontal={mean_horizontal_score:.2f} '
            f'circular={mean_compactness_score:.2f} '
            f'containment={stack["min_contour_box_containment"]:.2f}'
        )
    return "\n".join(lines) if lines else "No stacked curve contours detected."

def main():
    robot_camera = create_camera(ROBOT_CAMERA_INDEX)
    birds_eye_camera = create_camera(BIRDS_EYE_CAMERA_INDEX)
    detector = ChessDetector(
        model_path="runs/detect/runs/chess_pieces_white_tuned/weights/best.pt",
        confidence_threshold=0.25,
        image_size=1088,
    )
    feature_detector = ContourFeatureDetector()

    try:
        robot_camera.open()
        birds_eye_camera.open()

        robot_width, robot_height = robot_camera.resolution()
        birds_eye_width, birds_eye_height = birds_eye_camera.resolution()
        print(
            f"Robot camera {ROBOT_CAMERA_INDEX} opened at {robot_width}x{robot_height}."
        )
        print(
            f"Bird's-eye camera {BIRDS_EYE_CAMERA_INDEX} opened at "
            f"{birds_eye_width}x{birds_eye_height}. "
            "Press ESC to quit."
        )
        frame_count = 0

        while True:
            frame_count += 1
            robot_frame, robot_curve_stacks, robot_fallback_features = detect_grasp_features(
                feature_detector,
                robot_camera.read(),
            )
            birds_eye_frame, birds_eye_detections, birds_eye_board, _ = detect_and_map(
                detector,
                birds_eye_camera.read(),
            )

            if frame_count % 5 == 0:
                print("\nBird's-eye board map:")
                print(format_board_map(birds_eye_board))
                print("\nRobot-side grasp features:")
                print(format_curve_stacks(robot_curve_stacks))

            # for detection in detections:
            #     print(detection)

            robot_annotated = feature_detector.draw_curve_stacks(
                robot_frame,
                robot_curve_stacks,
                fallback_features=robot_fallback_features,
            )
            birds_eye_annotated = detector.draw_results(birds_eye_frame, birds_eye_detections)
            cv2.imshow("Robot Camera Grasp Features", robot_annotated)
            cv2.imshow("Bird's-Eye Camera Detection", birds_eye_annotated)

            if cv2.waitKey(1) & 0xFF == 27:
                break

    finally:
        robot_camera.release()
        birds_eye_camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
