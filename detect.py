import cv2
from ultralytics import YOLO

from board_map import build_board_map

class ChessDetector:
    def __init__(
        self,
        model_path,
        confidence_threshold=0.25,
        max_detections=50,
        image_size=640,
        class_thresholds=None,
        white_piece_value_threshold=120,
    ):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.max_detections = max_detections
        self.image_size = image_size
        self.class_thresholds = class_thresholds or {
            "white-pawn": 0.10,
            "black-rook": 0.35,
        }
        self.white_piece_value_threshold = white_piece_value_threshold
        self.white_pawn_class_id = self._class_id_for("white-pawn")

    def _class_id_for(self, class_name):
        names = self.model.names.items() if hasattr(self.model.names, "items") else enumerate(self.model.names)
        for class_id, name in names:
            if name == class_name:
                return class_id
        raise ValueError(f"Model does not contain class: {class_name}")

    def _minimum_model_confidence(self):
        thresholds = [self.confidence_threshold, *self.class_thresholds.values()]
        return min(thresholds)

    def _class_threshold(self, class_name):
        return self.class_thresholds.get(class_name, self.confidence_threshold)

    def _mean_crop_value(self, frame, bbox):
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = bbox

        # Use the center of the detection so the board color has less influence.
        crop_width = max(1, x2 - x1)
        crop_height = max(1, y2 - y1)
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        half_w = max(1, int(crop_width * 0.35))
        half_h = max(1, int(crop_height * 0.35))

        left = max(0, center_x - half_w)
        right = min(width, center_x + half_w)
        top = max(0, center_y - half_h)
        bottom = min(height, center_y + half_h)

        if right <= left or bottom <= top:
            return 0.0

        crop = frame[top:bottom, left:right]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        return float(hsv[:, :, 2].mean())

    def _adjust_class(self, frame, class_id, class_name, bbox):
        mean_value = self._mean_crop_value(frame, bbox)
        adjusted_class_id = class_id
        adjusted_class_name = class_name
        color_corrected = False

        if class_name == "black-rook" and mean_value >= self.white_piece_value_threshold:
            adjusted_class_id = self.white_pawn_class_id
            adjusted_class_name = "white-pawn"
            color_corrected = True

        return adjusted_class_id, adjusted_class_name, mean_value, color_corrected

    def detect(self, frame):
        results = self.model(
            frame,
            conf=self._minimum_model_confidence(),
            max_det=self.max_detections,
            imgsz=self.image_size,
            verbose=False,
        )

        detections = []

        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            class_name = self.model.names[cls_id]
            bbox = (int(x1), int(y1), int(x2), int(y2))

            adjusted_cls_id, adjusted_class_name, mean_value, color_corrected = self._adjust_class(
                frame,
                cls_id,
                class_name,
                bbox,
            )

            if conf < self._class_threshold(adjusted_class_name):
                continue

            detections.append({
                "class_id": adjusted_cls_id,
                "class_name": adjusted_class_name,
                "confidence": conf,
                "raw_class_id": cls_id,
                "raw_class_name": class_name,
                "raw_confidence": conf,
                "mean_value": mean_value,
                "color_corrected": color_corrected,
                "bbox": {
                    "x1": bbox[0],
                    "y1": bbox[1],
                    "x2": bbox[2],
                    "y2": bbox[3],
                },
            })

        return results, detections

    def build_board_map(self, detections, **board_geometry):
        return build_board_map(detections, **board_geometry)

    def draw_results(self, frame, detections):
        annotated = frame.copy()

        for detection in detections:
            bbox = detection["bbox"]
            x1 = bbox["x1"]
            y1 = bbox["y1"]
            x2 = bbox["x2"]
            y2 = bbox["y2"]

            color = (255, 255, 255) if detection["class_name"].startswith("white") else (0, 255, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            label = f'{detection["class_name"]} {detection["confidence"]:.2f}'
            if detection["color_corrected"]:
                label += " adjusted"

            text_y = max(20, y1 - 8)
            cv2.putText(
                annotated,
                label,
                (x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )

        return annotated
