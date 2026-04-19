import cv2
import numpy as np


def _horizontal_overlap(left_feature, right_feature):
    left_bbox = left_feature["bbox"]
    right_bbox = right_feature["bbox"]
    left_start = left_bbox["x"]
    left_end = left_bbox["x"] + left_bbox["width"]
    right_start = right_bbox["x"]
    right_end = right_bbox["x"] + right_bbox["width"]
    overlap = max(0, min(left_end, right_end) - max(left_start, right_start))
    smaller_width = max(1, min(left_bbox["width"], right_bbox["width"]))
    return overlap / smaller_width


def _clamp(value, lower=0.0, upper=1.0):
    return max(lower, min(upper, value))


def _contour_containment(contour, bbox):
    points = contour.reshape(-1, 2)
    left = bbox["x"]
    top = bbox["y"]
    right = left + bbox["width"]
    bottom = top + bbox["height"]
    inside = (
        (points[:, 0] >= left)
        & (points[:, 0] <= right)
        & (points[:, 1] >= top)
        & (points[:, 1] <= bottom)
    )
    return float(np.count_nonzero(inside) / max(1, len(points)))


class ContourFeatureDetector:
    def __init__(
        self,
        min_area=250,
        max_area_ratio=0.25,
        blur_kernel_size=5,
        canny_low=40,
        canny_high=130,
        horizontal_bias=1.5,
        circular_bias=1.5,
        min_box_containment=0.70,
    ):
        self.min_area = min_area
        self.max_area_ratio = max_area_ratio
        self.blur_kernel_size = blur_kernel_size
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.horizontal_bias = horizontal_bias
        self.circular_bias = circular_bias
        self.min_box_containment = min_box_containment

    def _preprocess(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        blurred = cv2.GaussianBlur(
            gray,
            (self.blur_kernel_size, self.blur_kernel_size),
            0,
        )
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)
        kernel = np.ones((3, 3), dtype=np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        edges = cv2.dilate(edges, kernel, iterations=1)
        return edges

    def detect(self, frame):
        mask = self._preprocess(frame)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        frame_area = frame.shape[0] * frame.shape[1]
        max_area = frame_area * self.max_area_ratio
        features = []
        hierarchy = hierarchy[0] if hierarchy is not None else []

        for index, contour in enumerate(contours):
            area = float(cv2.contourArea(contour))
            if area < self.min_area or area > max_area:
                continue

            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue

            center_x = int(moments["m10"] / moments["m00"])
            center_y = int(moments["m01"] / moments["m00"])
            x, y, width, height = cv2.boundingRect(contour)
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect).astype(int)
            perimeter = float(cv2.arcLength(contour, True))
            if perimeter <= 0:
                continue

            hull = cv2.convexHull(contour)
            hull_area = float(cv2.contourArea(hull))
            solidity = area / hull_area if hull_area > 0 else 0.0
            circularity = (4.0 * np.pi * area) / (perimeter * perimeter)
            bbox_aspect_ratio = width / max(1, height)
            horizontal_score = _clamp((bbox_aspect_ratio - 1.0) / 2.0)
            compactness_score = _clamp(circularity)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            ellipse = None
            ellipse_aspect_ratio = 0.0

            if len(contour) >= 5:
                ellipse_rect = cv2.fitEllipse(contour)
                ellipse_width, ellipse_height = ellipse_rect[1]
                larger_axis = max(ellipse_width, ellipse_height)
                smaller_axis = min(ellipse_width, ellipse_height)
                if larger_axis > 0:
                    ellipse_aspect_ratio = smaller_axis / larger_axis
                ellipse = {
                    "center": tuple(float(value) for value in ellipse_rect[0]),
                    "axes": tuple(float(value) for value in ellipse_rect[1]),
                    "angle": float(ellipse_rect[2]),
                }

            child_count = 0
            if len(hierarchy) > index:
                child_index = hierarchy[index][2]
                while child_index != -1:
                    child_count += 1
                    child_index = hierarchy[child_index][0]

            is_curve = len(approx) >= 6 or ellipse is not None
            curve_score = (
                0.25 * min(1.0, len(approx) / 12.0)
                + self.circular_bias * 0.30 * compactness_score
                + self.circular_bias * 0.20 * ellipse_aspect_ratio
                + self.horizontal_bias * 0.35 * horizontal_score
                + 0.10 * min(1.0, child_count / 2.0)
            )

            features.append(
                {
                    "center": (center_x, center_y),
                    "bbox": {
                        "x": int(x),
                        "y": int(y),
                        "width": int(width),
                        "height": int(height),
                    },
                    "area": area,
                    "perimeter": perimeter,
                    "angle": float(rect[2]),
                    "min_area_box": box.tolist(),
                    "solidity": solidity,
                    "circularity": circularity,
                    "bbox_aspect_ratio": bbox_aspect_ratio,
                    "horizontal_score": horizontal_score,
                    "compactness_score": compactness_score,
                    "approx_vertices": int(len(approx)),
                    "ellipse": ellipse,
                    "ellipse_aspect_ratio": ellipse_aspect_ratio,
                    "child_count": child_count,
                    "is_curve": is_curve,
                    "curve_score": curve_score,
                    "contour": contour,
                }
            )

        features.sort(key=lambda feature: feature["curve_score"], reverse=True)
        return features

    def detect_curve_stacks(self, frame, min_curves_per_stack=2):
        features = [
            feature
            for feature in self.detect(frame)
            if feature["is_curve"] and feature["approx_vertices"] >= 5
        ]
        features.sort(key=lambda feature: feature["center"][1])
        stacks = []

        for feature in features:
            center_x, _ = feature["center"]
            added = False

            for stack in stacks:
                stack_center_x = stack["center"][0]
                stack_bbox = stack.get("raw_bbox", stack["bbox"])
                stack_width = max(1, stack_bbox["width"])
                x_close = abs(center_x - stack_center_x) <= max(20, stack_width * 0.20)
                overlaps = any(_horizontal_overlap(feature, member) >= 0.55 for member in stack["features"])

                if x_close and overlaps:
                    stack["features"].append(feature)
                    self._refresh_stack(stack)
                    added = True
                    break

            if not added:
                stacks.append({"features": [feature]})
                self._refresh_stack(stacks[-1])

        stacks = [
            stack
            for stack in stacks
            if len(stack["features"]) >= min_curves_per_stack
            and stack["min_contour_box_containment"] >= self.min_box_containment
        ]
        stacks.sort(key=lambda stack: stack["curve_count"], reverse=True)
        return stacks, features

    def _refresh_stack(self, stack):
        features = stack["features"]
        left = min(feature["bbox"]["x"] for feature in features)
        top = min(feature["bbox"]["y"] for feature in features)
        right = max(feature["bbox"]["x"] + feature["bbox"]["width"] for feature in features)
        bottom = max(feature["bbox"]["y"] + feature["bbox"]["height"] for feature in features)
        total_area = sum(feature["area"] for feature in features)
        weighted_center_x = sum(feature["center"][0] * feature["area"] for feature in features) / total_area
        weighted_center_y = sum(feature["center"][1] * feature["area"] for feature in features) / total_area

        mean_curve_width = sum(feature["bbox"]["width"] for feature in features) / len(features)
        mean_curve_height = sum(feature["bbox"]["height"] for feature in features) / len(features)
        raw_width = right - left
        raw_height = bottom - top
        square_center_x = (left + right) / 2.0
        square_center_y = (top + bottom) / 2.0
        square_side = int(max(20, max(mean_curve_width, mean_curve_height) * 1.15))
        max_square_side = int(max(raw_width, raw_height))

        stack["raw_bbox"] = {
            "x": int(left),
            "y": int(top),
            "width": int(right - left),
            "height": int(bottom - top),
        }

        while True:
            square_left = int(square_center_x - square_side / 2)
            square_top = int(square_center_y - square_side / 2)
            stack["bbox"] = {
                "x": square_left,
                "y": square_top,
                "width": square_side,
                "height": square_side,
            }
            containment_scores = [
                _contour_containment(feature["contour"], stack["bbox"])
                for feature in features
            ]
            if (
                min(containment_scores) >= self.min_box_containment
                or square_side >= max_square_side
            ):
                break
            square_side = min(max_square_side, int(square_side * 1.10) + 1)

        stack["center"] = (int(weighted_center_x), int(weighted_center_y))
        stack["curve_count"] = len(features)
        stack["total_area"] = float(total_area)
        stack["min_contour_box_containment"] = float(min(containment_scores))
        stack["mean_contour_box_containment"] = float(
            sum(containment_scores) / len(containment_scores)
        )
        stack["mean_curve_score"] = float(
            sum(feature["curve_score"] for feature in features) / len(features)
        )

    def draw_results(self, frame, features, max_features=12):
        annotated = frame.copy()

        for feature in features[:max_features]:
            contour = feature["contour"]
            center_x, center_y = feature["center"]
            box = np.array(feature["min_area_box"], dtype=np.int32)

            cv2.drawContours(annotated, [contour], -1, (0, 255, 255), 2)
            cv2.drawContours(annotated, [box], 0, (255, 0, 0), 2)
            cv2.circle(annotated, (center_x, center_y), 5, (0, 0, 255), -1)

            label = (
                f'area {feature["area"]:.0f} '
                f'h {feature["horizontal_score"]:.2f} '
                f'c {feature["compactness_score"]:.2f}'
            )
            cv2.putText(
                annotated,
                label,
                (center_x + 8, center_y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        return annotated

    def draw_curve_stacks(self, frame, stacks, fallback_features=None, max_stacks=6):
        annotated = frame.copy()
        colors = [
            (0, 255, 255),
            (255, 0, 255),
            (0, 200, 0),
            (255, 120, 0),
            (0, 120, 255),
            (180, 255, 0),
        ]

        for stack_index, stack in enumerate(stacks[:max_stacks]):
            color = colors[stack_index % len(colors)]
            bbox = stack["bbox"]
            center = stack["center"]
            cv2.rectangle(
                annotated,
                (bbox["x"], bbox["y"]),
                (bbox["x"] + bbox["width"], bbox["y"] + bbox["height"]),
                color,
                2,
            )
            cv2.circle(annotated, center, 6, (0, 0, 255), -1)

            for feature in stack["features"]:
                cv2.drawContours(annotated, [feature["contour"]], -1, color, 2)

            label = f'curves {stack["curve_count"]} score {stack["mean_curve_score"]:.2f}'
            cv2.putText(
                annotated,
                label,
                (bbox["x"], max(20, bbox["y"] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )

        if not stacks and fallback_features is not None:
            annotated = self.draw_results(annotated, fallback_features)

        return annotated
