import cv2

class Camera:
    def __init__(self, index=1, width=None, height=None, fourcc=None):
        self.index = index
        self.width = width
        self.height = height
        self.fourcc = fourcc
        self.cap = None

    def open(self):
        self.cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)

        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera index {self.index}")

        if self.fourcc is not None:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))

        if self.width is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)

        if self.height is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def resolution(self):
        if self.cap is None:
            raise RuntimeError("Camera has not been opened")

        return (
            int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    def read(self):
        if self.cap is None:
            raise RuntimeError("Camera has not been opened")

        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to read frame from camera")

        return frame

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
