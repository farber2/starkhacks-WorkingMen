import cv2
import platform
import time

class Camera:
    def __init__(self, index=1, width=None, height=None, fourcc=None):
        self.index = index
        self.width = width
        self.height = height
        self.fourcc = fourcc
        self.cap = None

    def open(self):
        # Try platform-appropriate backends first, then generic fallback.
        system = platform.system().lower()
        backend_candidates = []
        if system == "windows":
            backend_candidates = [cv2.CAP_DSHOW, cv2.CAP_MSMF, None]
        elif system == "darwin":
            backend_candidates = [cv2.CAP_AVFOUNDATION, None]
        else:
            backend_candidates = [cv2.CAP_V4L2, None]

        self.cap = None
        for backend in backend_candidates:
            cap = cv2.VideoCapture(self.index) if backend is None else cv2.VideoCapture(self.index, backend)
            if cap.isOpened():
                self.cap = cap
                break
            cap.release()

        if self.cap is None:
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

        # Continuity/external cameras on macOS can fail transiently right after open.
        for _ in range(8):
            ret, frame = self.cap.read()
            if ret and frame is not None and frame.size > 0:
                return frame
            time.sleep(0.03)

        raise RuntimeError("Failed to read frame from camera")

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
