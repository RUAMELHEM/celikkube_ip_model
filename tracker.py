import cv2
import numpy as np
import math
from collections import deque
import config


class BalloonKalmanFilter:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        dt = 1.0 / config.FPS_TARGET
        self.kf.transitionMatrix = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], np.float32)

        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], np.float32)
        self.kf.processNoiseCov = np.array([
            [1e-2, 0, 0, 0],
            [0, 1e-2, 0, 0],
            [0, 0, 5.0, 0],
            [0, 0, 0, 5.0]
        ], np.float32)
        self.kf.measurementNoiseCov = np.array([
            [1e-1, 0],
            [0, 1e-1]
        ], np.float32)
        
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)

        self.initialized = False
        self.lost_frames = 0
        self.predicted_state = None
        self.velocity = (0.0, 0.0)

    def update(self, coord):
        """Kameradan gelen yeni koordinatla Kalman filtresini güncelle."""
        measurement = np.array([[np.float32(coord[0])], [np.float32(coord[1])]])
        if not self.initialized:
            self.kf.statePost = np.array([
                [np.float32(coord[0])],
                [np.float32(coord[1])],
                [0.0],
                [0.0]
            ], np.float32)
            self.initialized = True
            self.lost_frames = 0
            return coord

        estimated = self.kf.correct(measurement)
        self.lost_frames = 0
        self.velocity = (float(estimated[2][0]), float(estimated[3][0]))
        return int(estimated[0][0]), int(estimated[1][0])

    def predict(self):
        if not self.initialized:
            return None
        self.predicted_state = self.kf.predict()
        self.lost_frames += 1
        pos_x = int(self.predicted_state[0][0])
        pos_y = int(self.predicted_state[1][0])
        self.velocity = (float(self.predicted_state[2][0]), float(self.predicted_state[3][0]))
        return pos_x, pos_y

    def get_lead_point(self, lead_seconds=config.KALMAN_LEAD_TIME):
        if not self.initialized or self.predicted_state is None:
            return None
        px = self.predicted_state[0][0]
        py = self.predicted_state[1][0]
        vx, vy = self.velocity
        lead_x = int(px + vx * lead_seconds * config.FPS_TARGET)
        lead_y = int(py + vy * lead_seconds * config.FPS_TARGET)
        return lead_x, lead_y

    def is_active(self):
        return self.initialized and (self.lost_frames <= config.MAX_TRACK_LOST_FRAMES)

    def reset(self):
        self.initialized = False
        self.lost_frames = 0
        self.predicted_state = None
        self.velocity = (0.0, 0.0)


class GeometricFilter:
    @staticmethod
    def evaluate_contour(contour):
        area = cv2.contourArea(contour)
        if area < config.MIN_BALLOON_AREA or area > config.MAX_BALLOON_AREA:
            return False, 0.0, 0.0

        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return False, 0.0, 0.0
        circularity = (4.0 * math.pi * area) / (perimeter * perimeter)

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / max(1, h)

        is_valid = (
            circularity >= config.MIN_CIRCULARITY and
            config.MIN_ASPECT_RATIO <= aspect_ratio <= config.MAX_ASPECT_RATIO
        )
        return is_valid, circularity, aspect_ratio


class TemporalDebounceFilter:
    def __init__(self, window_size=config.DEBOUNCE_WINDOW_SIZE, threshold=config.DEBOUNCE_THRESHOLD):
        self.window_size = window_size
        self.threshold = threshold
        self.history_red = deque(maxlen=window_size)
        self.history_blue = deque(maxlen=window_size)

    def update(self, has_red, has_blue):
        self.history_red.append(1 if has_red else 0)
        self.history_blue.append(1 if has_blue else 0)

    def is_red_alarm_active(self):
        return sum(self.history_red) >= self.threshold

    def is_blue_active(self):
        return sum(self.history_blue) >= self.threshold

    def get_confidence_rate(self):
        red_rate = sum(self.history_red) / max(1, len(self.history_red))
        blue_rate = sum(self.history_blue) / max(1, len(self.history_blue))
        return red_rate, blue_rate


class ThreatPrioritizer:
    @staticmethod
    def select_primary_target(detections, frame_center):
        if not detections:
            return None

        cx, cy = frame_center
        scored = []
        for det in detections:
            tx, ty = det["center"]
            area = det.get("area", 2000)
            target_type = det["type"]
            dist = math.hypot(tx - cx, ty - cy)

            threat_score = (1000 if target_type == "red_balloon" else 200)
            threat_score += (area * 0.05) - (dist * 0.5)

            scored.append((threat_score, det))

        # En yüksek puanlı hedefi birinci seç
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]
