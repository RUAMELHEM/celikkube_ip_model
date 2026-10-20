import cv2
CAMERA_INDEX = 0          
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FPS_TARGET = 30

TARGET_CONFIG = {
    "red_balloon": {
        "display_name": "DUSMAN: KIRMIZI BALON",
        "color_bgr": (0, 0, 255),       # Kırmızı
        "alert_level": "CRITICAL",
        "sound_alert": True,
        "sound_freq": 1600,              # Hz
        "sound_duration": 150            # ms
    },
    "blue_balloon": {
        "display_name": "DOST: MAVI BALON",
        "color_bgr": (255, 140, 0),     # Mavi
        "alert_level": "WARNING",
        "sound_alert": False,
        "sound_freq": 800,
        "sound_duration": 100
    }
}

USE_CLAHE = True              
CLAHE_CLIP_LIMIT = 2.0
CLAHE_GRID_SIZE = (8, 8)
GAUSSIAN_BLUR_KERNEL = (5, 5)

MIN_CIRCULARITY = 0.55       
MIN_BALLOON_AREA = 900       
MAX_BALLOON_AREA = 250000     
MIN_ASPECT_RATIO = 0.55       
MAX_ASPECT_RATIO = 1.80


DEBOUNCE_WINDOW_SIZE = 8      
DEBOUNCE_THRESHOLD = 4       

KALMAN_LEAD_TIME = 0.4      
MAX_TRACK_LOST_FRAMES = 12    


MODEL_PATH = "best.pt"
CONFIDENCE_THRESHOLD = 0.28

HSV_RANGES = {
    "red_balloon": [
        ((0, 110, 70), (10, 255, 255)),
        ((170, 110, 70), (180, 255, 255))
    ],
    "blue_balloon": [
        ((95, 100, 60), (135, 255, 255))
    ]
}
