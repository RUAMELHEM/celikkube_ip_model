# Çelik Kubbe (Steel Dome) - Konfigürasyon Dosyası
import cv2

# Kamera Ayarları
CAMERA_INDEX = 0          # 0: Dahili/Birincil Webcam, 1: İkincil USB Kamera
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FPS_TARGET = 30

# Hedef ve Uyarı Tanımları
# Kırmızı Balon: DÜŞMAN / TEHDİT (THREAT / HOSTILE)
# Mavi Balon: DOST / MÜTTEFİK (FRIENDLY / NEUTRAL)
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
        "color_bgr": (255, 140, 0),     # Turkuaz / Mavi
        "alert_level": "WARNING",
        "sound_alert": False,
        "sound_freq": 800,
        "sound_duration": 100
    }
}

# -------------------------------------------------------------
# 1. OPTİK & GÖRÜNTÜ İŞLEME FİLTRELERİ
# -------------------------------------------------------------
USE_CLAHE = True              # Açık hava güneş parlamasını ve gölgeleri dengeleme
CLAHE_CLIP_LIMIT = 2.0
CLAHE_GRID_SIZE = (8, 8)
GAUSSIAN_BLUR_KERNEL = (5, 5)

# -------------------------------------------------------------
# 2. GEOMETRİK VE DAİRESELLİK FİLTRESİ (Sahte Hedef / Tişört Önleme)
# -------------------------------------------------------------
# Dairesellik: 4 * pi * Alan / (Çevre^2). Tam daire = 1.0, elips ~ 0.75 - 0.95
MIN_CIRCULARITY = 0.55        # 0.55 altındaki nesneler (tişört, levha, bayrak) elenir
MIN_BALLOON_AREA = 900        # Minimum balon piksel alanı (Çok uzak gürültüleri eler)
MAX_BALLOON_AREA = 250000     # Maksimum alan (Ekranı kaplayan devasa sahte renkleri eler)
MIN_ASPECT_RATIO = 0.55       # w / h oranı
MAX_ASPECT_RATIO = 1.80

# -------------------------------------------------------------
# 3. ZAMANSAL KARAR FİLTRESİ (Temporal Debounce)
# -------------------------------------------------------------
# Anlık 1 karelik parlamalarda alarm ötmesini engeller
DEBOUNCE_WINDOW_SIZE = 8      # İncelenecek son kare sayısı
DEBOUNCE_THRESHOLD = 4        # Son 8 karenin en az 4'ünde tespit varsa ALARM aktif

# -------------------------------------------------------------
# 4. KALMAN FİLTRESİ VE HEDEF TAHMİNİ (Tracking & Lead Pursuit)
# -------------------------------------------------------------
KALMAN_LEAD_TIME = 0.4        # Kaç saniye sonrasının hedefleneceği (Öncü vuruş süresi)
MAX_TRACK_LOST_FRAMES = 12    # Hedef kamerada kaybolursa kaç kare boyunca Kalman tahmini sürsün

# -------------------------------------------------------------
# MODEL VE RENK PARAMETRELERİ
# -------------------------------------------------------------
MODEL_PATH = "best.pt"
CONFIDENCE_THRESHOLD = 0.28

# Renk Tabanlı Fallback (HSV) Eşikleri
HSV_RANGES = {
    "red_balloon": [
        ((0, 110, 70), (10, 255, 255)),
        ((170, 110, 70), (180, 255, 255))
    ],
    "blue_balloon": [
        ((95, 100, 60), (135, 255, 255))
    ]
}
