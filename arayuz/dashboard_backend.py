# -*- coding: utf-8 -*-
"""
TEKNOFEST 2027 Çelikkubbe — Taret Kontrol & Hava Savunma Dashboard Backend
============================================================================
WebSocket sunucu: Kamera → YOLO V5 P2 → ROS2 bridge → GUI arasında köprü.

Başlatma:
    python dashboard_backend.py --source 0 --model ../../models/Phase5_Kaggle/best.pt

ROS2 kurulu değilse fallback modunda çalışır (AI + GUI aktif, motor komutları loglanır).
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import http.server
import json
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import cv2
import numpy as np

# ─── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("CelikkubbeDashboard")

# ─── ROS2 Optional Import ────────────────────────────────────────────────────
ROS2_AVAILABLE = False
try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Bool, Float32, Int16, Int32, String
    ROS2_AVAILABLE = True
    log.info("✅ ROS2 (rclpy) mevcut — gerçek donanım modu aktif")
except ImportError:
    log.warning("⚠️  ROS2 bulunamadı — fallback (AI+GUI) modunda çalışılıyor")

# ─── YOLO Optional Import ────────────────────────────────────────────────────
YOLO_AVAILABLE = False
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    log.info("✅ Ultralytics YOLO mevcut")
except ImportError:
    log.warning("⚠️  ultralytics yüklü değil — simülasyon modu (pip install ultralytics)")

# ─── Keyboard Shortcut Import ────────────────────────────────────────────────
try:
    import keyboard
    KEYBOARD_AVAILABLE = True
except ImportError:
    KEYBOARD_AVAILABLE = False

# ─── WebSockets ──────────────────────────────────────────────────────────────
try:
    import websockets
except ImportError:
    log.error("❌ websockets paketi eksik: pip install websockets")
    sys.exit(1)

# ─── Çelikkubbe Savunma Algoritmaları ve Ayarları ────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    import config
    from tracker import BalloonKalmanFilter, GeometricFilter, TemporalDebounceFilter, ThreatPrioritizer
    log.info("✅ Çelikkubbe Savunma Motoru (Kalman, Geometrik, Debounce, Tehdit) yüklendi")
except ImportError as exc:
    log.warning(f"⚠️  tracker/config yüklenemedi: {exc}")
    config = None
    BalloonKalmanFilter = None
    GeometricFilter = None
    TemporalDebounceFilter = None
    ThreatPrioritizer = None

# Windows sesli alarm desteği
try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False

class SoundAlertManager:
    """Arayüzü dondurmadan arka planda sesli uyarı veren yönergeci."""
    def __init__(self):
        self.is_playing = False
        self.enabled = True
        self.last_played = 0

    def trigger(self, freq=1600, duration=150, cooldown=0.35):
        if not self.enabled or not HAS_WINSOUND:
            return
        current_time = time.time()
        if self.is_playing or (current_time - self.last_played < cooldown):
            return

        def _play():
            self.is_playing = True
            try:
                winsound.Beep(freq, duration)
            except Exception:
                pass
            finally:
                self.is_playing = False
                self.last_played = time.time()

        threading.Thread(target=_play, daemon=True).start()


# =============================================================================
# GLOBAL SYSTEM STATE
# =============================================================================

@dataclass
class SystemState:
    """Tüm sistem durumunu tutan merkezi veri yapısı."""
    # Telemetri — AI & Hedef (None = ölçüm yok; uydurma 0.0 gösterme)
    target_class: str = "BELIRSIZ"           # ENEMY_RED / FRIEND_BLUE / BELIRSIZ
    target_label: str = "BELİRSİZ"          # Türkçe: DÜŞMAN / DOST / BELİRSİZ
    target_label_color: str = "#fbbf24"      # CSS rengi
    confidence: Optional[float] = None
    distance_m: Optional[float] = None       # Mesafe henüz hesaplanmıyor → null
    track_id: int = -1
    turret_error_dx: float = 0.0
    turret_error_dy: float = 0.0
    fire_enable: bool = False
    fps: float = 0.0
    tracks_count: int = 0

    # Telemetri — Donanım
    motor_x_pos: int = 0
    motor_y_pos: int = 0
    temperature: Optional[float] = None      # ROS2 gelmeden sahte 0.0 yok
    fan_speed: Optional[int] = None
    servo_angle: int = 90
    laser_power: int = 0

    # Sistem / kaynak dürüstlüğü
    emergency: bool = False
    ros2_status: str = "offline"             # online / offline / fallback
    mode: str = "OTONOM"                     # OTONOM / MANUEL
    frame_b64: str = ""
    active_source: str = "synthetic"
    active_source_label: str = "Sentetik (kamera kapalı)"
    source_kind: str = "synthetic"           # camera | video | synthetic | failed
    camera_connected: bool = False           # fiziksel kamera gerçekten açık mı
    stream_ok: bool = False                  # bu kare için geçerli görüntü var mı
    vision_ok: bool = False                  # YOLO sonucu anlamlı mı (canlı akış + model)
    available_sources: List[Dict[str, str]] = field(default_factory=list)
    log_entries: List[str] = field(default_factory=list)

    # PID Parametreleri
    pid_horizontal: Dict[str, float] = field(default_factory=lambda: {"kp": 0.03, "ki": 0.01, "kd": 0.30, "dz": 0})
    pid_vertical: Dict[str, float] = field(default_factory=lambda: {"kp": 0.04, "ki": 0.015, "kd": 0.30, "dz": 0})

    # Mekanik Hız Ayarları
    pan_speed: int = 5
    tilt_speed: int = 5
    fire_delay_ms: int = 200
    manual_step: int = 30

    def add_log(self, msg: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        entry = f"[{ts}] {msg}"
        self.log_entries.append(entry)
        if len(self.log_entries) > 100:
            self.log_entries = self.log_entries[-100:]
        log.info(msg)

    def clear_detections(self, reason: str = "") -> None:
        """Kamera/akış yokken uydurma hedef/doğruluk bırakma."""
        self.target_class = "BELIRSIZ"
        self.target_label = "BELİRSİZ"
        self.target_label_color = "#fbbf24"
        self.confidence = None
        self.distance_m = None
        self.track_id = -1
        self.turret_error_dx = 0.0
        self.turret_error_dy = 0.0
        self.fire_enable = False
        self.tracks_count = 0
        self.vision_ok = False
        if reason:
            self.add_log(reason)

    def to_telemetry_dict(self) -> Dict[str, Any]:
        trusted = metrics_are_trusted()
        return {
            "type": "telemetry",
            "target_class": self.target_class if trusted else "BELIRSIZ",
            "target_label": self.target_label if trusted else "BELİRSİZ",
            "target_label_color": self.target_label_color if trusted else "#fbbf24",
            # Kamera/video yokken ASLA doğruluk yüzdesi gönderme (sentetik dahil)
            "confidence": (
                None
                if (not trusted or self.confidence is None)
                else round(self.confidence, 3)
            ),
            "distance_m": None if self.distance_m is None else round(self.distance_m, 2),
            "track_id": self.track_id if trusted else -1,
            "turret_error": {
                "dx": round(self.turret_error_dx, 4) if trusted else 0.0,
                "dy": round(self.turret_error_dy, 4) if trusted else 0.0,
            },
            "fire_enable": bool(self.fire_enable and trusted and self.vision_ok),
            "fps": round(self.fps, 1) if self.stream_ok else 0.0,
            "tracks_count": self.tracks_count if trusted else 0,
            "motor_x_pos": self.motor_x_pos,
            "motor_y_pos": self.motor_y_pos,
            "temperature": None if self.temperature is None else round(self.temperature, 1),
            "fan_speed": self.fan_speed,
            "servo_angle": self.servo_angle,
            "laser_power": self.laser_power,
            "emergency": self.emergency,
            "ros2_status": self.ros2_status,
            "mode": self.mode,
            "frame_b64": self.frame_b64,
            "active_source": self.active_source,
            "active_source_label": self.active_source_label,
            "source_kind": self.source_kind,
            "camera_connected": self.camera_connected,
            "stream_ok": self.stream_ok,
            # vision_ok sadece güvenilir kaynakta true olabilir
            "vision_ok": bool(self.vision_ok and trusted),
            "metrics_trusted": trusted,
            "available_sources": self.available_sources,
            "log_entries": self.log_entries[-20:],
            "pid_horizontal": self.pid_horizontal,
            "pid_vertical": self.pid_vertical,
            "pan_speed": self.pan_speed,
            "tilt_speed": self.tilt_speed,
            "fire_delay_ms": self.fire_delay_ms,
        }


STATE = SystemState()
CLIENTS: Set[Any] = set()
VISION_PIPELINE: Optional["VisionPipeline"] = None
INSTANCE_LOCK_PATH = Path(__file__).with_name(".dashboard_backend.lock")
_INSTANCE_LOCK_FH = None


def acquire_instance_lock() -> None:
    """İkinci kopya webcam'i kilitlemesin / port çakışması olmasın."""
    global _INSTANCE_LOCK_FH
    try:
        INSTANCE_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
        _INSTANCE_LOCK_FH = open(INSTANCE_LOCK_PATH, "w", encoding="utf-8")
        _INSTANCE_LOCK_FH.write(f"pid={os.getpid()}\n")
        _INSTANCE_LOCK_FH.flush()
    except Exception as exc:
        log.warning(f"Kilit dosyası uyarısı: {exc}")


def metrics_are_trusted() -> bool:
    """Doğruluk/hedef metrikleri kamera, video ve simülasyon modunda geçerlidir."""
    return STATE.stream_ok and STATE.source_kind in ("camera", "video", "synthetic")


def _search_roots() -> List[Path]:
    """Portable kökler: paket (arayuz/..) veya repo (src/software/arayuz/../../..)."""
    here = Path(__file__).resolve().parent
    roots = [here.parent, Path.cwd()]
    if len(here.parents) >= 3:
        roots.append(here.parents[3])  # repo kökü
    # benzersiz sıra koru
    out: List[Path] = []
    seen = set()
    for r in roots:
        key = str(r.resolve()) if r.exists() else str(r)
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def discover_available_sources() -> List[Dict[str, str]]:
    """
    Sadece gerçekten var olan / portable kaynakları listele.
    Yerel makineye özel sabit test_media listesi YOK — klasör varsa eklenir.
    """
    sources: List[Dict[str, str]] = [
        {"id": "synthetic", "label": "Sentetik hedef simülatörü", "group": "Simülasyon"},
        {"id": "0", "label": "Webcam 0", "group": "Kameralar"},
        {"id": "1", "label": "Webcam 1", "group": "Kameralar"},
    ]
    seen_names = set()
    for root in _search_roots():
        media = root / "test_media"
        if not media.is_dir():
            continue
        for mp4 in sorted(media.glob("*.mp4")):
            name = mp4.name
            if name in seen_names:
                continue
            seen_names.add(name)
            # Göreli id: test_media/xxx.mp4 (açılırken köklerde aranır)
            sources.append({
                "id": f"test_media/{name}",
                "label": f"Test: {name}",
                "group": "Test videoları (bulunan)",
            })
    return sources


def resolve_media_path(source: str) -> Optional[Path]:
    """Dosya yolunu paket/repo/cwd altında çöz."""
    p = Path(source)
    if p.is_file():
        return p
    for root in _search_roots():
        cand = root / source
        if cand.is_file():
            return cand
        cand2 = root / "test_media" / Path(source).name
        if cand2.is_file():
            return cand2
    return None


def find_default_model() -> Optional[str]:
    candidates = []
    for root in _search_roots():
        candidates.extend([
            root / "best.pt",
            root / "models" / "best.pt",
            root / "models" / "Phase5_Kaggle" / "best.pt",
        ])
    # arayuz yanındaki klasik repo yolu
    here = Path(__file__).resolve().parent
    candidates.extend([
        here.parent / "best.pt",
        here.parent / "models" / "best.pt",
        here.parent.parent.parent / "models" / "Phase5_Kaggle" / "best.pt"
    ])
    for c in candidates:
        if c.is_file():
            return str(c)
    return None


# =============================================================================
# ROS2 BRIDGE NODE
# =============================================================================

class CelikkubbeROS2Bridge:
    """
    ROS2 publisher/subscriber köprüsü.
    Tüm motor/servo/laser/emergency komutlarını topic'e gönderir,
    pozisyon/sıcaklık/fan bilgilerini dinler.
    """

    def __init__(self):
        self._node: Optional[Any] = None
        self._pubs: Dict[str, Any] = {}
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self) -> None:
        if not ROS2_AVAILABLE:
            STATE.ros2_status = "fallback"
            STATE.add_log("ROS2 fallback modu: Komutlar loglanıyor, donanıma gitmiyor")
            return
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def _spin(self) -> None:
        try:
            rclpy.init()
            self._node = rclpy.create_node("celikkubbe_dashboard_bridge")

            # Publishers
            self._pubs["motor_x"] = self._node.create_publisher(Int32, "/motor_x/setpoint", 10)
            self._pubs["motor_y"] = self._node.create_publisher(Int32, "/motor_y/setpoint", 10)
            self._pubs["servo"] = self._node.create_publisher(Int16, "/servo/angle", 10)
            self._pubs["laser"] = self._node.create_publisher(Int16, "/laser/power", 10)
            self._pubs["emergency"] = self._node.create_publisher(Bool, "/system/emergency_stop", 10)
            self._pubs["fire"] = self._node.create_publisher(Bool, "/air_defense/fire_enable", 10)
            self._pubs["pid_h"] = self._node.create_publisher(String, "/pid/horizontal", 10)
            self._pubs["pid_v"] = self._node.create_publisher(String, "/pid/vertical", 10)

            # Subscribers
            self._node.create_subscription(Int32, "/motor_x/position", self._cb_motor_x, 10)
            self._node.create_subscription(Int32, "/motor_y/position", self._cb_motor_y, 10)
            self._node.create_subscription(Float32, "/temperature", self._cb_temp, 10)
            self._node.create_subscription(Int16, "/fan/speed", self._cb_fan, 10)

            STATE.ros2_status = "online"
            STATE.add_log("✅ ROS2 node aktif: celikkubbe_dashboard_bridge")

            self._running = True
            rclpy.spin(self._node)
        except Exception as exc:
            log.error(f"ROS2 bridge hatası: {exc}")
            STATE.ros2_status = "fallback"
        finally:
            if self._node:
                self._node.destroy_node()
            try:
                rclpy.shutdown()
            except Exception:
                pass

    def _cb_motor_x(self, msg) -> None:
        STATE.motor_x_pos = msg.data

    def _cb_motor_y(self, msg) -> None:
        STATE.motor_y_pos = msg.data

    def _cb_temp(self, msg) -> None:
        STATE.temperature = msg.data

    def _cb_fan(self, msg) -> None:
        STATE.fan_speed = msg.data

    def _pub(self, key: str, msg) -> None:
        if ROS2_AVAILABLE and self._node and key in self._pubs:
            self._pubs[key].publish(msg)
        else:
            log.debug(f"[FALLBACK] {key}: {msg}")

    def send_motor_x(self, value: int) -> None:
        if ROS2_AVAILABLE:
            msg = Int32(); msg.data = value
            self._pub("motor_x", msg)
        STATE.add_log(f"Motor X → {value}")

    def send_motor_y(self, value: int) -> None:
        if ROS2_AVAILABLE:
            msg = Int32(); msg.data = value
            self._pub("motor_y", msg)
        STATE.add_log(f"Motor Y → {value}")

    def send_servo(self, angle: int) -> None:
        if ROS2_AVAILABLE:
            msg = Int16(); msg.data = angle
            self._pub("servo", msg)
        STATE.servo_angle = angle
        STATE.add_log(f"Servo → {angle}°")

    def send_laser(self, power: int) -> None:
        if ROS2_AVAILABLE:
            msg = Int16(); msg.data = power
            self._pub("laser", msg)
        STATE.laser_power = power
        STATE.add_log(f"Lazer güç → {power}")

    def send_emergency_stop(self) -> None:
        if ROS2_AVAILABLE:
            msg = Bool(); msg.data = True
            self._pub("emergency", msg)
            # Tüm motorları durdur
            self.send_motor_x(0)
            self.send_motor_y(0)
            self.send_servo(90)
            self.send_laser(0)
        STATE.emergency = True
        STATE.add_log("🚨 ACİL DURDURMA AKTİF!")

    def send_emergency_reset(self) -> None:
        if ROS2_AVAILABLE:
            msg = Bool(); msg.data = False
            self._pub("emergency", msg)
        STATE.emergency = False
        STATE.add_log("✅ Acil durdurma sıfırlandı — sistem normale döndü")

    def send_pid(self, axis: str, params: Dict[str, float]) -> None:
        key = "pid_h" if axis == "horizontal" else "pid_v"
        if ROS2_AVAILABLE:
            msg = String(); msg.data = json.dumps(params)
            self._pub(key, msg)
        STATE.add_log(f"PID {axis} güncellendi: Kp={params['kp']} Ki={params['ki']} Kd={params['kd']}")

    def center_turret(self) -> None:
        self.send_motor_x(0)
        self.send_motor_y(0)
        STATE.add_log("Taret merkezlendi (0,0)")


ROS2_BRIDGE = CelikkubbeROS2Bridge()


# =============================================================================
# VISION PIPELINE
# =============================================================================

def class_to_turkish(cls: str) -> tuple[str, str]:
    """(Türkçe etiket, CSS renk kodu) döndürür."""
    if cls == "ENEMY_RED":
        return "DÜŞMAN", "#ef4444"
    if cls == "FRIEND_BLUE":
        return "DOST", "#3b82f6"
    return "BELİRSİZ", "#fbbf24"


class VisionPipeline:
    """Kamera/Video okuma + YOLO inference + frame annotation + Sentetik Hedef Simülasyonu."""

    def __init__(self, source: str, model_path: Optional[str], conf: float):
        self.source = str(source).strip()
        self.conf = conf if conf is not None else 0.28
        self.model = None
        self.cap: Optional[cv2.VideoCapture] = None
        self._fps_meter = _FPSMeter()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._synthetic_time = 0.0

        # Çelikkubbe Savunma ve İzleme Algoritmaları
        self.kalman = BalloonKalmanFilter() if BalloonKalmanFilter else None
        self.debounce = TemporalDebounceFilter(
            window_size=getattr(config, "DEBOUNCE_WINDOW_SIZE", 8) if config else 8,
            threshold=getattr(config, "DEBOUNCE_THRESHOLD", 4) if config else 4
        ) if TemporalDebounceFilter else None
        self.sound = SoundAlertManager()
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        if YOLO_AVAILABLE and model_path and Path(model_path).exists():
            log.info(f"YOLO modeli yükleniyor: {model_path}")
            self.model = YOLO(model_path)
            log.info("✅ YOLO modeli hazır")
        elif model_path:
            log.warning(f"Model bulunamadı: {model_path} — simülasyon modunda çalışılıyor")
        else:
            log.warning("Model yolu verilmedi — simülasyon modunda çalışılıyor")

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        with self._lock:
            if self.cap:
                self.cap.release()
                self.cap = None

    def _get_source_label(self, src: str) -> str:
        s = str(src).strip()
        if s == "0":
            return "📷 Webcam 0 (Dahili)"
        elif s == "1":
            return "📷 Webcam 1 (Harici)"
        elif s == "synthetic":
            return "🎮 Sentetik Simülasyon"
        elif "youtube_red" in s:
            return "🎯 Test: Kırmızı Balon (Saha Uçuş)"
        elif "matrix" in s:
            return "🎯 Test: Matrix Çoklu Balon"
        elif "216045" in s:
            return "🎯 Test: 216045 Hedef Video"
        elif "5954" in s:
            return "🎯 Test: 5954 Hedef Video"
        elif "128372" in s:
            return "🎯 Test: 128372 Hedef Video"
        elif "static" in s:
            return "🎯 Test: Sabit Sürekli Video"
        p = Path(s)
        return p.name if p.exists() else f"Kaynak: {s}"

    def switch_source(self, new_source: str) -> None:
        with self._lock:
            new_source = str(new_source).strip()
            log.info(f"Video kaynağı değiştiriliyor: {self.source} -> {new_source}")
            self.source = new_source
            if self.cap:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None

            STATE.camera_connected = False
            STATE.stream_ok = False
            STATE.clear_detections()

            self.cap = self._open_capture()
            STATE.active_source = self.source
            label = self._get_source_label(self.source)
            STATE.active_source_label = label
            if self.source == "synthetic":
                STATE.source_kind = "synthetic"
                STATE.camera_connected = False
                STATE.stream_ok = True
                STATE.add_log("🎮 Sentetik simülatör — fiziksel kamera KAPALI")
            elif self.cap and self.cap.isOpened():
                is_cam = str(self.source).isdigit()
                STATE.source_kind = "camera" if is_cam else "video"
                STATE.camera_connected = is_cam
                STATE.stream_ok = True
                STATE.add_log(f"✅ Kaynak bağlandı: {label} ({STATE.source_kind})")
            else:
                STATE.source_kind = "failed"
                STATE.camera_connected = False
                STATE.stream_ok = False
                STATE.add_log(f"❌ Kaynak açılamadı: {label} — doğruluk/hedef gizlendi")

    def _open_capture(self) -> Optional[cv2.VideoCapture]:
        """Kaynağı aç. Başarısızsa sessizce test videosuna DÜŞME — dürüst fail."""
        import os as _os
        _os.environ["OPENCV_VIDEOIO_PRIORITY_OBSENSOR"] = "0"

        if self.source == "synthetic":
            STATE.active_source = "synthetic"
            STATE.active_source_label = self._get_source_label("synthetic")
            STATE.source_kind = "synthetic"
            STATE.camera_connected = False
            STATE.stream_ok = True
            return None

        # 1. Dosya (portable çözümleme)
        file_path = resolve_media_path(self.source)
        if file_path is not None and file_path.is_file():
            cap = cv2.VideoCapture(str(file_path))
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    log.info("Video dosyası açıldı: %s", file_path)
                    STATE.active_source = self.source
                    STATE.active_source_label = self._get_source_label(self.source)
                    STATE.source_kind = "video"
                    STATE.camera_connected = False
                    STATE.stream_ok = True
                    return cap
                cap.release()
            log.warning("Video dosyası açılamadı: %s", file_path)
            STATE.source_kind = "failed"
            STATE.stream_ok = False
            return None

        # Kaynak dosya gibi görünüyor ama bulunamadı
        if ("/" in str(self.source) or "\\" in str(self.source) or str(self.source).endswith(".mp4")) \
                and not str(self.source).isdigit() and self.source != "synthetic":
            STATE.source_kind = "failed"
            STATE.stream_ok = False
            STATE.add_log(f"❌ Video bulunamadı (portable tarama): {self.source}")
            return None

        # 2. Kamera indeksi — Windows'ta DSHOW önce; MSMF sık bozulur
        if str(self.source).isdigit():
            src_idx = int(self.source)
            backends = [cv2.CAP_DSHOW, cv2.CAP_ANY, cv2.CAP_MSMF]
            for backend in backends:
                backend_name = {
                    cv2.CAP_MSMF: "MSMF",
                    cv2.CAP_DSHOW: "DSHOW",
                    cv2.CAP_ANY: "AUTO",
                }.get(backend, str(backend))
                try:
                    cap = cv2.VideoCapture(src_idx, backend)
                    if not cap.isOpened():
                        continue
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        cap.release()
                        continue
                    # Çözünürlük isteğe bağlı; kırılırsa eski haline dön
                    w0, h0 = frame.shape[1], frame.shape[0]
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    ret2, frame2 = cap.read()
                    if not ret2 or frame2 is None:
                        cap.release()
                        cap = cv2.VideoCapture(src_idx, backend)
                        if not (cap.isOpened() and cap.read()[0]):
                            if cap:
                                cap.release()
                            continue
                        log.info(
                            "Kamera %s @ native %sx%s (1280x720 reddedildi)",
                            backend_name, w0, h0,
                        )
                    else:
                        log.info(
                            "Kamera bağlandı: backend=%s index=%s shape=%s",
                            backend_name, src_idx, frame2.shape,
                        )
                    STATE.active_source = str(src_idx)
                    STATE.active_source_label = self._get_source_label(str(src_idx))
                    STATE.source_kind = "camera"
                    STATE.camera_connected = True
                    STATE.stream_ok = True
                    return cap
                except Exception as exc:
                    log.warning("Kamera denemesi başarısız (%s): %s", backend_name, exc)

            STATE.source_kind = "failed"
            STATE.camera_connected = False
            STATE.stream_ok = False
            STATE.active_source = str(src_idx)
            STATE.active_source_label = self._get_source_label(str(src_idx))
            STATE.add_log(
                f"❌ Fiziksel kamera index={src_idx} açılamadı "
                "(meşgul, izin kapalı veya Code 45). Otomatik sahte video YOK."
            )
            return None

        STATE.source_kind = "failed"
        STATE.stream_ok = False
        STATE.add_log(f"❌ Bilinmeyen kaynak: {self.source}")
        return None

    def _no_signal_frame(self, message: str) -> np.ndarray:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.putText(frame, "KAMERA / KAYNAK YOK", (320, 320),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 220), 3)
        cv2.putText(frame, message[:80], (80, 380),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
        cv2.putText(frame, "Dogruluk gosterilmiyor — olcum yok", (280, 440),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        return frame

    def _generate_synthetic_frame(self) -> tuple[np.ndarray, List[Dict[str, Any]]]:
        """Hareketli sentetik balonlar + bilinen kutular (YOLO'ya bağımlı olmadan çizilir)."""
        self._synthetic_time += 0.06
        t = self._synthetic_time
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        for y in range(0, 720, 4):
            val = int(20 + (y / 720) * 45)
            frame[y:y+4, :] = (val + 25, val + 15, val + 5)

        for gx in range(0, 1280, 80):
            cv2.line(frame, (gx, 0), (gx, 720), (45, 55, 65), 1)
        for gy in range(0, 720, 80):
            cv2.line(frame, (0, gy), (1280, gy), (45, 55, 65), 1)

        targets: List[Dict[str, Any]] = []

        # Kırmızı düşman
        rx = int(640 + 260 * np.sin(t * 0.9) - STATE.motor_x_pos * 0.4)
        ry = int(360 + 130 * np.cos(t * 1.3) - STATE.motor_y_pos * 0.4)
        rx = max(60, min(1220, rx))
        ry = max(60, min(660, ry))
        rw, rh = 32, 42
        cv2.ellipse(frame, (rx, ry), (rw, rh), 0, 0, 360, (20, 20, 225), -1)
        cv2.circle(frame, (rx - 10, ry - 12), 7, (120, 120, 255), -1)
        cv2.line(frame, (rx, ry + rh), (rx + int(8 * np.sin(t * 4)), ry + rh + 28), (190, 190, 190), 2)
        targets.append({
            "cls": "ENEMY_RED",
            "type": "red_balloon",
            "label": "DUSMAN",
            "color": (0, 0, 220),
            "xyxy": (rx - rw - 4, ry - rh - 4, rx + rw + 4, ry + rh + 4),
            "center": (rx, ry),
            "area": (2 * rw + 8) * (2 * rh + 8),
            "conf": 0.92,
        })

        # Mavi dost
        bx = int(380 + 190 * np.cos(t * 0.6) - STATE.motor_x_pos * 0.4)
        by = int(480 + 95 * np.sin(t * 0.8) - STATE.motor_y_pos * 0.4)
        bx = max(60, min(1220, bx))
        by = max(60, min(660, by))
        bw, bh = 28, 36
        cv2.ellipse(frame, (bx, by), (bw, bh), 0, 0, 360, (225, 125, 20), -1)
        cv2.circle(frame, (bx - 8, by - 10), 6, (255, 210, 140), -1)
        cv2.line(frame, (bx, by + bh), (bx, by + bh + 24), (190, 190, 190), 2)
        targets.append({
            "cls": "FRIEND_BLUE",
            "type": "blue_balloon",
            "label": "DOST",
            "color": (220, 80, 0),
            "xyxy": (bx - bw - 4, by - bh - 4, bx + bw + 4, by + bh + 4),
            "center": (bx, by),
            "area": (2 * bw + 8) * (2 * bh + 8),
            "conf": 0.90,
        })

        cv2.putText(frame, "SENTETIK HEDEF SIMULATORU", (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 215, 255), 2)
        return frame, targets

    def _draw_tactical_hud(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        primary_target: Optional[Dict[str, Any]],
        lead_point: Optional[tuple[int, int]]
    ) -> np.ndarray:
        """Askeri standartta dinamik köşeli taktik nişangah ve hedef göstergeleri."""
        annotated = frame.copy()
        h_frame, w_frame = annotated.shape[:2]
        cx_frame, cy_frame = w_frame // 2, h_frame // 2

        # 1. Merkez Taktik Nişangah
        cv2.line(annotated, (cx_frame - 26, cy_frame), (cx_frame - 8, cy_frame), (0, 255, 120), 1)
        cv2.line(annotated, (cx_frame + 8, cy_frame), (cx_frame + 26, cy_frame), (0, 255, 120), 1)
        cv2.line(annotated, (cx_frame, cy_frame - 26), (cx_frame, cy_frame - 8), (0, 255, 120), 1)
        cv2.line(annotated, (cx_frame, cy_frame + 8), (cx_frame, cy_frame + 26), (0, 255, 120), 1)
        cv2.circle(annotated, (cx_frame, cy_frame), 4, (0, 255, 120), 1)

        # 2. Hedef Çizimleri (Köşeli Taktik Kutu)
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det["xyxy"]]
            color = det["color"]
            conf = float(det["conf"])
            is_primary = (primary_target is not None and det is primary_target)

            bw = max(10, x2 - x1)
            bh = max(10, y2 - y1)
            corner = min(16, max(6, bw // 4), max(6, bh // 4))
            th = 3 if is_primary else 2

            # Taktiksel köşe çizgileri
            cv2.line(annotated, (x1, y1), (x1 + corner, y1), color, th)
            cv2.line(annotated, (x1, y1), (x1, y1 + corner), color, th)
            cv2.line(annotated, (x2, y1), (x2 - corner, y1), color, th)
            cv2.line(annotated, (x2, y1), (x2, y1 + corner), color, th)
            cv2.line(annotated, (x1, y2), (x1 + corner, y2), color, th)
            cv2.line(annotated, (x1, y2), (x1, y2 - corner), color, th)
            cv2.line(annotated, (x2, y2), (x2 - corner, y2), color, th)
            cv2.line(annotated, (x2, y2), (x2 - corner, y2), color, th)

            # Hedef Etiketi
            badge = "[KILIT: DUSMAN]" if (is_primary and det["cls"] == "ENEMY_RED") else ("[DUSMAN]" if det["cls"] == "ENEMY_RED" else "[DOST]")
            label = f"{badge} %{int(conf * 100)}"
            (tw, th_box), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
            cv2.rectangle(annotated, (x1, max(0, y1 - th_box - 10)), (x1 + tw + 8, y1), (15, 20, 25), -1)
            cv2.rectangle(annotated, (x1, max(0, y1 - th_box - 10)), (x1 + tw + 8, y1), color, 1)
            cv2.putText(annotated, label, (x1 + 4, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2)

            # Merkezden birincil hedefe kilit çizgisi
            if is_primary:
                tx, ty = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.line(annotated, (cx_frame, cy_frame), (tx, ty), color, 1, cv2.LINE_AA)

        # 3. Kalman Hız Vektörü ve Öncü Vuruş Nişangahı (Lead Reticle)
        if self.kalman and self.kalman.is_active():
            vx, vy = self.kalman.velocity
            if primary_target is not None:
                x1, y1, x2, y2 = [int(v) for v in primary_target["xyxy"]]
                ptx, pty = (x1 + x2) // 2, (y1 + y2) // 2
                arrow_end = (int(ptx + vx * 0.8), int(pty + vy * 0.8))
                if (arrow_end[0] - ptx)**2 + (arrow_end[1] - pty)**2 > 9:
                    cv2.arrowedLine(annotated, (ptx, pty), arrow_end, (0, 255, 255), 2, tipLength=0.3)

            if lead_point is not None:
                lx, ly = lead_point
                lx = max(20, min(w_frame - 20, lx))
                ly = max(20, min(h_frame - 20, ly))
                cv2.circle(annotated, (lx, ly), 11, (0, 220, 255), 2)
                cv2.circle(annotated, (lx, ly), 3, (0, 220, 255), -1)
                cv2.line(annotated, (lx - 16, ly), (lx + 16, ly), (0, 220, 255), 1)
                cv2.line(annotated, (lx, ly - 16), (lx, ly + 16), (0, 220, 255), 1)
                cv2.putText(annotated, "ONCU VURUS", (lx + 15, ly - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1)

        # 4. Askeri Başlık
        cv2.putText(annotated, "CELIK KUBBE // HAVA SAVUNMA SISTEMI", (20, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        return annotated

    def _loop(self) -> None:
        STATE.available_sources = discover_available_sources()
        self.cap = self._open_capture()
        fail_streak = 0

        while self._running:
            with self._lock:
                if STATE.emergency:
                    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                    overlay = frame.copy()
                    overlay[:] = (0, 0, 120)
                    frame = cv2.addWeighted(frame, 0.4, overlay, 0.6, 0)
                    cv2.putText(frame, "!!! ACIL DURDURMA AKTIF !!!", (230, 360),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 255), 4)
                    STATE.frame_b64 = _encode_frame(frame)
                    STATE.stream_ok = False
                    STATE.clear_detections()
                    time.sleep(0.08)
                    continue

                frame = None
                synth_targets: List[Dict[str, Any]] = []
                if self.source == "synthetic":
                    frame, synth_targets = self._generate_synthetic_frame()
                    STATE.source_kind = "synthetic"
                    STATE.camera_connected = False
                    STATE.stream_ok = True
                elif self.cap is not None and self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if not ret:
                        if STATE.source_kind == "video":
                            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            ret, frame = self.cap.read()
                        if not ret:
                            fail_streak += 1
                            if fail_streak >= 15 and STATE.source_kind == "camera":
                                STATE.camera_connected = False
                                STATE.stream_ok = False
                                STATE.source_kind = "failed"
                                try:
                                    self.cap.release()
                                except Exception:
                                    pass
                                self.cap = None
                                STATE.clear_detections(
                                    "❌ Kamera akışı koptu — hedef/doğruluk temizlendi"
                                )
                            time.sleep(0.05)
                            continue
                    fail_streak = 0
                    STATE.stream_ok = True
                    if str(self.source).isdigit():
                        STATE.camera_connected = True
                        STATE.source_kind = "camera"
                    else:
                        STATE.camera_connected = False
                        STATE.source_kind = "video"
                else:
                    STATE.stream_ok = False
                    STATE.camera_connected = False
                    STATE.source_kind = "failed"
                    frame = self._no_signal_frame(f"Kaynak: {self.source}")
                    STATE.clear_detections()
                    STATE.fps = 0.0
                    _draw_hud(frame, 0.0)
                    STATE.frame_b64 = _encode_frame(frame)
                    time.sleep(0.2)
                    continue

            if frame is None:
                time.sleep(0.05)
                continue

            if frame.shape[1] != 1280 or frame.shape[0] != 720:
                frame = cv2.resize(frame, (1280, 720))

            fps = self._fps_meter.update()
            STATE.fps = fps

            try:
                if self.source == "synthetic":
                    frame = self._run_defense_pipeline(frame, synth_targets)
                    STATE.stream_ok = True
                    STATE.source_kind = "synthetic"
                    STATE.camera_connected = False
                elif STATE.stream_ok:
                    dets = self._yolo_detections(frame)
                    frame = self._run_defense_pipeline(frame, dets)
                else:
                    _draw_simulation_overlay(frame)
            except Exception as exc:
                log.warning(f"Inference/çizim hatası: {exc}")
                STATE.clear_detections()

            _draw_hud(frame, fps)
            STATE.frame_b64 = _encode_frame(frame)
            time.sleep(0.012)

    def _run_defense_pipeline(self, frame: np.ndarray, dets: List[Dict[str, Any]]) -> np.ndarray:
        """
        Çelikkubbe Savunma Algoritmaları Entegrasyonu:
        1. Debounce (Zamansal Karar Filtresi)
        2. ThreatPrioritizer (En Tehlikeli Hedefe Kilitlenme)
        3. BalloonKalmanFilter (Konum & Hız Takibi, Öncü Vuruş Noktası)
        4. SoundAlertManager (Düşman Balon Sesli Uyarısı)
        5. Metrik Mesafe ve Taret Açısal Hata (dx, dy)
        6. Taktik HUD Çizimi
        """
        h_frame, w_frame = frame.shape[:2]
        cx_frame, cy_frame = w_frame // 2, h_frame // 2

        # 1. Debounce güncelle
        has_red = any(d["cls"] == "ENEMY_RED" for d in dets)
        has_blue = any(d["cls"] == "FRIEND_BLUE" for d in dets)
        if self.debounce:
            self.debounce.update(has_red, has_blue)
            red_alarm = self.debounce.is_red_alarm_active()
        else:
            red_alarm = has_red

        # 2. Tehdit Önceliklendirme (Threat Prioritizer)
        primary_target = None
        if ThreatPrioritizer and dets:
            primary_target = ThreatPrioritizer.select_primary_target(dets, (cx_frame, cy_frame))
        elif dets:
            sorted_dets = sorted(dets, key=lambda d: (1 if d["cls"] == "ENEMY_RED" else 0, d.get("area", 0)), reverse=True)
            primary_target = sorted_dets[0]

        # 3. Kalman Filtresi ve Öncü Vuruş (Lead Pursuit)
        lead_point = None
        if self.kalman:
            if primary_target is not None:
                self.kalman.update(primary_target["center"])
                lead_point = self.kalman.get_lead_point(lead_seconds=getattr(config, "KALMAN_LEAD_TIME", 0.4) if config else 0.4)
            elif self.kalman.is_active():
                self.kalman.predict()
                lead_point = self.kalman.get_lead_point(lead_seconds=getattr(config, "KALMAN_LEAD_TIME", 0.4) if config else 0.4)

        # 4. Sesli Uyarı (Düşman Balon Doğrulandığında)
        if red_alarm and primary_target and primary_target["cls"] == "ENEMY_RED":
            self.sound.trigger(freq=1600, duration=150)

        # 5. Hata ve Mesafe Hesabı
        if primary_target is not None:
            aim_x, aim_y = lead_point if (lead_point is not None and self.kalman and self.kalman.is_active()) else primary_target["center"]
            dx = (aim_x - cx_frame) / w_frame
            dy = (aim_y - cy_frame) / h_frame

            x1, y1, x2, y2 = primary_target["xyxy"]
            box_h = max(10, y2 - y1)
            # Standart 25 cm balon için yaklaşık metrik mesafe
            distance_m = round(200.0 / box_h, 1)
            primary_cls = primary_target["cls"]
            primary_conf = float(primary_target["conf"])
            fire_enable = bool(primary_cls == "ENEMY_RED" and red_alarm)
        else:
            dx, dy = 0.0, 0.0
            distance_m = None
            primary_cls = "BELIRSIZ"
            primary_conf = None
            fire_enable = False

        # 6. Global Telemetri
        STATE.target_class = primary_cls
        STATE.target_label, STATE.target_label_color = class_to_turkish(primary_cls)
        STATE.confidence = primary_conf
        STATE.distance_m = distance_m
        STATE.turret_error_dx = dx
        STATE.turret_error_dy = dy
        STATE.fire_enable = fire_enable
        STATE.tracks_count = len(dets)
        STATE.vision_ok = True
        STATE.track_id = 1 if primary_target else -1

        # 7. Taktik Çizim
        annotated = self._draw_tactical_hud(frame, dets, primary_target, lead_point)
        return annotated

    def _yolo_detections(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """CLAHE + YOLO V8 / V11 inference + HSV Renk Filtresi Fallback."""
        dets: List[Dict[str, Any]] = []

        # CLAHE ile güneş parlamasını dengele
        if getattr(config, "USE_CLAHE", True):
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l2 = self.clahe.apply(l)
            frame_proc = cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2BGR)
        else:
            frame_proc = frame

        if self.model is not None:
            results = self.model(frame_proc, conf=self.conf, verbose=False)
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue
                for box in boxes:
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    cls_name = ""
                    if hasattr(self.model, "names") and isinstance(self.model.names, dict):
                        cls_name = str(self.model.names.get(cls_id, "")).lower()

                    if "not" in cls_name:
                        continue  # not_balloon -> atla
                    elif "red" in cls_name or "kirmizi" in cls_name or cls_id == 0:
                        dets.append({
                            "cls": "ENEMY_RED",
                            "type": "red_balloon",
                            "label": "DUSMAN",
                            "color": (0, 0, 220),
                            "xyxy": (x1, y1, x2, y2),
                            "center": ((x1 + x2) // 2, (y1 + y2) // 2),
                            "area": (x2 - x1) * (y2 - y1),
                            "conf": conf,
                        })
                    elif "blue" in cls_name or "mavi" in cls_name or cls_id == 1:
                        dets.append({
                            "cls": "FRIEND_BLUE",
                            "type": "blue_balloon",
                            "label": "DOST",
                            "color": (220, 80, 0),
                            "xyxy": (x1, y1, x2, y2),
                            "center": ((x1 + x2) // 2, (y1 + y2) // 2),
                            "area": (x2 - x1) * (y2 - y1),
                            "conf": conf,
                        })

        # YOLO tespit edemezse veya model henüz yüklenmediyse: HSV + Geometrik Dairesellik Fallback
        if not dets:
            hsv = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2HSV)
            hsv = cv2.GaussianBlur(hsv, (5, 5), 0)
            r_mask1 = cv2.inRange(hsv, np.array([0, 110, 70]), np.array([10, 255, 255]))
            r_mask2 = cv2.inRange(hsv, np.array([170, 110, 70]), np.array([180, 255, 255]))
            r_mask = cv2.bitwise_or(r_mask1, r_mask2)
            b_mask = cv2.inRange(hsv, np.array([95, 100, 60]), np.array([135, 255, 255]))

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            for mask, c_type, c_cls, c_color in [
                (r_mask, "red_balloon", "ENEMY_RED", (0, 0, 220)),
                (b_mask, "blue_balloon", "FRIEND_BLUE", (220, 80, 0))
            ]:
                cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
                cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)
                contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if 900 <= area <= 250000:
                        x, y, w, h = cv2.boundingRect(cnt)
                        perim = cv2.arcLength(cnt, True)
                        circ = (4.0 * np.pi * area) / max(1.0, perim * perim)
                        if circ >= 0.50:
                            dets.append({
                                "cls": c_cls,
                                "type": c_type,
                                "label": "DUSMAN" if c_cls == "ENEMY_RED" else "DOST",
                                "color": c_color,
                                "xyxy": (x, y, x + w, y + h),
                                "center": (x + w // 2, y + h // 2),
                                "area": area,
                                "conf": min(0.95, circ * 0.90 + 0.05),
                            })
        return dets


    def _annotate_synthetic_only(self, frame: np.ndarray) -> np.ndarray:
        """Geriye uyumluluk — sentetikte geometrik kutular tercih edilir."""
        return frame


class _FPSMeter:
    def __init__(self):
        self._last = time.perf_counter()
        self.value = 0.0

    def update(self) -> float:
        now = time.perf_counter()
        dt = now - self._last
        self._last = now
        self.value = 1.0 / dt if dt > 0 else 0.0
        return self.value


def _draw_crosshair(frame: np.ndarray, cx: int, cy: int, color: tuple, size: int = 32) -> None:
    """Taktik nişangah çizer."""
    cv2.line(frame, (cx - size, cy), (cx - 6, cy), color, 2)
    cv2.line(frame, (cx + 6, cy), (cx + size, cy), color, 2)
    cv2.line(frame, (cx, cy - size), (cx, cy - 6), color, 2)
    cv2.line(frame, (cx, cy + 6), (cx, cy + size), color, 2)
    cv2.circle(frame, (cx, cy), 4, color, 1)
    cv2.rectangle(frame, (cx - size, cy - size), (cx + size, cy + size), color, 1)


def _draw_simulation_overlay(frame: np.ndarray) -> None:
    """YOLO yokken bilgilendirici overlay — sahte doğruluk YOK."""
    h, w = frame.shape[:2]
    cv2.putText(frame, "SIMULASYON — YOLO modeli bagli degil",
                (w // 2 - 300, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
    _draw_crosshair(frame, w // 2, h // 2, (80, 180, 80))
    STATE.clear_detections()


def _draw_synthetic_banner(frame: np.ndarray) -> None:
    """Sentetik kaynak: ekranda uyarı; telemetri doğruluğu yok."""
    cv2.putText(frame, "SENTETIK — KAMERA KAPALI — DOGRULUK TELEMETRIYE YAZILMAZ",
                (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 215, 255), 2)


def _draw_hud(frame: np.ndarray, fps: float) -> None:
    """FPS ve sistem durumu HUD paneli."""
    h, w = frame.shape[:2]
    cam = "CAM:ON" if STATE.camera_connected else "CAM:OFF"
    kind = STATE.source_kind.upper()
    status = (
        f"FPS: {fps:.1f}  |  {cam}  |  SRC:{kind}  |  "
        f"ROS2: {STATE.ros2_status.upper()}"
    )
    cv2.putText(frame, status, (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 255, 180), 1)


def _encode_frame(frame: np.ndarray, quality: int = 75) -> str:
    """Frame'i base64 JPEG string'e çevirir."""
    # 1280x720 çözünürlüğe küçült (bandwidth kontrolü)
    if frame.shape[1] > 1280:
        frame = cv2.resize(frame, (1280, 720))
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode("utf-8")


# =============================================================================
# COMMAND HANDLER
# =============================================================================

async def handle_command(data: Dict[str, Any]) -> None:
    """Frontend'den gelen komutları işler."""
    cmd = data.get("cmd", "")

    if STATE.emergency and cmd not in ("emergency_reset", "ping"):
        STATE.add_log(f"⛔ Acil durdurma aktif — komut reddedildi: {cmd}")
        return

    if cmd == "emergency_stop":
        ROS2_BRIDGE.send_emergency_stop()

    elif cmd == "emergency_reset":
        ROS2_BRIDGE.send_emergency_reset()

    elif cmd == "move":
        axis = data.get("axis", "x")
        value = int(data.get("value", 0))
        if axis == "x":
            new_pos = STATE.motor_x_pos + value * STATE.pan_speed
            ROS2_BRIDGE.send_motor_x(new_pos)
        else:
            new_pos = STATE.motor_y_pos + value * STATE.tilt_speed
            ROS2_BRIDGE.send_motor_y(new_pos)

    elif cmd == "center":
        ROS2_BRIDGE.center_turret()

    elif cmd == "fire":
        if STATE.fire_enable and metrics_are_trusted():
            ROS2_BRIDGE.send_servo(45)
            await asyncio.sleep(STATE.fire_delay_ms / 1000)
            ROS2_BRIDGE.send_servo(90)
            ROS2_BRIDGE.send_laser(100)
            await asyncio.sleep(0.5)
            ROS2_BRIDGE.send_laser(0)
            STATE.add_log("🔥 ATEŞ komutu gönderildi!")
        else:
            STATE.add_log("⛔ ATEŞ reddedildi — güvenilir kamera/video + fire_enable gerekli")

    elif cmd == "force_fire":
        # Manuel modda zorla ateş (IFF bypass)
        if STATE.mode == "MANUEL":
            ROS2_BRIDGE.send_servo(45)
            await asyncio.sleep(STATE.fire_delay_ms / 1000)
            ROS2_BRIDGE.send_servo(90)
            STATE.add_log("🔥 MANUEL ATEŞ gönderildi (IFF bypass)")

    elif cmd == "pid_update":
        axis = data.get("axis", "horizontal")
        params = {
            "kp": float(data.get("kp", 0.03)),
            "ki": float(data.get("ki", 0.01)),
            "kd": float(data.get("kd", 0.30)),
            "dz": float(data.get("dz", 0)),
        }
        if axis == "horizontal":
            STATE.pid_horizontal = params
        else:
            STATE.pid_vertical = params
        ROS2_BRIDGE.send_pid(axis, params)

    elif cmd == "set_speed":
        STATE.pan_speed = int(data.get("pan_speed", STATE.pan_speed))
        STATE.tilt_speed = int(data.get("tilt_speed", STATE.tilt_speed))
        STATE.fire_delay_ms = int(data.get("fire_delay_ms", STATE.fire_delay_ms))
        STATE.add_log(f"Hız ayarlandı: Pan={STATE.pan_speed} Tilt={STATE.tilt_speed} Gecikme={STATE.fire_delay_ms}ms")

    elif cmd == "set_mode":
        STATE.mode = data.get("mode", "OTONOM")
        STATE.add_log(f"Mod değiştirildi: {STATE.mode}")

    elif cmd == "set_step":
        STATE.manual_step = int(data.get("step", STATE.manual_step))

    elif cmd == "switch_source":
        new_source = data.get("source", "0")
        if VISION_PIPELINE is not None:
            VISION_PIPELINE.switch_source(new_source)

    elif cmd == "ping":
        pass  # Sadece bağlantı canlılık kontrolü


# =============================================================================
# WEBSOCKET SERVER
# =============================================================================

async def ws_handler(websocket) -> None:
    CLIENTS.add(websocket)
    remote = getattr(websocket, "remote_address", ("127.0.0.1", 0))
    STATE.add_log(f"Yeni bağlantı: {remote[0]}:{remote[1]}")
    try:
        async for raw_msg in websocket:
            try:
                data = json.loads(raw_msg)
                await handle_command(data)
            except json.JSONDecodeError:
                log.warning(f"Geçersiz JSON: {raw_msg[:100]}")
            except Exception as exc:
                log.warning(f"Komut işleme hatası: {exc}")
    except websockets.exceptions.ConnectionClosed:
        pass
    except Exception as exc:
        log.debug(f"WebSocket istemci ayrıldı: {exc}")
    finally:
        CLIENTS.discard(websocket)
        STATE.add_log(f"Bağlantı kapandı: {remote[0]}:{remote[1]}")


async def broadcast_loop(interval: float = 0.05) -> None:
    """~20 Hz telemetri yayını (video frame dahil)."""
    while True:
        try:
            if CLIENTS:
                payload = json.dumps(STATE.to_telemetry_dict())
                dead = set()
                for ws in list(CLIENTS):
                    try:
                        await ws.send(payload)
                    except Exception:
                        dead.add(ws)
                if dead:
                    CLIENTS.difference_update(dead)
        except Exception as exc:
            log.warning(f"Telemetri yayın döngüsü: {exc}")
        await asyncio.sleep(interval)


# =============================================================================
# HTTP STATIC SERVER (dashboard.html'i sunar)
# =============================================================================

def start_http_server(host: str, port: int) -> None:
    class QuietHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            pass  # HTTP loglarını sessizleştir

        def translate_path(self, path):
            # dashboard.html veya statik dosyalar bu dosyanın olduğu klasörden
            root = Path(__file__).parent
            rel = path.lstrip("/")
            target = root / rel
            return str(target)

    http.server.ThreadingHTTPServer.allow_reuse_address = True
    server = http.server.ThreadingHTTPServer(("0.0.0.0", port), QuietHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    log.info(f"🌐 HTTP sunucu: http://{host}:{port}/dashboard.html")


# =============================================================================
# KEYBOARD HOTKEYS (Windows / Linux)
# =============================================================================

def register_keyboard_hooks() -> None:
    if not KEYBOARD_AVAILABLE:
        return

    def _emergency():
        ROS2_BRIDGE.send_emergency_stop()
        log.info("⛔ KLAVYE ACİL DURDUR: F9 basıldı")

    def _reset():
        if STATE.emergency:
            ROS2_BRIDGE.send_emergency_reset()
            log.info("✅ KLAVYE RESET: F10 basıldı")

    keyboard.add_hotkey("f9", _emergency)
    keyboard.add_hotkey("f10", _reset)
    log.info("✅ Klavye kısayolları: F9=ACİL DURDUR | F10=RESET")


# =============================================================================
# ENTRY POINT
# =============================================================================

async def main_async(args: argparse.Namespace) -> None:
    global VISION_PIPELINE
    # Başlat
    ROS2_BRIDGE.start()
    register_keyboard_hooks()

    VISION_PIPELINE = VisionPipeline(
        source=args.source,
        model_path=args.model,
        conf=args.conf,
    )
    VISION_PIPELINE.start()
    STATE.add_log(f"🛡️  Çelikkubbe Dashboard başlatıldı | Kaynak: {args.source} | WS: ws://localhost:{args.ws_port}")

    # HTTP sunucu (ayrı thread)
    start_http_server(args.host, args.http_port)

    # WebSocket sunucu — resilient keepalive
    async with websockets.serve(
        ws_handler,
        "0.0.0.0",
        args.ws_port,
        ping_interval=20,
        ping_timeout=20,
        max_size=10 * 1024 * 1024,
    ):
        await broadcast_loop(interval=1.0 / args.broadcast_hz)


def main() -> None:
    acquire_instance_lock()
    parser = argparse.ArgumentParser(description="Çelikkubbe Dashboard Backend")
    parser.add_argument(
        "--source",
        default="synthetic",
        help="Kaynak: synthetic (varsayılan, portable), 0/1 webcam, veya test_media/*.mp4",
    )
    parser.add_argument("--model", default=None, help="YOLO .pt model yolu")
    parser.add_argument("--conf", type=float, default=0.40, help="YOLO güven eşiği")
    parser.add_argument("--ws-port", type=int, default=8765, help="WebSocket port")
    parser.add_argument("--http-port", type=int, default=8081, help="HTTP port (dashboard.html)")
    parser.add_argument("--host", default="localhost", help="Sunucu adresi")
    parser.add_argument("--broadcast-hz", type=float, default=20.0, help="WebSocket yayın frekansı")
    args = parser.parse_args()

    if args.model is None:
        found = find_default_model()
        if found:
            args.model = found
            log.info(f"Model otomatik bulundu: {args.model}")
        else:
            log.warning("Model bulunamadı — kutular/YOLO olmadan devam (simülasyon overlay)")

    STATE.available_sources = discover_available_sources()
    log.info("=" * 60)
    log.info("  TEKNOFEST 2027 — Çelikkubbe Dashboard Backend")
    log.info("=" * 60)
    log.info(f"  Kamera : {args.source}")
    log.info(f"  Model  : {args.model or 'YOK (simülasyon)'}")
    log.info(f"  ROS2   : {'✅ Mevcut' if ROS2_AVAILABLE else '⚠️  Fallback modu'}")
    log.info(f"  HTTP   : http://{args.host}:{args.http_port}/dashboard.html")
    log.info(f"  WS     : ws://{args.host}:{args.ws_port}")
    log.info("=" * 60)

    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        log.info("Kapatılıyor...")


if __name__ == "__main__":
    main()
