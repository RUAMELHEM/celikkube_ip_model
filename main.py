
import os
import sys
import time
import socket
import threading
import webbrowser
import argparse
import asyncio
from pathlib import Path

# Kök ve Arayüz dizinlerini Python yoluna ekle
ROOT_DIR = Path(__file__).resolve().parent
ARAYUZ_DIR = ROOT_DIR / "arayuz"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(ARAYUZ_DIR) not in sys.path:
    sys.path.insert(0, str(ARAYUZ_DIR))


def clean_stale_lock():
    """Eski oturumlardan kalan kilit dosyasını temizler."""
    lock_file = ARAYUZ_DIR / ".dashboard_backend.lock"
    if lock_file.exists():
        try:
            lock_file.unlink()
        except Exception:
            pass


def is_port_in_use(port: int) -> bool:
    """Portun kullanımda olup olmadığını test eder."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0


def auto_open_browser(url: str, delay: float = 1.2):
    """Sunucu ayağa kalktıktan sonra tarayıcıyı otomatik açar."""
    def _open():
        time.sleep(delay)
        print(f"\n[🚀] Taktik Arayüz tarayıcıda açılıyor: {url}")
        webbrowser.open_new(url)
    threading.Thread(target=_open, daemon=True).start()


def print_banner(host: str, http_port: int, ws_port: int, model_path: str, source: str):
    """Taktik konsol bilgi paneli."""
    url = f"http://{host}:{http_port}/dashboard.html"
    print("\n" + "═" * 70)
    print("  🛡️  ÇELİKKUBBE HAVA SAVUNMA & TARET KONTROL MERKEZİ (V5 DASHBOARD)")
    print("═" * 70)
    print(f"  ► ARAYÜZ ADRESİ     : {url}")
    print(f"  ► WEBSOCKET PORTU   : ws://{host}:{ws_port}")
    print(f"  ► AKTİF MODEL       : {model_path or 'Simülasyon Modu (best.pt)'}")
    print(f"  ► VİDEO KAYNAĞI     : {source}")
    print("─" * 70)
    print("  ► KISAYOLLAR        : F9: Acil Durdurma (E-STOP) | F10: Reset / Sıfırla")
    print("  ► HAREKET KONTROLÜ  : W/A/S/D veya Yön Tuşları ile Taret Yönlendirme")
    print("  ► ÇIKIŞ İÇİN        : Konsolda Ctrl + C tuşlarına basın")
    print("═" * 70 + "\n")


def main():
    clean_stale_lock()

    parser = argparse.ArgumentParser(description="Çelikkubbe Savunma Taktik Dashboard")
    parser.add_argument("--source", default=None, help="Video kaynağı: 0 (webcam), 1, synthetic, veya test_media/*.mp4")
    parser.add_argument("--model", default=None, help="YOLO model dosyası (varsayılan: best.pt)")
    parser.add_argument("--conf", type=float, default=0.28, help="YOLO tespit güven eşiği")
    parser.add_argument("--http-port", type=int, default=8081, help="HTTP web portu")
    parser.add_argument("--ws-port", type=int, default=8765, help="WebSocket akış portu")
    parser.add_argument("--host", default="localhost", help="Bağlantı host adresi")
    parser.add_argument("--no-browser", action="store_true", help="Tarayıcıyı otomatik açma")
    args = parser.parse_args()

    # Port çakışması kontrolü
    if is_port_in_use(args.http_port):
        print(f"[!] Uyarı: Port {args.http_port} kullanımda görünüyor. Bağlantı bekleniyor...")

    # dashboard_backend modülünü içeri al
    import dashboard_backend as backend

    # Model yolu tespiti
    if args.model is None:
        args.model = backend.find_default_model()

    # Kaynak tespiti: Belirtilmemişse webcam 0'ı dene, yoksa sentetik simülasyon
    if args.source is None:
        try:
            import cv2
            test_cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if test_cap.isOpened():
                ret, _ = test_cap.read()
                test_cap.release()
                args.source = "0" if ret else "synthetic"
            else:
                args.source = "synthetic"
        except Exception:
            args.source = "synthetic"

    dashboard_url = f"http://{args.host}:{args.http_port}/dashboard.html"
    print_banner(args.host, args.http_port, args.ws_port, args.model, args.source)

    # Tarayıcıyı otomatik aç
    if not args.no_browser:
        auto_open_browser(dashboard_url, delay=1.2)

    # Backend argümanlarını aktar ve başlat
    backend_args = argparse.Namespace(
        source=args.source,
        model=args.model,
        conf=args.conf,
        ws_port=args.ws_port,
        http_port=args.http_port,
        host=args.host,
        broadcast_hz=30.0,
    )

    try:
        asyncio.run(backend.main_async(backend_args))
    except KeyboardInterrupt:
        print("\n[+] Çelikkubbe Taktik Dashboard güvenli şekilde kapatıldı.")
    except Exception as e:
        print(f"\n[!] Hata oluştu: {e}")
    finally:
        clean_stale_lock()


if __name__ == "__main__":
    main()
