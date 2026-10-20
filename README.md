# 🛡️ ÇELİKKUBBE — DONANIM & SAHA TEST DAĞITIM PAKETİ
**TEKNOFEST Hava Savunma & Taret Kontrol Sistemi**

Bu paket, **donanım ve mekanik ekibinin** taret motorlarını, kameraları, PID açı sapmalarını ve lazer tetikleyicilerini doğrudan sahada ve atölyede test edebilmesi için hazırlanmış **temiz çalışma (runtime) sürümüdür**. Eğitim kodları, ham veri setleri ve gereksiz dosyalar çıkarılmıştır.

---

## 📦 Paket İçeriği

```text
celikkubbe_donanim_paketi/
├── baslat.bat                 ← Çift tıklayarak sistemi başlatma aracı
├── baslat.ps1                 ← PowerShell tek tıkla başlatma
├── main.py                    ← Ana başlatıcı (HTTP + WS + Otomatik Tarayıcı)
├── best.pt                    ← Eğitilmiş nihai YOLO balistik tespit modeli
├── config.py                  ← Kamera, PID katsayıları ve açı sınır ayarları
├── tracker.py                 ← Kalman filtresi, öncü vuruş ve tehdit motoru
├── requirements.txt           ← Gerekli Python kütüphaneleri
└── arayuz/
    ├── dashboard.html         ← 60 FPS Askeri HUD web arayüzü
    ├── dashboard_backend.py   ← WebSocket & kamera akış sunucusu
    ├── cam_test.py            ← USB/Kamera bağlantı test aracı
    └── check_cam.py           ← Windows kamera indeks kontrolcüsü
```

---

## ⚡ Hızlı Kurulum & Başlatma

### 1. Bağımlılıkları Yükleyin (İlk Seferde)
Terminal veya komut istemcisinde:
```bash
pip install -r requirements.txt
```

### 2. Sistemi Başlatın
**En Kolay Yol:** `baslat.bat` dosyasına çift tıklayın.

Veya terminalden:
```bash
python main.py
```
> **Not:** Sistem başladığında varsayılan tarayıcınızda (Chrome / Edge) **`http://localhost:8081/dashboard.html`** otomatik olarak açılır.

---

## 🎮 Taret Kontrolleri & Kısayollar

- **F9 / Kırmızı Buton:** `ACİL DURDURMA (E-STOP)` — Tüm motor hareketini ve lazeri anında kilitler.
- **F10 / Mavi Buton:** `RESET (Sıfırla)` — Tareti başlangıç koordinatlarına döndürür.
- **W, A, S, D / Yön Tuşları:** Manuel modda taret Pan/Tilt eksenlerini hareket ettirir.
- **Ateş Butonu:** Hedefe kilitlenildiğinde lazer tetikleme sinyali üretir.

---

## 📷 Kamera & Kaynak Seçimi

Arayüzün üst barındaki açılır menüden kaynak anında değiştirilebilir:
- **Webcam 0:** Bilgisayarın dahili kamerası.
- **Webcam 1:** USB üzerinden bağlı harici taret kamerası.
- **Sentetik Simülasyon:** Kamera takılı değilken motor ve PID algoritmalarını simüle eden sentetik balon hedef motoru.

*Kamera indeksini doğrulamak için `python arayuz/cam_test.py` çalıştırabilirsiniz.*

---

## ⚙️ Donanım & PID İnce Ayarları

Taret mekanik dişli oranları, motor hızları ve PID katsayıları hem web arayüzündeki kutulardan anlık olarak değiştirilebilir hem de kalıcı olarak `config.py` içerisinden yapılandırılabilir.
