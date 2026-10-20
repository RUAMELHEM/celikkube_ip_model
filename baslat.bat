@echo off
chcp 65001 > nul
title CELIKKUBBE HAVA SAVUNMA & TARET KONTROL
echo ======================================================================
echo   🛡️ CELIKKUBBE HAVA SAVUNMA & TARET KONTROL MERKEZI BASLATILIYOR...
echo ======================================================================
echo.
python main.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [!] Bir hata olustu. Bagimliliklari kontrol edin: pip install -r requirements.txt
    pause
)
