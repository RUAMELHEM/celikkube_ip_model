import subprocess
import json
import os
import sys

def check_pnp():
    ps_cmd = 'Get-PnpDevice | Where-Object { $_.Class -eq "Camera" -or $_.Class -eq "Image" -or $_.FriendlyName -like "*Camera*" -or $_.FriendlyName -like "*Webcam*" } | Select-Object FriendlyName, InstanceId, Class, Status, Problem, ProblemDescription | ConvertTo-Json'
    res = subprocess.run(['powershell', '-NoProfile', '-Command', ps_cmd], capture_output=True, text=True)
    print("=== PNP DEVICES ===")
    print(res.stdout)
    if res.stderr:
        print("STDERR:", res.stderr)

def check_media_foundation():
    # Use pygrabber or DirectShow if available, or test cv2
    import cv2
    print("=== CV2 BUILD INFO VIDEOIO ===")
    for line in cv2.getBuildInformation().split('\n'):
        if any(k in line for k in ['VIDEOIO', 'Media Foundation', 'DirectShow', 'MSMF', 'DSHOW']):
            print(line)
            
    print("\n=== TESTING INDICES WITH cv2 ===")
    os.environ['OPENCV_VIDEOIO_PRIORITY_OBSENSOR'] = '0'
    for idx in range(5):
        for name, api in [('AUTO', cv2.CAP_ANY), ('DSHOW', cv2.CAP_DSHOW), ('MSMF', cv2.CAP_MSMF)]:
            cap = cv2.VideoCapture(idx, api)
            opened = cap.isOpened()
            print(f"Index {idx} with {name}: isOpened = {opened}")
            if opened:
                ret, frame = cap.read()
                print(f"  -> read(): ret={ret}, shape={frame.shape if ret else None}")
                cap.release()

if __name__ == "__main__":
    check_pnp()
    check_media_foundation()
