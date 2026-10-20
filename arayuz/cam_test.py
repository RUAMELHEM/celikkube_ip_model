import os
os.environ['OPENCV_VIDEOIO_PRIORITY_OBSENSOR'] = '0'
import cv2

print("OpenCV:", cv2.__version__)
for idx in range(4):
    for backend_name, backend_id in [("AUTO", cv2.CAP_ANY), ("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF)]:
        cap = cv2.VideoCapture(idx, backend_id)
        opened = cap.isOpened()
        cap.release()
        if opened:
            print("ACIK: backend=%s index=%d" % (backend_name, idx))
        else:
            print("kapali: backend=%s index=%d" % (backend_name, idx))
