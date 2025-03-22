# from pathlib import Path

import cv2
# import numpy as np
# import IPython
# from skimage import io
from YuNetFace import FaceDetectorYunet

# def show_image(image):
#     _, ret = cv2.imencode('.jpg', image)
#     i = IPython.display.Image(data=ret)
#     IPython.display.display(i)

# def show_by_name(name):
#     fp = Path('../test_images').joinpath(name)
#     img = cv2.imread(str(fp))
#     show_image(img)

def start_webcam():
    rtsp_username = "admin"
    # rtsp_password = "123456789"
    rtsp_password = "cctv@123"
    width = 800
    height = 480
    cam_no = "1"
    rtsp = "rtsp://" + rtsp_username + ":" + rtsp_password + "@192.168.1.64:554/Streaming/channels/" + cam_no + "01"
    cap = cv2.VideoCapture(rtsp, cv2.CAP_FFMPEG)
    # cap.open(rtsp)
    cap.set(3, width)  # Set width
    cap.set(4, height)  # Set height
    # success, current_cam = cap.read()
    return cap

fd = FaceDetectorYunet()

# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# faces = fd.detect(img)


cap = start_webcam()
# cap = cv2.VideoCapture(0)

while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        faces = fd.detect(frame)
        if faces:
            fd.draw_faces(frame, faces)
            # print(type(faces))
        # show_image(frame)

        cv2.imshow('YuNet Face Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()