from detect_and_pick_people import cropImage
from dummy_data import dummy
import os
import cv2

os.mkdir("cropped_img")

cropped_img = cropImage("crowdhuman_yolov5m.pt", dummy)

for img in cropped_img:
    for img_result in img["image"]:
        cv2.imwrite(f"cropped_img/frame_{img['frame']}", img_result)