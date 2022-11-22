from detect_and_pick_people import cropImage
from dummy_data import dummy
import os
import cv2

if not os.path.exists("cropped_img"):
    os.mkdir("cropped_img")

cropped_img = cropImage("crowdhuman_yolov5m.pt", dummy)

for img in cropped_img:
    if not os.path.exists(f"cropped_img/frame_{img['frame']}"):
        os.mkdir(f"cropped_img/frame_{img['frame']}")

    for idx, img_result in enumerate(img["image"]):
        cv2.imwrite(f"./cropped_img/frame_{img['frame']}/people{idx}.jpg", img_result)