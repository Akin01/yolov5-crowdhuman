import torch
from utils.torch_utils import select_device, time_synchronized
import base64
import numpy as np
from io import BytesIO
from PIL import Image
from dummy_data import dummy
import time
from utils.plots import crop_img_by_bbox
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords, set_logging
from models.experimental import attempt_load


def cropImage(weights, img_obj, imgsz=640, device='cpu', augment=True, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic_nms=False):
    # Initialize
    set_logging()
    device = select_device(device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if half:
        model.half()  # to FP16

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    t0 = time.time()

    img_arr_result = []

    for idx, key in enumerate(img_obj["gallery_images"]):
        print(f"frame {idx}")
        parse_uri = (key["image"]).split(",")[1]
        decoded_img = base64.b64decode(parse_uri)

        opened_img = Image.open(BytesIO(decoded_img))
        img_nump_arr = np.asarray(opened_img, dtype='uint8')

        # Padded resize
        img = letterbox(img_nump_arr, imgsz, stride=stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(device)

        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
        t2 = time_synchronized()

        cropped_img_result = []

        # Process detections
        for det in pred:  # detections per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img_nump_arr.shape).round()

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    cropped_img_result.append(crop_img_by_bbox(xyxy, img_nump_arr))

        # Print time (inference + NMS)
        print(f'inference + NMS Done. ({t2 - t1:.3f}s)')

        img_arr_result.append(
            {
                "frame": idx,
                "timestamp": key["timestamp"],
                "image": cropped_img_result
            }
        )

    print(f'Done. ({time.time() - t0:.3f}s)')

    return img_arr_result


if __name__ == "__main__":
    print(cropImage("crowdhuman_yolov5m.pt", dummy))

