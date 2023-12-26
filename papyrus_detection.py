import argparse
import glob
import os

import PIL.Image
import cv2
import numpy as np
import torch
import torchvision
import tqdm
from groundingdino.util.inference import Model
from segment_anything import SamPredictor

from LightHQSAM.setup_light_hqsam import setup_model


PIL.Image.MAX_IMAGE_PIXELS = 933120000

parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
parser.add_argument("--dataset_path", type=str, required=True, help="path to dataset")
parser.add_argument("--output_path", type=str, required=True, help="path to segmented dataset")
args = parser.parse_args()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# GroundingDINO config and checkpoint
GROUNDING_DINO_CONFIG_PATH = "GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH = "./groundingdino_swint_ogc.pth"

# Building GroundingDINO inference model
grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH,
                             model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH,
                             device=DEVICE)


# Predict classes and hyper-param for GroundingDINO
CLASSES = ["cm ruler", "papyrus"]
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.25
NMS_THRESHOLD = 0.8

def crop_image(image, pixel_value=0):
    # Remove the zeros padding
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    crop_rows_gray = gray[~np.all(gray == pixel_value, axis=1), :]

    crop_rows = image[~np.all(gray == pixel_value, axis=1), :]
    cropped_image = crop_rows[:, ~np.all(crop_rows_gray == pixel_value, axis=0)]

    black_pixels = np.where(
        (cropped_image[:, :, 0] == 0) &
        (cropped_image[:, :, 1] == 0) &
        (cropped_image[:, :, 2] == 0)
    )

    # set those pixels to white
    cropped_image[black_pixels] = [255, 255, 255]

    return cropped_image


image_paths = glob.glob(os.path.join(args.dataset_path, '**', '*.jpg'), recursive=True)

for image_path in tqdm.tqdm(image_paths):

    # load image
    image = cv2.imread(image_path)

    # detect objects
    detections = grounding_dino_model.predict_with_classes(
        image=image,
        classes=CLASSES,
        box_threshold=BOX_THRESHOLD,
        text_threshold=BOX_THRESHOLD
    )

    # NMS post process
    nms_idx = torchvision.ops.nms(
        torch.from_numpy(detections.xyxy),
        torch.from_numpy(detections.confidence),
        NMS_THRESHOLD
    ).numpy().tolist()

    detections.xyxy = detections.xyxy[nms_idx]
    detections.confidence = detections.confidence[nms_idx]
    detections.class_id = detections.class_id[nms_idx]

    output_img_dir = image_path.replace(args.dataset_path, args.output_path)
    output_img_dir = os.path.splitext(output_img_dir)[0]

    for idx, (box, label, scores) in enumerate(zip(detections.xyxy.astype(int),  detections.class_id, detections.confidence)):
        cropped_img = image[box[1]:box[3], box[0]:box[2]]
        out_img_path = os.path.join(output_img_dir, CLASSES[label], f'{idx}_{round(scores, 2)}.jpg')
        os.makedirs(os.path.dirname(out_img_path), exist_ok=True)
        cv2.imwrite(out_img_path, cropped_img)