import numpy as np
import matplotlib.pyplot as plt
import cv2
from copy import deepcopy
from typing import Tuple

from pycocotools import mask as coco_mask

def overlay_mask_on_image(image, mask, color=(0, 0, 255), opacity=0.4):
    gray_mask = mask.astype(np.uint8) * 255
    gray_mask = cv2.merge([gray_mask, gray_mask, gray_mask])
    color_mask = cv2.bitwise_and(gray_mask, color)
    masked_image = cv2.bitwise_and(image.copy(), color_mask)
    overlay_on_masked_image = cv2.addWeighted(
        masked_image, opacity, color_mask, 1 - opacity, 0
    )
    background = cv2.bitwise_and(image.copy(), cv2.bitwise_not(color_mask))
    image = cv2.add(background, overlay_on_masked_image)
    return image


def convert_ann_to_mask(ann, height, width):
    mask = np.zeros((height, width), dtype=np.uint8)
    poly = ann["segmentation"]
    rles = coco_mask.frPyObjects(poly, height, width)
    rle = coco_mask.merge(rles)
    mask_instance = coco_mask.decode(rle)
    mask_instance = np.logical_not(mask_instance)
    mask = np.logical_or(mask, mask_instance)
    mask = np.logical_not(mask)
    return mask


def draw_box_on_image(image, ann, color):
    x, y, w, h = ann["bbox"]
    x, y, w, h = int(x), int(y), int(w), int(h)
    image = cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    return image


def draw_annotations(image, annotations, colors):
    for ann, color in zip(annotations, colors):
        image = draw_box_on_image(image, ann, color)
        mask = convert_ann_to_mask(ann, image.shape[0], image.shape[1])
        image = overlay_mask_on_image(image, mask, color)
    return image


def draw_points(
    image, points, labels, colors={1: (0, 255, 0), 0: (255, 0, 0)}, radius=5
):
    for i in range(points.shape[0]):
        point = points[i, :]
        label = labels[i]
        color = colors[label]
        image = cv2.circle(image, tuple(point), radius, color, -1)
    return image
