import numpy as np
import matplotlib.pyplot as plt
import cv2
from copy import deepcopy
from typing import Tuple

from pycocotools import mask as coco_mask

def overlay_mask_on_image(image, mask, color=(0, 0, 255), opacity=0.4):
    """
    Overlay mask on image.
    """
    gray_mask = mask.astype(np.uint8) * 255
    gray_mask = cv2.merge([gray_mask, gray_mask, gray_mask])
    color_mask = cv2.bitwise_and(gray_mask, color)
    masked_image = cv2.bitwise_and(image.copy(), color_mask)
    overlay_on_masked_image = cv2.addWeighted(masked_image, opacity, color_mask, 1 - opacity, 0)
    background = cv2.bitwise_and(image.copy(), cv2.bitwise_not(color_mask))
    image = cv2.add(background, overlay_on_masked_image)
    return image

def convert_ann_to_mask(ann, height, width):
    """
    Convert annotation to mask.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    for seg in ann["segmentation"]:
        poly = ann['segmentation']
        rles = coco_mask.frPyObjects(poly, height, width)
        rle = coco_mask.merge(rles)
        mask_instance = coco_mask.decode(rle)
        mask_instance = np.logical_not(mask_instance)
        mask = np.logical_or(mask, mask_instance)
    mask = np.logical_not(mask)
    return mask

def draw_box_on_image(image, ann, color):
    """
    Draw box on image.
    """
    x, y, w, h = ann["bbox"]
    x, y, w, h = int(x), int(y), int(w), int(h)
    image = cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
    return image

def draw_annotations(image, annotations, colors):
    """
    Draw annotations on image.
    """
    for ann, color in zip(annotations, colors):
        image = draw_box_on_image(image, ann, color)
        mask = convert_ann_to_mask(ann, image.shape[0], image.shape[1])
        image = overlay_mask_on_image(image, mask, color)
    return image
    
def draw_points(image, points, labels, colors={1: (0, 255, 0), 0: (255, 0, 0)}, radius=5):
    """
    Draw points on image.
    """
    for i in range(points.shape[0]):
        point = points[i, :]
        label = labels[i]
        color = colors[label]
        image = cv2.circle(image, tuple(point), radius, color, -1)
    return image

def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
    """
    Compute the output size given input size and target long side length.
    """
    scale = long_side_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    return (newh, neww)

def apply_coords(coords: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
    """
    Expects a numpy array of length 2 in the final dimension. Requires the
    original image size in (H, W) format.
    """
    old_h, old_w = original_size
    new_h, new_w = get_preprocess_shape(
        original_size[0], original_size[1], 1024
    )
    coords = deepcopy(coords).astype(float)
    coords[..., 0] = coords[..., 0] * (new_w / old_w)
    coords[..., 1] = coords[..., 1] * (new_h / old_h)
    return coords
