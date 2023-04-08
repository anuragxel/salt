import os, copy
import numpy as np
import cv2
from utils import overlay_mask_on_image, draw_points, draw_annotations
from onnx_model import OnnxModel
from dataset_explorer import DatasetExplorer

onnx_model_path = "sam_onnx_example.onnx"
onnx_model_quantized_path = "sam_onnx_quantized_example.onnx"
onnx_model_path = onnx_model_quantized_path

onnx_helper = OnnxModel(onnx_model_path)
dataset_explorer = DatasetExplorer(
    "dataset", categories=["generic_object"], coco_json_path="dataset/annotations.json"
)

image_id = 0
category_id = 0
image, image_bgr, image_embedding = dataset_explorer.get_image_data(image_id)

input_point = np.array([])
input_label = np.array([])
low_res_logits = None
curr_mask = None
display = copy.deepcopy(image_bgr)


def update_display(draw_again=True):
    global input_point, input_label, low_res_logits, display, curr_mask, image_id, category_id, image, image_bgr, image_embedding
    if draw_again == False:
        display = copy.deepcopy(image_bgr)
    anns, colors = dataset_explorer.get_annotations(image_id, return_colors=True)
    display = draw_annotations(display, anns, colors)


def reset():
    global input_point, input_label, low_res_logits, display, curr_mask, image_id, category_id, image, image_bgr, image_embedding
    input_point = np.array([])
    input_label = np.array([])
    low_res_logits = None
    curr_mask = None


def save():
    global curr_mask
    if curr_mask is not None:
        dataset_explorer.add_annotation(image_id, category_id, curr_mask)


def on_mouse(x, y, label):
    global input_point, input_label, low_res_logits, display, curr_mask, image_id, category_id, image, image_bgr, image_embedding
    if len(input_point) == 0:
        input_point = np.array([[x, y]])
    else:
        input_point = np.vstack([input_point, np.array([x, y])])
    input_label = np.append(input_label, label)
    masks, low_res_logits = onnx_helper.call(
        image, image_embedding, input_point, input_label, low_res_logits=low_res_logits
    )
    display = copy.deepcopy(image_bgr)
    anns, colors = dataset_explorer.get_annotations(image_id, return_colors=True)
    display = draw_annotations(display, anns, colors)
    display = draw_points(display, input_point, input_label)
    display = overlay_mask_on_image(display, masks[0, 0, :, :])
    curr_mask = masks[0, 0, :, :]


def mouse_callback(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        on_mouse(x, y, 1)
    if event == cv2.EVENT_RBUTTONDOWN:
        on_mouse(x, y, 0)


reset()
anns, colors = dataset_explorer.get_annotations(image_id, return_colors=True)
display = draw_annotations(display, anns, colors)

cv2.namedWindow("AnnotationTool")
cv2.setMouseCallback("AnnotationTool", mouse_callback)
while True:
    cv2.imshow("AnnotationTool", display)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
    if key == ord("n"):
        reset()
        update_display(draw_again=False)
    if key == ord("s"):
        save()
        reset()
        update_display(draw_again=True)

dataset_explorer.save_annotation()
cv2.destroyAllWindows()
