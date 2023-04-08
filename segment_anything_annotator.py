import os
import argparse

import cv2

from salt.editor import Editor

# @TODO: Add Tktinter GUI    
class AnnotationInterface:
    def __init__(self, editor):
        self.editor = editor
    
    def run(self):
        pass

if __name__ == "__main__":
    print("Barebones annotation interface for segment_anything model")
    print("Press 'r' to reset current mask annotation")
    print("Press 't' to toggle visibility of previously made annotations")
    print("Press 'n' to save and next image")
    print("Press 'd' to goto next image")
    print("Press 'a' to goto prev image")
    print("Press 'w' to annotate next category")
    print("Press 's' to annotate prev category")
    print("Press 'l' to increase previously made annotations transparency")
    print("Press 'k' to decrease previously made annotations transparency")
    print("Press 'q' to quit")
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx-model-path", type=str, default="models/sam_onnx.onnx")
    parser.add_argument("--dataset-path", type=str, default="dataset")
    parser.add_argument("--categories", type=str)
    args = parser.parse_args()

    onnx_model_path = args.onnx_model_path
    dataset_path = args.dataset_path
    categories = None
    if args.categories is not None:
        categories = args.categories.split(",")
    
    coco_json_path = os.path.join(dataset_path,"annotations.json")


    editor = Editor(
        onnx_model_path,
        dataset_path,
        categories=categories,
        coco_json_path=coco_json_path
    )

    def mouse_callback(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            editor.add_click([x, y], 1)
        if event == cv2.EVENT_RBUTTONDOWN:
            editor.add_click([x, y], 0)

    cv2.namedWindow("Editor")
    cv2.setMouseCallback("Editor", mouse_callback)
    while True:
        cv2.imshow("Editor", editor.display)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        if key == ord("r"):
            editor.reset()
        if key == ord("t"):
            editor.toggle_anns()
        if key == ord("n"):
            editor.save_ann()
            editor.reset()
        if key == ord("d"):
            editor.next_image()
        if key == ord("a"):
            editor.prev_image()
        if key == ord("w"):
            editor.next_category()
        if key == ord("s"):
            editor.prev_category()
        if key == ord("l"):
            editor.step_up_transparency()
        if key == ord("k"):
            editor.step_down_transparency()

    editor.save()

    cv2.destroyAllWindows()
