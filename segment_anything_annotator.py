import os
import argparse
import sys

from PyQt5.QtWidgets import QApplication

from salt.editor import Editor
from salt.interface import ApplicationInterface
# os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx-model-path", type=str, default="models/sam_onnx.onnx")
    parser.add_argument("--dataset-path", type=str, default="dataset")
    parser.add_argument("--categories", type=str)
    parser.add_argument("--save_segmap", action='store_true')
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
        coco_json_path=coco_json_path,
        save_segmaps=args.save_segmap
    )

    app = QApplication(sys.argv)
    window = ApplicationInterface(app, editor)
    window.show()
    sys.exit(app.exec_())