import numpy as np
import onnxruntime

from salt.utils import apply_coords


class OnnxModel:
    def __init__(self, onnx_model_path, threshold=0.5):
        self.ort_session = onnxruntime.InferenceSession(
            onnx_model_path, providers=["CPUExecutionProvider"]
        )
        self.threshold = threshold

    def __translate_input(
        self,
        image,
        image_embedding,
        input_point,
        input_label,
        input_box=None,
        onnx_mask_input=None,
    ):
        if input_box is None:
            onnx_coord = np.concatenate([input_point, np.array([[0.0, 0.0]])], axis=0)[
                None, :, :
            ]
            onnx_label = np.concatenate([input_label, np.array([-1])], axis=0)[
                None, :
            ].astype(np.float32)
        else:
            onnx_box_coords = input_box.reshape(2, 2)
            onnx_box_labels = np.array([2, 3])
            onnx_coord = np.concatenate([input_point, onnx_box_coords], axis=0)[
                None, :, :
            ]
            onnx_label = np.concatenate([input_label, onnx_box_labels], axis=0)[
                None, :
            ].astype(np.float32)

        onnx_coord = apply_coords(onnx_coord, image.shape[:2]).astype(np.float32)
        if onnx_mask_input is None:
            onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
            onnx_has_mask_input = np.zeros(1, dtype=np.float32)
        else:
            onnx_has_mask_input = np.ones(1, dtype=np.float32)
        ort_inputs = {
            "image_embeddings": image_embedding,
            "point_coords": onnx_coord,
            "point_labels": onnx_label,
            "mask_input": onnx_mask_input,
            "has_mask_input": onnx_has_mask_input,
            "orig_im_size": np.array(image.shape[:2], dtype=np.float32),
        }
        return ort_inputs

    def call(
        self,
        image,
        image_embedding,
        input_point,
        input_label,
        selected_box=None,
        low_res_logits=None,
    ):
        onnx_mask_input = None
        input_box = None
        if low_res_logits is not None:
            onnx_mask_input = low_res_logits
        if input_box is not None:
            input_box = selected_box
        ort_inputs = self.__translate_input(
            image,
            image_embedding,
            input_point,
            input_label,
            input_box=input_box,
            onnx_mask_input=onnx_mask_input,
        )
        masks, _, low_res_logits = self.ort_session.run(None, ort_inputs)
        masks = masks > self.threshold
        return masks, low_res_logits
