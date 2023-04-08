import cv2
from salt.editor import Editor
    
class AnnotationInterface:
    def __init__(self, editor):
        self.editor = editor
    
    def run(self):
        pass

if __name__ == "__main__":
    editor = Editor(
        "models/sam_onnx_example.onnx",
        "dataset",
        ["generic_object"],
        "dataset/annotations.json",
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

    editor.save()

    cv2.destroyAllWindows()
