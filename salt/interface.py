import cv2
from PyQt5.QtWidgets import (
    QScrollArea,
    QWidget,
    QVBoxLayout,
    QLabel,
    QGraphicsView,
    QGraphicsScene,
    QApplication,
    QListWidget,
    QListWidgetItem,
    QAbstractItemView,
)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QWheelEvent, QMouseEvent
from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtWidgets import (
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QLabel,
    QRadioButton,
    QInputDialog
)

selected_annotations = []


class CustomGraphicsView(QGraphicsView):
    def __init__(self, editor, tracking_mode=False):
        super(CustomGraphicsView, self).__init__()

        self.editor = editor
        self.tracking_mode = tracking_mode

        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setRenderHint(QPainter.TextAntialiasing)

        self.setOptimizationFlag(QGraphicsView.DontAdjustForAntialiasing, True)
        self.setOptimizationFlag(QGraphicsView.DontSavePainterState, True)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)

        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setInteractive(True)

        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        self.image_item = None

    def set_image(self, q_img, fit_to_view=False):
        pixmap = QPixmap.fromImage(q_img)
        first_image = self.image_item is None

        # Preserve current scroll position when refreshing the same image view
        # (e.g. ROI updates) so zoom/pan remain pixel-stable.
        prev_scroll = None
        if not first_image:
            prev_scroll = (
                self.horizontalScrollBar().value(),
                self.verticalScrollBar().value(),
            )

        if self.image_item:
            self.image_item.setPixmap(pixmap)
            self.setSceneRect(QRectF(pixmap.rect()))
        else:
            self.image_item = self.scene.addPixmap(pixmap)
            self.setSceneRect(QRectF(pixmap.rect()))

        # Fit only on first load (or when explicitly requested), otherwise
        # keep the user's current zoom level.
        if fit_to_view or first_image:
            view_size = self.viewport().size()
            img_size = pixmap.size()
            if img_size.width() > 0 and img_size.height() > 0:
                scale_x = view_size.width() / img_size.width()
                scale_y = view_size.height() / img_size.height()
                scale = min(scale_x, scale_y)
                self.resetTransform()
                self.scale(scale, scale)
            else:
                self.resetTransform()
            self.centerOn(self.sceneRect().center())
        elif prev_scroll is not None:
            h_val, v_val = prev_scroll
            h_bar = self.horizontalScrollBar()
            v_bar = self.verticalScrollBar()
            h_bar.setValue(min(max(h_val, h_bar.minimum()), h_bar.maximum()))
            v_bar.setValue(min(max(v_val, v_bar.minimum()), v_bar.maximum()))

    def wheelEvent(self, event: QWheelEvent):
        modifiers = QApplication.keyboardModifiers()
        if modifiers == Qt.ControlModifier:
            adj = (event.angleDelta().y() / 120) * 0.1
            self.scale(1 + adj, 1 + adj)
        else:
            delta_y = event.angleDelta().y()
            delta_x = event.angleDelta().x()
            x = self.horizontalScrollBar().value()
            self.horizontalScrollBar().setValue(x - delta_x)
            y = self.verticalScrollBar().value()
            self.verticalScrollBar().setValue(y - delta_y)

    def keyPressEvent(self, event):
        adj = 0.1
        # Zoom in/out with Ctrl++ and Ctrl+- (applies only if this is the image view widget)
        if event.modifiers() == Qt.ControlModifier:
            if event.key() == Qt.Key_Plus or event.key() == Qt.Key_Equal:
                self.scale(1 + adj, 1 + adj)
            elif event.key() == Qt.Key_Minus:
                self.scale(1 - adj, 1 - adj)
                
        super().keyPressEvent(event)

    def imshow(self, img, fit_to_view=False):
        height, width, channel = img.shape
        bytes_per_line = 3 * width
        q_img = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        self.set_image(q_img, fit_to_view=fit_to_view)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self.tracking_mode:
            return

        if self.image_item is None:
            super().mousePressEvent(event)
            return

        # FUTURE USE OF RIGHT CLICK EVENT IN THIS AREA
        modifiers = QApplication.keyboardModifiers()
        if modifiers == Qt.ControlModifier:
            print("Control/ Command key pressed during a mouse click")
            # self.editor.remove_click([int(x), int(y)])
        else:
            if event.button() not in (Qt.LeftButton, Qt.RightButton):
                super().mousePressEvent(event)
                return

            pos = event.pos()
            pos_in_item = self.mapToScene(pos) - self.image_item.pos()
            x, y = pos_in_item.x(), pos_in_item.y()
            if event.button() == Qt.LeftButton:
                label = 1
            else:
                label = 0
            self.editor.add_click([int(x), int(y)], label, selected_annotations)
        self.imshow(self.editor.display)


class ApplicationInterface(QWidget):
    def __init__(self, app, editor, tracking_mode=False, panel_size=(1920, 1080)):
        super(ApplicationInterface, self).__init__()
        self.app = app
        self.editor = editor
        self.panel_size = panel_size
        self.tracking_mode = tracking_mode

        self.layout = QVBoxLayout()

        self.top_bar = self.get_top_bar()
        self.layout.addWidget(self.top_bar)

        self.main_window = QHBoxLayout()
        if not tracking_mode:
            self.graphics_view = CustomGraphicsView(editor)
            self.main_window.addWidget(self.graphics_view)
        else:
            self.images_column = QVBoxLayout()
            self.graphics_view = CustomGraphicsView(editor)
            self.images_column.addWidget(self.graphics_view)
            self.next_graphics_view = CustomGraphicsView(editor, tracking_mode=True)
            self.images_column.addWidget(self.next_graphics_view)
            self.main_window.addLayout(self.images_column)

        self.panel = self.get_side_panel()
        self.panel_annotations = QListWidget()
        self.panel_annotations.setFixedWidth(200)
        self.panel_annotations.setSelectionMode(QAbstractItemView.MultiSelection)
        self.panel_annotations.itemClicked.connect(self.annotation_list_item_clicked)
        self.get_side_panel_annotations()
        self.main_window.addWidget(self.panel)
        self.main_window.addWidget(self.panel_annotations)

        self.layout.addLayout(self.main_window)

        self.setLayout(self.layout)

        self.update_view()

    def update_view(self):
        self.graphics_view.imshow(self.editor.display)
        self.get_side_panel_annotations()

        if self.tracking_mode:
            self.next_graphics_view.imshow(self.editor.draw_next_image_with_annotations())

    def reset(self):
        global selected_annotations
        self.editor.reset(selected_annotations)
        self.update_view()

    def add(self):
        global selected_annotations
        self.editor.save_ann()
        self.editor.reset(selected_annotations)
        self.update_view()

    def next_image(self):
        global selected_annotations
        self.editor.next_image()
        selected_annotations = []
        self.update_view()

    def prev_image(self):
        global selected_annotations
        self.editor.prev_image()
        selected_annotations = []
        self.update_view()
    
    def go_to_image(self):
        global selected_annotations
        
        # propose a list of image ids to go to
        # use image ids not a sequential index
        items = self.editor.dataset_explorer.getImgIds()
        items = [str(i) for i in sorted(items)]
        item, ok = QInputDialog.getItem(self, "Go To Image", "Select Image ID:", items, 0, False)
        if not ok or not item:
            return
        image_id = int(item)

        if ok:
            self.editor.go_to_image(image_id)
            selected_annotations = []
            self.update_view()

    def toggle(self):
        global selected_annotations
        self.editor.toggle(selected_annotations)
        self.update_view()

    def transparency_up(self):
        global selected_annotations
        self.editor.step_up_transparency(selected_annotations)
        self.update_view()

    def transparency_down(self):
        self.editor.step_down_transparency(selected_annotations)
        self.update_view()

    def save_all(self):
        self.editor.save()

    def change_category(self):
        global selected_annotations
        self.editor.change_category(selected_annotations)
        self.editor.reset(selected_annotations)
        self.update_view()

    def get_top_bar(self):
        top_bar = QWidget()
        button_layout = QHBoxLayout(top_bar)
        # suppress warning "QLayout::addChildLayout: layout "" already has a parent"
        # self.layout.addLayout(button_layout)
        buttons = [
            ("Add", lambda: self.add()),
            ("Reset", lambda: self.reset()),
            ("Prev", lambda: self.prev_image()),
            ("Next", lambda: self.next_image()),
            ("Go To", lambda: self.go_to_image()),
            ("Toggle", lambda: self.toggle()),
            ("Transparency Up", lambda: self.transparency_up()),
            ("Transparency Down", lambda: self.transparency_down()),
            ("Save", lambda: self.save_all()),
            (
                "Remove Selected Annotations",
                lambda: self.delete_annotations(),
            ),
            ("Change Category", lambda: self.change_category()),
        ]
        for button, lmb in buttons:
            bt = QPushButton(button)
            bt.clicked.connect(lmb)
            button_layout.addWidget(bt)

        return top_bar

    def get_side_panel(self):
        panel = QWidget()
        panel_layout = QVBoxLayout(panel)
        categories, colors = self.editor.get_categories(get_colors=True)
        label_array = []
        for i, _ in enumerate(categories):
            label_array.append(QRadioButton(categories[i]))
            label_array[i].clicked.connect(lambda state, x=categories[i]: self.editor.select_category(x))
            label_array[i].setStyleSheet("background-color: rgba({},{},{},0.6)".format(*colors[i][::-1]))
            panel_layout.addWidget(label_array[i])

        scroll = QScrollArea()
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setWidget(panel)
        scroll.setFixedWidth(200)
        return scroll

    def get_side_panel_annotations(self):
        anns, colors = self.editor.list_annotations()
        list_widget = self.panel_annotations
        list_widget.clear()
        # anns, colors = self.editor.get_annotations(self.editor.image_id)
        categories = self.editor.get_categories(get_colors=False)
        for i, ann in enumerate(anns):
            listWidgetItem = QListWidgetItem(str(ann["id"]) + " - " + (categories[ann["category_id"]]))
            list_widget.addItem(listWidgetItem)
        return list_widget

    def delete_annotations(self):
        global selected_annotations
        for annotation in selected_annotations:
            self.editor.delete_annotations(annotation)
        self.get_side_panel_annotations()
        selected_annotations = []
        self.reset()

    def annotation_list_item_clicked(self, item):
        global selected_annotations
        if item.isSelected():
            selected_annotations.append(int(item.text().split(" ")[0]))
            self.editor.draw_selected_annotations(selected_annotations)
        else:
            selected_annotations.remove(int(item.text().split(" ")[0]))
            self.editor.draw_selected_annotations(selected_annotations)
        self.graphics_view.imshow(self.editor.display)

    def keyPressEvent(self, event):
        # if event.key() == Qt.Key_Escape:
        #     self.app.quit()
        if event.key() == Qt.Key_A:
            self.prev_image()
        if event.key() == Qt.Key_D:
            self.next_image()
        if event.key() == Qt.Key_K:
            self.transparency_down()
        if event.key() == Qt.Key_L:
            self.transparency_up()
        if event.key() == Qt.Key_N:
            self.add()
        if event.key() == Qt.Key_R:
            self.reset()
        if event.key() == Qt.Key_T:
            self.toggle()
        if event.key() == Qt.Key_C:
            self.change_category()
        if event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_S:
            self.save_all()
        elif event.key() == Qt.Key_Space:
            print("Space pressed")
            # self.clear_annotations(selected_annotations)
            # Do something if the space bar is pressed
            # pass
