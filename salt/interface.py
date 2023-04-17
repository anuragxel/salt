import cv2
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QGraphicsView,
    QGraphicsScene,
    QApplication,
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
)


class CustomGraphicsView(QGraphicsView):
    def __init__(self, editor):
        super(CustomGraphicsView, self).__init__()

        self.editor = editor
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

    def set_image(self, q_img):
        pixmap = QPixmap.fromImage(q_img)
        if self.image_item:
            self.image_item.setPixmap(pixmap)
            self.setSceneRect(QRectF(pixmap.rect()))
        else:
            self.image_item = self.scene.addPixmap(pixmap)
            self.setSceneRect(QRectF(pixmap.rect()))

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

    def imshow(self, img):
        height, width, channel = img.shape
        bytes_per_line = 3 * width
        q_img = QImage(
            img.data, width, height, bytes_per_line, QImage.Format_RGB888
        ).rgbSwapped()
        self.set_image(q_img)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        pos = event.pos()
        pos_in_item = self.mapToScene(pos) - self.image_item.pos()
        x, y = pos_in_item.x(), pos_in_item.y()
        if event.button() == Qt.LeftButton:
            label = 1
        elif event.button() == Qt.RightButton:
            label = 0
        self.editor.add_click([int(x), int(y)], label)
        self.imshow(self.editor.display)


class ApplicationInterface(QWidget):
    def __init__(self, app, editor, panel_size=(1920, 1080)):
        super(ApplicationInterface, self).__init__()

        self.app = app
        self.editor = editor
        self.panel_size = panel_size

        self.layout = QVBoxLayout()

        self.top_bar = self.get_top_bar()
        self.layout.addWidget(self.top_bar)

        self.main_window = QHBoxLayout()

        self.graphics_view = CustomGraphicsView(editor)
        self.main_window.addWidget(self.graphics_view)

        self.panel = self.get_side_panel()
        self.main_window.addWidget(self.panel)
        self.layout.addLayout(self.main_window)

        self.setLayout(self.layout)

        self.graphics_view.imshow(self.editor.display)

    def reset(self):
        self.editor.reset()
        self.graphics_view.imshow(self.editor.display)

    def add(self):
        self.editor.save_ann()
        self.editor.reset()
        self.graphics_view.imshow(self.editor.display)

    def next_image(self):
        self.editor.next_image()
        self.graphics_view.imshow(self.editor.display)

    def prev_image(self):
        self.editor.prev_image()
        self.graphics_view.imshow(self.editor.display)

    def toggle(self):
        self.editor.toggle()
        self.graphics_view.imshow(self.editor.display)

    def transparency_up(self):
        self.editor.step_up_transparency()
        self.graphics_view.imshow(self.editor.display)

    def transparency_down(self):
        self.editor.step_down_transparency()
        self.graphics_view.imshow(self.editor.display)

    def save_all(self):
        self.editor.save()

    def get_top_bar(self):
        top_bar = QWidget()
        button_layout = QHBoxLayout(top_bar)
        self.layout.addLayout(button_layout)
        buttons = [
            ("Add", lambda: self.add()),
            ("Reset", lambda: self.reset()),
            ("Prev", lambda: self.prev_image()),
            ("Next", lambda: self.next_image()),
            ("Toggle", lambda: self.toggle()),
            ("Transparency Up", lambda: self.transparency_up()),
            ("Transparency Down", lambda: self.transparency_down()),
            ("Save", lambda: self.save_all()),
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
            label_array[i].clicked.connect(
                lambda state, x=categories[i]: self.editor.select_category(x)
            )
            label_array[i].setStyleSheet(
                "background-color: rgba({},{},{},0.6)".format(*colors[i][::-1])
            )
            panel_layout.addWidget(label_array[i])
        return panel

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.app.quit()
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
        if event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_S:
            self.save_all()
        # elif event.key() == Qt.Key_Space:
        #     # Do something if the space bar is pressed
        #     pass
