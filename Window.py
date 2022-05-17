import sys
from PyQt5.Qt import *
from PyQt5 import uic
import cv2
import Image


class MainWindow(QMainWindow):

    def __init__(self):
        QMainWindow.__init__(self)
        self.ui = uic.loadUi("Window.ui", self)
        self.setWindowTitle("image correction")
        self.setAcceptDrops(True)
        self.ui.lbl_source.mousePressEvent = self.labelMousePressEvent
        self.polygon = []
        self.image = None
        self.ui.btn_deform.released.connect(self.deform)

    def dragEnterEvent(self, event: QDragMoveEvent) -> None:
        event.accept()

    def dropEvent(self, event: QDropEvent) -> None:
        filename = event.mimeData().text().replace("file:///", "")
        self.image = cv2.imread(filename)
        if self.image is None:
            return
        y, x = self.image.shape[:2]
        self.ui.lbl_source.setFixedSize(x, y)
        self.lbl_source.setPixmap(QPixmap(QImage(self.image.data, x, y, 3 * x, QImage.Format_BGR888)))

    def labelMousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            y = event.y()
            x = event.x()
            self.polygon.append((x, y))
        elif event.button() == Qt.RightButton:
            self.polygon.clear()

    def deform(self):
        if self.image is None:
            print("load an image")
            return
        elif len(self.polygon) != 4:
            print("pick 4 points")
            return

        image = Image.deformImage(self.image, self.polygon)
        image = Image.colorCorrection(image)
        y, x = image.shape[:2]
        y *= 2
        x *= 2
        image = cv2.resize(image, (x, y))
        self.ui.lbl_target.setFixedSize(x, y)
        self.lbl_target.setPixmap(QPixmap(QImage(image.data, x, y, 3 * x, QImage.Format_BGR888)))


app = QApplication([])
window = MainWindow()
window.show()
sys.exit(app.exec())
