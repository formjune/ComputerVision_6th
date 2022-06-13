import os.path
import sys
from PyQt5.Qt import *
from PyQt5 import uic
import Image


class MainWindow(QWidget):

    def __init__(self):
        QWidget.__init__(self)
        self.ui = uic.loadUi("Window.ui", self)
        self.setWindowTitle("image correction")
        self.setFixedSize(601, 171)
        self.ui.btn_run.released.connect(self.run)

    def run(self):
        folder_in = self.ui.line_in.text()
        if not os.path.isdir(folder_in):
            print("входная директория не существует")
            return
        folder_out = self.ui.line_out.text()
        if not os.path.isdir(folder_out):
            print("выходная директория не существует")
            return
        Image.main(folder_in, folder_out, self.ui.chb_lighten.isChecked(), self.ui.spin_scale.value())


app = QApplication([])
window = MainWindow()
window.show()
sys.exit(app.exec())
