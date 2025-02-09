# # This Python file uses the following encoding: utf-8
# import sys
# from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
# from PySide6.QtGui import QPixmap
# from PySide6.QtCore import Qt

# # Important:
# # You need to run the following command to generate the ui_form.py file
# #     pyside6-uic form.ui -o ui_form.py, or
# #     pyside2-uic form.ui -o ui_form.py

# from ui_form import Ui_MainWindow

# class MainWindow(QMainWindow):
#     def __init__(self, parent=None):
#         super().__init__(parent)
#         self.ui = Ui_MainWindow()
#         self.ui.setupUi(self)

#         self.button = self.findChild(QPushButton, "pushButton_2")
#         self.graphicsView = self.findChild(QGraphicsView, "graphicsView")

#         # Create a graphics scene
#         self.scene = QGraphicsScene()
#         self.graphicsView.setScene(self.scene)

#         self.button.clicked.connect(self.clicker)

#     def clicker(self):
#         file_name, _ = QFileDialog.getOpenFileName(self, "Load PNG", "", "PNG Files (*.png)")

#         if file_name:
#             pixmap = QPixmap(file_name)
#             self.scene.clear()  # Clear previous image
#             item = QGraphicsPixmapItem(pixmap)
#             self.scene.addItem(item)
#             self.graphicsView.fitInView(item.boundingRect(), Qt.KeepAspectRatio)

# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     widget = MainWindow()
#     widget.show()
#     sys.exit(app.exec())

import os
import pydicom
import numpy as np
import matplotlib.pyplot as plt
import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QComboBox, QProgressDialog,
    QDoubleSpinBox, QProgressBar, QFileDialog, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
)
from PySide6.QtGui import QPixmap, QIcon, QImage
from PySide6.QtCore import Qt, QThread, Signal
from ui_form import Ui_MainWindow  # Ensure form.ui is compiled to ui_form.py
from PIL import Image
from PIL.ImageQt import ImageQt

stylesheet = """
QMainWindow {
    background-color: #f9f9f9;  /* Light background color */
}

QPushButton {
    background-color: #ff80bf;  /* Light pink */
    color: white;
    font-size: 14px;
    border: 2px solid #ff66b3;  /* Border with darker pink */
    border-radius: 5px;
    padding: 10px 15px;
}

QPushButton:hover {
    background-color: #ff66b3;  /* Darker pink on hover */
    border-color: #ff4d94;  /* Darker border color */
}

QPushButton:pressed {
    background-color: #ff4d94;  /* Darker pink when pressed */
    border-color: #ff3385;
}

QLabel {
    font-size: 16px;
    color: #333333;  /* Dark gray for text */
    font-weight: bold;
}

QComboBox {
    background-color: #ffffff;  /* White background for combobox */
    border: 1px solid #ff80bf;  /* Light pink border */
    padding: 5px;
    font-size: 14px;
}

QComboBox:hover {
    border-color: #ff66b3;  /* Darker pink border on hover */
}

QComboBox::drop-down {
    border: none;
}

QDoubleSpinBox {
    background-color: #ffffff;
    border: 1px solid #ff80bf;
    padding: 5px;
    font-size: 14px;
}

QDoubleSpinBox:hover {
    border-color: #ff66b3;
}

QProgressBar {
    border: 1px solid #ff80bf;
    border-radius: 5px;
    text-align: center;
}

QProgressBar::chunk {
    background-color: #ff66b3;  /* Progress bar color */
    width: 1px;
}

QGraphicsView {
    border: 2px solid #ff80bf;
    border-radius: 8px;
}

QFileDialog {
    background-color: #ffffff;
}

QProgressDialog {
    background-color: #ffffff;
    border: 2px solid #ff80bf;
    border-radius: 10px;
}

QProgressDialog::label {
    color: #333333;
}

QProgressDialog::cancelButton {
    background-color: #ff80bf;
    border: 2px solid #ff66b3;
    color: white;
    border-radius: 5px;
}

QProgressDialog::cancelButton:hover {
    background-color: #ff66b3;
    border-color: #ff4d94;
}
"""

class DicomConverterThread(QThread):
    progress = Signal(int)
    conversion_done = Signal(np.ndarray)  # Emit the processed image array

    def __init__(self, dcm_path, slice_number):
        super().__init__()
        self.dcm_path = dcm_path
        self.slice_number = slice_number
        self.processed_image = None  # Store the processed image

    def run(self):
        try:
            ds = pydicom.dcmread(self.dcm_path)
            img_array = ds.pixel_array

            if img_array.ndim == 3:
                if 0 <= self.slice_number < img_array.shape[0]:
                    img_array = img_array[int(self.slice_number)]
                else:
                    self.progress.emit(0)
                    print("Invalid slice number")
                    return

            img_array = ((img_array - img_array.min()) / (img_array.max() - img_array.min()) * 255).astype(np.uint8)

            for i in range(101):
                self.msleep(20)
                self.progress.emit(i)

            self.processed_image = img_array  # Store processed image
            self.conversion_done.emit(img_array)  # Emit the image array

        except Exception as e:
            print(f"Error: {e}")
            self.progress.emit(0)

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle("Mammographer Viewer")

        logo_path = os.path.join(os.path.dirname(__file__), 'logo2.png')  # Get the path to logo.png
        self.setWindowIcon(QIcon(logo_path))

        self.dcm_path = None
        self.processed_image = None  # Store processed image

        # Find UI elements
        self.buttonFile = self.findChild(QPushButton, "pushButtonFile")
        self.graphicsView = self.findChild(QGraphicsView, "graphicsView")
        self.viewSelect = self.findChild(QComboBox, "comboBox")
        self.sliceNumber = self.findChild(QDoubleSpinBox, "doubleSpinBox")
        self.loadButton = self.findChild(QPushButton, "pushButton")
        self.extractButton = self.findChild(QPushButton, "extractButton")

        # Setup scene for displaying images
        self.scene = QGraphicsScene()
        self.graphicsView.setScene(self.scene)

        # Connect buttons
        self.buttonFile.clicked.connect(self.select_dcm)
        self.loadButton.clicked.connect(self.convert_dcm)
        self.extractButton.clicked.connect(self.save_image)

        self.progress_dialog = None  # Initially set to None
        self.progressBar = None  # Progress bar will be accessed through the dialog
        self.disable_ui()

    def disable_ui(self):
        self.viewSelect.setEnabled(False)
        self.sliceNumber.setEnabled(False)
        self.loadButton.setEnabled(False)
        self.extractButton.setEnabled(False)

    def enable_ui(self):
        self.viewSelect.setEnabled(True)
        self.sliceNumber.setEnabled(True)
        self.loadButton.setEnabled(True)
        self.extractButton.setEnabled(True)

    def select_dcm(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select DICOM File", "", "DICOM Files (*.dcm)")
        if file_path:
            try:
                pydicom.dcmread(file_path)
                self.dcm_path = file_path
                self.ui.label.setText(f"Selected: {os.path.basename(file_path)}")
                self.enable_ui()
            except Exception:
                self.ui.label.setText("Invalid DICOM file!")
                self.disable_ui()

    def convert_dcm(self):
        if not self.dcm_path:
            print("No DICOM file selected.")
            return

        slice_number = self.sliceNumber.value()
        output_dir = os.path.dirname(self.dcm_path)

        # Create and show progress dialog
        self.progress_dialog = QProgressDialog("Processing DICOM...", "Cancel", 0, 100, self)
        self.progress_dialog.setWindowTitle("Load Slice")
        self.progress_dialog.setWindowModality(Qt.WindowModal)  # Make dialog modal
        self.progress_dialog.setValue(0)  # Start at 0% progress
        self.progress_dialog.show()

        self.thread = DicomConverterThread(self.dcm_path, slice_number)
        self.thread.progress.connect(self.update_progress)
        self.thread.conversion_done.connect(self.display_image)
        self.thread.start()

    def update_progress(self, value):
        if self.progress_dialog:
            self.progress_dialog.setValue(value)  # Update progress bar value

    def display_image(self, img_array):
        self.progress_dialog.hide()
        self.processed_image = img_array  # Store processed image

        height, width = img_array.shape
        q_image = QImage(img_array.data, width, height, width, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_image)

        self.scene.clear()
        self.scene.addPixmap(pixmap)

    def save_image(self):
        if self.processed_image is not None:
            save_path, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "PNG Files (*.png)")
            if save_path:
                img = Image.fromarray(self.processed_image)
                img.save(save_path)
                print(f"Image saved to {save_path}")
        else:
            print("No image to save.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    #app.setStyleSheet(stylesheet)
    widget = MainWindow()
    widget.show()
    sys.exit(app.exec())

