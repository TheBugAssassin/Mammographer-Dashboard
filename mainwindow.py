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
from PySide6.QtGui import QPixmap, QIcon
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

# # Important:
# # You need to run the following command to generate the ui_form.py file
# #     pyside6-uic form.ui -o ui_form.py, or
# #     pyside2-uic form.ui -o ui_form.py

class DicomConverterThread(QThread):
    progress = Signal(int)
    conversion_done = Signal(str)

    def __init__(self, dcm_path, output_dir, slice_number):
        super().__init__()
        self.dcm_path = dcm_path  # Store the DICOM file path
        self.output_dir = output_dir  # Store the output directory
        self.slice_number = slice_number  # Store the slice number
        self.converted_image_path = None  # Initialize the converted image path

    def run(self):
        try:
            ds = pydicom.dcmread(self.dcm_path)  # Read the DICOM file
            img_array = ds.pixel_array  # Extract pixel array from DICOM

            # If the image array is 3D, slice it according to the requested slice number
            if img_array.ndim == 3:
                if 0 <= self.slice_number < img_array.shape[0]:
                    img_array = img_array[int(self.slice_number)]
                else:
                    self.progress.emit(0)  # If invalid slice number, emit progress as 0
                    print("Invalid slice number")
                    return

            # Normalize the image array
            img_array = ((img_array - img_array.min()) / (img_array.max() - img_array.min()) * 255).astype(np.uint8)

            # Simulate progress during conversion
            for i in range(101):  # 0 to 100
                self.msleep(20)  # Adjust delay for smooth progress
                self.progress.emit(i)

            # Output the image path where it will be saved
            output_path = os.path.join(self.output_dir, "processed_slice.png")
            plt.imsave(output_path, img_array, cmap='gray', format='png')  # Save image as PNG

            # Resize the image to 512x512 pixels
            img = Image.open(output_path)
            img = img.resize((512, 512))  # Resize to 512x512
            img.save(output_path)  # Save the resized image

            # Emit the converted image path once the conversion is done
            self.converted_image_path = output_path
            self.conversion_done.emit(output_path)

        except Exception as e:
            print(f"Error: {e}")
            self.progress.emit(0)  # In case of error, emit progress as 0

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle("Mammographer Viewer")

        logo_path = os.path.join(os.path.dirname(__file__), 'logo2.png')  # Get the path to logo.png
        self.setWindowIcon(QIcon(logo_path))

        self.dcm_path = None
        self.converted_image_path = None  # Track the path of the converted image

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

    def select_dcm(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select DICOM File", "", "DICOM Files (*.dcm)")
        if file_path:
            self.dcm_path = file_path
            self.ui.label.setText(f"Selected: {os.path.basename(file_path)}")

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

        # Start the conversion thread
        self.thread = DicomConverterThread(self.dcm_path, output_dir, slice_number)
        self.thread.progress.connect(self.update_progress)  # Connect to update progress
        self.thread.conversion_done.connect(self.display_image)  # Connect to display image once done
        self.thread.start()

    def update_progress(self, value):
        if self.progress_dialog:
            self.progress_dialog.setValue(value)  # Update progress bar value

    def display_image(self, image_path):
        # Hide progress dialog once conversion is done
        if self.progress_dialog:
            self.progress_dialog.hide()

        self.converted_image_path = image_path
        pixmap = QPixmap.fromImage(ImageQt(Image.open(image_path)))  # Use ImageQt to convert PIL Image to QPixmap
        self.scene.clear()
        item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(item)
        self.graphicsView.setScene(self.scene)

    def save_image(self):
        if self.converted_image_path:
            # Ask the user where to save the converted image
            save_path, _ = QFileDialog.getSaveFileName(
                self, "Save Image", self.converted_image_path, "PNG Files (*.png)"
            )
            if save_path:
                # Save the image to the user-selected path
                img = Image.open(self.converted_image_path)
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

