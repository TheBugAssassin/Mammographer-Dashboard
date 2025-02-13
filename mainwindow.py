# # This Python file uses the following encoding: utf-8
# import sys
# from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
# from PySide6.QtGui import QPixmap
# from PySide6.QtCore import Qt

# # Important:
# # You need to run the following command to generate the ui_form.py file
# #     pyside6-uic form.ui -o ui_form.py, or
# #     pyside2-uic form.ui -o ui_form.py

import os
import pydicom
import numpy as np
import matplotlib.pyplot as plt
import sys
import torch
import torch.nn as nn
import torchvision.transforms.v2 as transforms
import torchvision.models as models
import torchvision.transforms.functional as F
from torchvision.io import read_image
from typing import AnyStr, BinaryIO, Dict, List, NamedTuple, Optional, Union
from skimage.exposure import rescale_intensity
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QComboBox, QProgressDialog, QMessageBox,
    QDoubleSpinBox, QProgressBar, QFileDialog, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
)
from PySide6.QtGui import QPixmap, QIcon, QImage
from PySide6.QtCore import Qt, QThread, Signal
from ui_form import Ui_MainWindow  # Ensure form.ui is compiled to ui_form.py
from PIL import Image
from PIL.ImageQt import ImageQt

CLASS_MAPPING = {'Normal': 0, 'Actionable': 1, 'Benign': 2, 'Cancer': 3}
REVERSE_CLASS_MAPPING = {v: k for k, v in CLASS_MAPPING.items()}

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

    def __init__(self, dcm_path, slice_number, view_laterality):
        super().__init__()
        self.dcm_path = dcm_path
        self.slice_number = slice_number
        self.view_laterality = view_laterality
        self.processed_image = None  # Store the processed image

    def run(self):
        try:
            # ds = pydicom.dcmread(self.dcm_path)
            # img_array = ds.pixel_array
            # img_array = img_array[int(self.slice_number)]
            # img_array = ((img_array - img_array.min()) / (img_array.max() - img_array.min()) * 255).astype(np.uint8)

            img_array = dcmread_image(self.dcm_path, self.view_laterality, int(self.slice_number))
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
        self.model = load_model("resnet50_trained_iteration_4_100_epochs.pth")

        # Find UI elements
        self.buttonFile = self.findChild(QPushButton, "pushButtonFile")
        self.graphicsView = self.findChild(QGraphicsView, "graphicsView")
        self.viewSelect = self.findChild(QComboBox, "comboBox")
        self.sliceNumber = self.findChild(QDoubleSpinBox, "doubleSpinBox")
        self.loadButton = self.findChild(QPushButton, "pushButton")
        self.classifyButton = self.findChild(QPushButton, "classifyButton")
        self.extractButton = self.findChild(QPushButton, "extractButton")

        # Setup scene for displaying images
        self.scene = QGraphicsScene()
        self.graphicsView.setScene(self.scene)

        # Connect buttons
        self.buttonFile.clicked.connect(self.select_dcm)
        self.loadButton.clicked.connect(self.convert_dcm)
        self.classifyButton.clicked.connect(self.classify_image)
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
                ds = pydicom.dcmread(file_path)
                self.dcm_path = file_path
                self.ui.label.setText(f"Selected: {os.path.basename(file_path)}")
                if "NumberOfFrames" in ds:
                    num_slices = ds.NumberOfFrames
                    print(f"Total Slices: {num_slices}")
                    self.sliceNumber.setMaximum(num_slices - 1)  # Ensure 0-based indexing
                    self.sliceNumber.setValue(0)  # Reset to first slice
                else:
                    self.sliceNumber.setMaximum(0)  # Single slice case
                    self.sliceNumber.setValue(0)
                if "SliceThickness" in ds:
                    slice_thickness = ds.SliceThickness
                    print(f"Slice Thickness: {slice_thickness} mm")
                else:
                    print("Slice Thickness: N/A")
                self.enable_ui()
            except Exception:
                self.ui.label.setText("Invalid DICOM File!")
                self.disable_ui()

    def convert_dcm(self):
        if not self.dcm_path:
            print("No DICOM File Selected")
            return

        slice_number = self.sliceNumber.value()
        view_laterality = self.viewSelect.currentText()
        output_dir = os.path.dirname(self.dcm_path)

        # Create and show progress dialog
        self.progress_dialog = QProgressDialog("Processing DICOM...", "Cancel", 0, 100, self)
        self.progress_dialog.setWindowTitle("Load Slice")
        self.progress_dialog.setWindowModality(Qt.WindowModal)  # Make dialog modal
        self.progress_dialog.setValue(0)  # Start at 0% progress
        self.progress_dialog.show()

        self.thread = DicomConverterThread(self.dcm_path, slice_number, view_laterality)
        self.thread.progress.connect(self.update_progress)
        self.thread.conversion_done.connect(self.display_image)
        self.thread.start()

    def update_progress(self, value):
        if self.progress_dialog:
            self.progress_dialog.setValue(value)  # Update progress bar value

    def display_image(self, img_array):
        self.progress_dialog.hide()
        self.processed_image = img_array  # Store processed image

        img = Image.fromarray(self.processed_image).resize((512, 512), Image.LANCZOS)
        self.processed_image = np.array(img)  # Store resized image for later saving

        height, width = self.processed_image.shape[:2]  # Ensure dimensions are correct
        q_image = QImage(self.processed_image.data, width, height, width * self.processed_image.shape[2] if len(self.processed_image.shape) == 3 else width, QImage.Format_Grayscale8 if len(self.processed_image.shape) == 2 else QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image).scaled(512, 512, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        self.scene.clear()
        self.scene.addPixmap(pixmap)
        self.graphicsView.setSceneRect(0, 0, 512, 512)

    def save_image(self):
        if self.processed_image is not None:
            save_path, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "PNG Files (*.png)")
            if save_path:
                img = Image.fromarray(self.processed_image)
                img.save(save_path)
                print(f"Image Saved @ {save_path}")
        else:
            print("No Image")

    def classify_image(self):
        if self.processed_image is None:
            QMessageBox.warning(self, "Error", "No image loaded for classification.")
            return

        try:
            # Convert processed NumPy array to PIL image
            img = Image.fromarray(self.processed_image)

            # Convert to grayscale (1-channel)
            img = img.convert("L")

            # Apply transformations
            img_tensor = image_transforms(torch.tensor(np.array(img)).unsqueeze(0))

            # Add batch dimension
            img_tensor = img_tensor.unsqueeze(0)

            # Run inference
            with torch.no_grad():
                outputs = self.model(img_tensor)
                _, predicted_class = torch.max(outputs, 1)

            # Map class index to label
            class_label = REVERSE_CLASS_MAPPING[predicted_class.item()]

            # Show result in a dialog box
            QMessageBox.information(self, "Classification Result", f"{class_label}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Classification Failed: {str(e)}")

def _get_image_laterality(pixel_array: np.ndarray) -> str:
    left_edge = np.sum(pixel_array[:, 0])  # sum of left edge pixels
    right_edge = np.sum(pixel_array[:, -1])  # sum of right edge pixels
    return "R" if left_edge < right_edge else "L"


def _get_window_center(ds: pydicom.dataset.FileDataset) -> np.float32:
    return np.float32(ds[0x5200, 0x9229][0][0x0028, 0x9132][0][0x0028, 0x1050].value)


def _get_window_width(ds: pydicom.dataset.FileDataset) -> np.float32:
    return np.float32(ds[0x5200, 0x9229][0][0x0028, 0x9132][0][0x0028, 0x1051].value)

def dcmread_image(
    fp: Union[str, "os.PathLike[AnyStr]", BinaryIO],
    view: str,
    index: Optional[np.uint] = None,
) -> np.ndarray:
    """Read pixel array from DBT DICOM file"""
    ds = pydicom.dcmread(fp)
    ds.decompress(handler_name="pylibjpeg")
    pixel_array = ds.pixel_array
    view_laterality = view[0].upper()
    image_laterality = _get_image_laterality(pixel_array[index or 0])
    if index is not None:
        pixel_array = pixel_array[index]
    if not image_laterality == view_laterality:
        pixel_array = np.flip(pixel_array, axis=(-1, -2))
    window_center = _get_window_center(ds)
    window_width = _get_window_width(ds)
    low = (2 * window_center - window_width) / 2
    high = (2 * window_center + window_width) / 2
    pixel_array = rescale_intensity(
        pixel_array, in_range=(low, high), out_range="dtype"
    )
    return pixel_array

# Load the trained ResNet-50 model
def load_model(model_path, num_classes=4):
    model = models.resnet50(weights=None)  # No pre-trained weights
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Adjust for 1-channel input
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)  # Modify output layer for classification

    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Define the same image transformations as used during training
image_transforms = transforms.Compose([
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Lambda(lambda img: F.crop(img, top=50, left=0, height=400, width=400)),
    transforms.Resize((224, 224)),  # ResNet-50 expects 224x224 input
])

if __name__ == "__main__":
    app = QApplication(sys.argv)
    #app.setStyleSheet(stylesheet)
    widget = MainWindow()
    widget.show()
    sys.exit(app.exec())

