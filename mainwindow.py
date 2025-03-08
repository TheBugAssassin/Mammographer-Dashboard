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
import matplotlib.patches as patches
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms.v2 as transforms
import torchvision.transforms as T
import torchvision.models as models
import torchvision.transforms.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_Weights
from torchvision.io import read_image
from torchvision.ops import box_iou
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
from io import BytesIO
from datetime import datetime

CLASS_MAPPING = {'Normal': 0, 'Actionable': 1, 'Benign': 2, 'Cancer': 3}
REVERSE_CLASS_MAPPING = {v: k for k, v in CLASS_MAPPING.items()}

image_transforms = transforms.Compose([
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Lambda(lambda img: F.crop(img, top=50, left=0, height=400, width=400)),
])

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

global_image = "NULL"
global_image_view = "NULL"
global_image_path = "NULL"

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
        self.setWindowTitle("MammoScan")

        logo_path = os.path.join(os.path.dirname(__file__), 'logo2.png')  # Get the path to logo.png
        self.setWindowIcon(QIcon(logo_path))

        self.dcm_path = None
        self.processed_image = None  # Store processed image
        self.model = load_model_resnet("resnet50_trained_iteration_4_100_epochs.pth")
        self.classification = 'null'

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
                global global_image_path
                global_image_path = file_path
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
        global global_image
        global_image = img
        self.processed_image = np.array(img)  # Store resized image for later saving

        height, width = self.processed_image.shape[:2]  # Ensure dimensions are correct
        q_image = QImage(self.processed_image.data, width, height, width * self.processed_image.shape[2] if len(self.processed_image.shape) == 3 else width, QImage.Format_Grayscale8 if len(self.processed_image.shape) == 2 else QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image).scaled(512, 512, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        self.scene.clear()
        self.scene.addPixmap(pixmap)
        self.graphicsView.setSceneRect(0, 0, 512, 512)

    def display_png(self, png_path):
        img = Image.open(png_path)
        img_array = np.array(img)
        if img_array.shape[2] == 4:
            img_array = img_array[:, :, :3]
        img_array = img_array.astype(np.uint8)
        height, width = img_array.shape[:2]
        q_image = QImage(img_array.data, width, height, width * 3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image).scaled(512, 512, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.scene.clear()
        self.scene.addPixmap(pixmap)
        self.graphicsView.setSceneRect(0, 0, self.graphicsView.width(), self.graphicsView.height())

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
            QMessageBox.warning(self, "Error", "No Image Loaded")
            return

        try:
            img = Image.fromarray(self.processed_image)
            global global_image
            global_image = img
            img = img.convert("L")
            img_tensor = image_transforms(torch.tensor(np.array(img)).unsqueeze(0))
            img_tensor = img_tensor.unsqueeze(0)

            with torch.no_grad():
                outputs = self.model(img_tensor)
                _, predicted_class = torch.max(outputs, 1)

            class_label = REVERSE_CLASS_MAPPING[predicted_class.item()]
            QMessageBox.information(self, "Classification Result", f"{class_label}")
            self.bounding_box()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Classification Failed: {str(e)}")

    def get_model(self,num_classes):
        # Use the newer weights parameter instead of pretrained
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
        #model.load_state_dict(torch.load("fasterrcnn_resnet50_fpn_coco-258fb6c6.pth"))

        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        model.roi_heads.nms_thresh = 0.3

        return model

    def visualize_boxes(self, image, pred_boxes, pred_labels, class_mapping):
        """Visualizes predicted and ground truth boxes with labels on the image."""
        image = T.ToPILImage()(image.cpu())
        fig, ax = plt.subplots(1)
        ax.imshow(image)
        ax.set_axis_off()

        pred_color = 'red'  # Predicted boxes
        gt_color = 'yellow'  # Ground truth boxes

        # --- Plot Predicted Boxes ---
        for box, label in zip(pred_boxes, pred_labels):
            x1, y1, x2, y2 = box.cpu().numpy()
            label_name = class_mapping.get(label.item(), str(label.item()))
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=pred_color, facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1 - 5, f"Pred: {label_name}", color=pred_color, fontsize=8, bbox=dict(facecolor='white', alpha=0.6))

        # Save the plot to a BytesIO object with no background
        buf = BytesIO()
        plt.savefig(buf, format='png', facecolor='none', edgecolor='none', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        # Create a PIL Image from the BytesIO object
        image = Image.open(buf)
        plt.close(fig)  # Close the figure to release resources

        return image # Return the image

    def test(self,model, image, device):
        """Run model on test set and evaluate mAP."""
        print('Evaluating ...')
        model.eval()
        all_iou = []

        with torch.no_grad():
                images = [image.to(device)]

                # Get predictions (no need for loss)
                predictions = model(images)  # Returns list of dicts with 'boxes', 'labels', 'scores'

                for img, pred in zip(images, predictions):

                    image_with_boxes = self.visualize_boxes(
                        img.cpu(), pred['boxes'], pred['labels'],
                        REVERSE_CLASS_MAPPING
                    )
                    return image_with_boxes

    def bounding_box(self):
        num_classes = len(CLASS_MAPPING)
        model_path = "faster_rcnn_det_test_101.pth"  # Update with actual path

        device = 'cpu'
        if torch.cuda.is_available():
            device = 'cuda'
        print('\t\tusing device ', device)

        model = self.get_model(num_classes)
        #device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        device = torch.device('cpu')
        model.to(device)

        model.load_state_dict(torch.load(model_path, map_location=device))

        if isinstance(global_image, Image.Image):
            img = global_image
        else:
            img = Image.open(global_image)  # Adjust if global_image is a path to a file

        transform = transforms.ToTensor()
        image_tensor = transform(img)  # Add batch dimension

        # Normalize the image to [0, 1] (already done by ToTensor but ensure it's done)
        image_tensor = image_tensor.float() / image_tensor.max()

        #image = read_image(global_image)

        # image = global_image.type(torch.float32)
        # image = image / image.max()

        # Begin the testing of the model with testing dataset and specific neural network (.pth file)
        image_with_boxes = self.test(
            model=model,
            image=image_tensor,
            device=device
        )

        #self.display_image(image_with_boxes)
        #image_with_boxes_array = np.array(image_with_boxes)

        save_path = "mamogram.png"  # Specify your save path
        image_with_boxes.save(save_path, 'PNG')

        # Display the saved image in the front-end
        self.display_png(save_path)
        #self.display_image(image_with_boxes)

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
    global global_image_view
    global_image_view = image_laterality
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

def load_model_resnet(model_path, num_classes=4):
    model = models.resnet50(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model


def scale_bbox(bbox, original_width, original_height, target_width=512, target_height=512):
    x, y, width, height = bbox
    x = x * target_width / original_width
    y = y * target_height / original_height
    width = width * target_width / original_width
    height = height * target_height / original_height
    return [x, y, width, height]

if __name__ == "__main__":
    app = QApplication(sys.argv)
    #app.setStyleSheet(stylesheet)
    widget = MainWindow()
    widget.show()
    sys.exit(app.exec())

