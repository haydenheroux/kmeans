import os
import sys
from PyQt6.QtWidgets import (
    QApplication,
    QPushButton,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QSlider,
    QComboBox,
    QFileDialog,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QMouseEvent, QPixmap
import cv2
import qdarktheme
from color import ColorSpace
from image import Image
from pipeline import PipelineConfig
from palette import load


def format_pixel_size(tick: int) -> str:
    return f"""Pixel Size: {2**tick}"""


def new_file_path_of(file_path: str) -> str:
    name, extension = os.path.splitext(file_path)
    return f"{name}-edit{extension}"


class PipelineRunner(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("kmeans - image editor")

        self.file_path = None
        self.output_image = None
        self.palettes = load("palettes.yaml")

        main_layout = QVBoxLayout()

        self.input_image = QLabel("Click to select image", self)
        self.input_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.input_image.setStyleSheet("border: 2px dashed gray; padding: 20px;")
        self.input_image.setFixedHeight(400)
        self.input_image.mousePressEvent = self.open_file_dialog
        main_layout.addWidget(self.input_image)

        pixel_layout = QHBoxLayout()
        pixel_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        default_pixel_size_tick = 0
        self.slider_label = QLabel(format_pixel_size(tick=default_pixel_size_tick))
        pixel_layout.addWidget(self.slider_label)
        self.pixel_size_slider = QSlider(Qt.Orientation.Horizontal)
        self.pixel_size_slider.setMinimum(0)
        self.pixel_size_slider.setMaximum(6)
        self.pixel_size_slider.setValue(default_pixel_size_tick)
        self.pixel_size_slider.setTickInterval(1)
        self.pixel_size_slider.setTickPosition(QSlider.TickPosition.TicksBothSides)
        self.pixel_size_slider.valueChanged.connect(self.run_pipeline)
        self.pixel_size_slider.valueChanged.connect(self.change_pixel_size)
        pixel_layout.addWidget(self.pixel_size_slider)
        main_layout.addLayout(pixel_layout)

        color_layout = QHBoxLayout()
        color_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        palettes = ["auto"] + list(self.palettes.keys())
        self.palette_dropdown = QComboBox()
        self.palette_dropdown.addItems(palettes)
        self.palette_dropdown.currentIndexChanged.connect(self.run_pipeline)
        self.palette_dropdown.currentIndexChanged.connect(self.change_palette_size)
        color_layout.addWidget(self.palette_dropdown)
        default_palette_size = 16
        self.size_label = QLabel("Palette Size: 16")
        color_layout.addWidget(self.size_label)
        self.palette_size_slider = QSlider(Qt.Orientation.Horizontal)
        self.palette_size_slider.setMinimum(2)
        self.palette_size_slider.setMaximum(64)
        self.palette_size_slider.setValue(default_palette_size)
        self.palette_size_slider.valueChanged.connect(self.run_pipeline)
        self.palette_size_slider.valueChanged.connect(self.change_palette_size)
        color_layout.addWidget(self.palette_size_slider)
        self.color_space_dropdown = QComboBox()
        self.color_space_dropdown.addItems(list(ColorSpace))
        self.color_space_dropdown.currentIndexChanged.connect(self.run_pipeline)
        color_layout.addWidget(self.color_space_dropdown)
        main_layout.addLayout(color_layout)

        self.output_image_viewer = QLabel(self)
        self.output_image_viewer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.output_image_viewer.setFixedHeight(400)
        main_layout.addWidget(self.output_image_viewer)

        self.save_button = QPushButton("&Save")
        self.save_button.clicked.connect(self.save_image)
        main_layout.addWidget(self.save_button)

        self.setLayout(main_layout)

    def open_file_dialog(self, ev: QMouseEvent | None = None):
        self.file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image File", "", "Images (*.png *.jpg *.jpeg *.bmp *.gif)"
        )
        if self.file_path:
            pixmap = QPixmap(self.file_path)
            self.input_image.setPixmap(
                pixmap.scaled(
                    self.input_image.width(),
                    self.input_image.height(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                )
            )
            self.run_pipeline()

    def change_palette_size(self):
        self.size_label.setText(f"Palette Size: {self.palette_size_slider.value()}")
        palette = self.palette_dropdown.currentText()
        if palette == "auto":
            self.palette_size_slider.setEnabled(True)
        else:
            pal = self.palettes[palette]
            self.palette_size_slider.setValue(pal.shape[0])
            self.palette_size_slider.setEnabled(False)

    def change_pixel_size(self):
        self.slider_label.setText(format_pixel_size(self.pixel_size_slider.value()))

    def run_pipeline(self):
        if not self.file_path:
            return
        pixel_size = 2 ** self.pixel_size_slider.value()
        image = Image.from_cv2(cv2.imread(self.file_path), pixel_size)

        pipeline = self.create_config().create_pipeline()
        if pipeline:
            self.output_image = pipeline.run(image).qimage()
            self.output_image_viewer.setPixmap(
                QPixmap.fromImage(self.output_image).scaled(
                    self.output_image_viewer.width(),
                    self.output_image_viewer.height(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                )
            )
            self.save_button.setText(f"&Save {new_file_path_of(self.file_path)}")

    def save_image(self):
        if not self.file_path or not self.output_image:
            return
        self.output_image.save(new_file_path_of(self.file_path))

    def create_config(self):
        config = PipelineConfig()

        palette = self.palette_dropdown.currentText()
        if palette == "auto":
            config.auto_generate_palette(self.palette_size_slider.value())
        else:
            config.use_palette(self.palettes[palette])
        color_space = self.color_space_dropdown.currentText()
        config.use_color_space(ColorSpace(color_space))

        return config


def main():
    app = QApplication(sys.argv)
    qdarktheme.setup_theme()
    window = PipelineRunner()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
