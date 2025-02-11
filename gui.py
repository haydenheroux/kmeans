import sys
from PyQt6.QtWidgets import (
    QApplication,
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


class ImageEditor(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("kmeans - image editor")

        self.file_path = None
        self.palettes = load("palettes.yaml")

        main_layout = QVBoxLayout()

        self.image_label = QLabel("Drop an image here or click to upload", self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("border: 2px dashed gray; padding: 20px;")
        self.image_label.setFixedHeight(400)
        self.image_label.mousePressEvent = self.open_file_dialog
        main_layout.addWidget(self.image_label)

        middle_layout = QHBoxLayout()
        middle_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        palettes = ["auto"] + list(self.palettes.keys())
        self.palette_dropdown = QComboBox()
        self.palette_dropdown.addItems(palettes)
        self.palette_dropdown.currentIndexChanged.connect(self.main)
        middle_layout.addWidget(self.palette_dropdown)

        default_palette_size = 16
        self.size_label = QLabel("Palette Size: 16")
        middle_layout.addWidget(self.size_label)
        self.palette_size_slider = QSlider(Qt.Orientation.Horizontal)
        self.palette_size_slider.setMinimum(2)
        self.palette_size_slider.setMaximum(64)
        self.palette_size_slider.setValue(default_palette_size)
        self.palette_size_slider.valueChanged.connect(self.main)
        middle_layout.addWidget(self.palette_size_slider)

        default_pixel_size_tick = 0
        self.slider_label = QLabel(format_pixel_size(tick=default_pixel_size_tick))
        middle_layout.addWidget(self.slider_label)
        self.pixel_size_slider = QSlider(Qt.Orientation.Horizontal)
        self.pixel_size_slider.setMinimum(0)
        self.pixel_size_slider.setMaximum(6)
        self.pixel_size_slider.setValue(default_pixel_size_tick)
        self.pixel_size_slider.setTickInterval(1)
        self.pixel_size_slider.setTickPosition(QSlider.TickPosition.TicksBothSides)
        self.pixel_size_slider.valueChanged.connect(self.main)
        middle_layout.addWidget(self.pixel_size_slider)

        color_spaces = list(ColorSpace)
        self.color_space_dropdown = QComboBox()
        self.color_space_dropdown.addItems(color_spaces)
        self.color_space_dropdown.currentIndexChanged.connect(self.main)
        middle_layout.addWidget(self.color_space_dropdown)

        main_layout.addLayout(middle_layout)

        self.display_label = QLabel(self)
        self.display_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.display_label.setFixedHeight(400)
        main_layout.addWidget(self.display_label)

        self.setLayout(main_layout)

    def open_file_dialog(self, ev: QMouseEvent | None = None):
        self.file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image File", "", "Images (*.png *.jpg *.jpeg *.bmp *.gif)"
        )
        if self.file_path:
            self.main()

    def change_palette_size(self):
        self.size_label.setText(f"Palette Size: {self.palette_size_slider.value()}")
        palette = self.palette_dropdown.currentText()
        if palette == "auto":
            self.palette_size_slider.setEnabled(True)
        else:
            pal = self.palettes[palette]
            self.palette_size_slider.setValue(pal.shape[0])
            self.palette_size_slider.setEnabled(False)

    def main(self):
        self.change_palette_size()
        self.slider_label.setText(format_pixel_size(self.pixel_size_slider.value()))

        if not self.file_path:
            return

        pixmap = QPixmap(self.file_path)
        self.image_label.setPixmap(
            pixmap.scaled(
                self.image_label.width(),
                self.image_label.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
            )
        )

        config = PipelineConfig()
        image = cv2.imread(self.file_path)
        pixel_size = 2 ** self.pixel_size_slider.value()
        palette = self.palette_dropdown.currentText()
        if palette == "auto":
            config.auto_generate_palette(self.palette_size_slider.value())
        else:
            config.use_palette(self.palettes[palette])
        color_space = self.color_space_dropdown.currentText()
        config.use_color_space(ColorSpace(color_space))
        pipeline = config.create_pipeline()
        if pipeline:
            new_image = pipeline.run(Image.from_cv2(image, pixel_size))
            display_pixmap = QPixmap.fromImage(new_image.qimage())
            self.display_label.setPixmap(
                display_pixmap.scaled(
                    self.display_label.width(),
                    self.display_label.height(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                )
            )


def main():
    app = QApplication(sys.argv)
    qdarktheme.setup_theme()
    window = ImageEditor()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
