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
from PyQt6.QtGui import QMouseEvent, QPixmap, QImage
import cv2
import kmeans
import palette

def format_pixel_size(tick: int) -> str:
    return f"""Pixel Size: {2**tick}"""

class ImageEditor(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("kmeans - image editor")

        self.palettes = palette.load("palettes.yaml")

        main_layout = QVBoxLayout()

        self.image_label = QLabel("Drop an image here or click to upload", self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("border: 2px dashed gray; padding: 20px;")
        self.image_label.setFixedHeight(400)
        self.image_label.mousePressEvent = self.open_file_dialog
        main_layout.addWidget(self.image_label)

        middle_layout = QHBoxLayout()

        palettes = ["auto"] + list(self.palettes.keys())
        self.palette_dropdown = QComboBox()
        self.palette_dropdown.addItems(palettes)
        self.palette_dropdown.currentIndexChanged.connect(self.process_image)
        middle_layout.addWidget(self.palette_dropdown)

        default_pixel_size_tick = 0
        self.slider_label = QLabel(format_pixel_size(tick=default_pixel_size_tick))
        middle_layout.addWidget(self.slider_label)
        self.pixel_size_slider = QSlider(Qt.Orientation.Horizontal)
        self.pixel_size_slider.setMinimum(0)
        self.pixel_size_slider.setMaximum(6)
        self.pixel_size_slider.setValue(default_pixel_size_tick)
        self.pixel_size_slider.setTickInterval(1)
        self.pixel_size_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.pixel_size_slider.valueChanged.connect(self.update_slider_label)
        self.pixel_size_slider.valueChanged.connect(self.process_image)
        middle_layout.addWidget(self.pixel_size_slider)

        color_spaces = list(kmeans.ColorSpace)
        self.color_space_dropdown = QComboBox()
        self.color_space_dropdown.addItems(color_spaces)
        self.color_space_dropdown.currentIndexChanged.connect(self.process_image)
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
            self.process_image()

    def process_image(self):
        if not self.file_path:
            return

        pixmap = QPixmap(self.file_path)
        self.image_label.setPixmap(
            pixmap.scaled(
                self.image_label.width(), self.image_label.height(), Qt.AspectRatioMode.KeepAspectRatio
            )
        )

        app = kmeans.KMeansApp()
        image = cv2.imread(self.file_path)
        pixel_size = 2 ** self.pixel_size_slider.value()
        app.use_image(image, pixel_size)
        palette = self.palette_dropdown.currentText()
        if palette == "auto":
            # TODO Add number of clusters
            app.auto_generate_palette(16)
        else:
            app.use_palette(self.palettes[palette])
        color_space = self.color_space_dropdown.currentText()
        app.use_color_space(kmeans.ColorSpace(color_space))
        new_image = app.process()
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
        height, width, channels = new_image.shape
        # FIXME
        qimage = QImage(new_image.data, width, height, channels * width, QImage.Format.Format_RGB888)
        display_pixmap = QPixmap.fromImage(qimage)

        self.display_label.setPixmap(
            display_pixmap.scaled(
                self.display_label.width(),
                self.display_label.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
            )
        )

    def update_slider_label(self, tick):
        self.slider_label.setText(format_pixel_size(tick))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageEditor()
    window.show()
    sys.exit(app.exec())
