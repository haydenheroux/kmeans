from dataclasses import dataclass
from PyQt6.QtGui import QImage
import cv2
import numpy as np

@dataclass
class Shape:
    width: int
    height: int
    channels: int

    @classmethod
    def from_cv2(cls, size: cv2.typing.Size, channels: int = 3):
        return cls(width=size[0], height=size[1], channels=channels)

    @classmethod
    def from_numpy(cls, shape):
        return cls(width=shape[1], height=shape[0], channels=shape[2])

    def cv2(self) -> cv2.typing.Size:
        return (self.width, self.height)

    def numpy(self) -> tuple[int, int, int]:
        return (self.height, self.width, self.channels)

    def __str__(self) -> str:
        return f"{self.width=} {self.height=} {self.channels=}"


@dataclass
class Image:
    pixels: np.typing.NDArray
    pixels_shape: Shape
    image_shape: Shape

    @classmethod
    def from_cv2(cls, image: cv2.typing.MatLike, downscale_factor: int = 1):
        original_image_shape = image.shape
        if downscale_factor != 1:
            image = cv2.resize(
                image,
                (0, 0),
                fx=1 / downscale_factor,
                fy=1 / downscale_factor,
                interpolation=cv2.INTER_AREA,
            )
        pixels = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).reshape(-1, 3)
        return cls(
            pixels=pixels,
            pixels_shape=Shape.from_numpy(image.shape),
            image_shape=Shape.from_numpy(original_image_shape),
        )

    def cv2(self) -> cv2.typing.MatLike:
        new_image = self.pixels.reshape(self.pixels_shape.numpy()).astype(np.uint8)
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
        if self.pixels_shape != self.image_shape:
            new_image = cv2.resize(
                new_image,
                self.image_shape.cv2(),
                interpolation=cv2.INTER_NEAREST,
            )
        return new_image

    def qimage(self) -> QImage:
        # FIXME
        return QImage(
            self.pixels.data, width, height, channels * width, QImage.Format.Format_RGB888
        )


