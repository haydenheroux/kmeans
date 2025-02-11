import dataclasses
import cv2
import numpy as np
import color
import typing
import clusterer

Mapper = typing.Callable[[np.typing.NDArray], np.typing.NDArray]

@dataclasses.dataclass
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

class Pipeline:
    color_mapper: Mapper
    space_mapper: Mapper
    space_unmapper: Mapper
    pixels: np.typing.NDArray
    pixels_shape: Shape
    result_shape: Shape
    clusterer_: clusterer.Clusterer

    def __init__(
        self,
        color_mapper: Mapper,
        space_mapper: Mapper,
        space_unmapper: Mapper,
        pixels: np.typing.NDArray,
        pixels_shape: Shape,
        scaled_shape: Shape,
        clusterer: clusterer.Clusterer,
    ):
        self.color_mapper = color_mapper
        self.space_mapper = space_mapper
        self.space_unmapper = space_unmapper
        self.pixels = pixels
        self.pixels_shape = pixels_shape
        self.result_shape = scaled_shape
        self.clusterer_ = clusterer

    def run(self) -> cv2.typing.MatLike:
        transformed_pixels = self.cluster_and_map()
        new_image = transformed_pixels.reshape(self.pixels_shape.numpy()).astype(np.uint8)
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
        if self.pixels_shape != self.result_shape:
            new_image = cv2.resize(
                new_image,
                self.result_shape.cv2(),
                interpolation=cv2.INTER_NEAREST,
            )
        return new_image

    def cluster_and_map(
        self,
    ) -> np.typing.NDArray:
        """
        Clusters the pixels into then maps each cluster to a color.
        """
        mapped_pixels = self.space_mapper(self.pixels)
        centers = self.clusterer_.fit(mapped_pixels)
        labels = self.clusterer_.predict(mapped_pixels)
        centers = np.array(centers)
        # TODO Replace with a NumPy-friendly operation
        cluster_index_to_palette_color = np.array(
            [self.color_mapper(center) for center in centers]
        )
        new_pixels = cluster_index_to_palette_color[labels]
        return self.space_unmapper(new_pixels)


class KMeansAppConfig:
    original_shape: Shape | None
    scaled_shape: Shape | None
    pixels: np.typing.NDArray | None
    palette: np.typing.NDArray | None
    num_clusters: int
    pixel_size: int
    color_space: color.ColorSpace

    def __init__(self):
        self.original_shape = None
        self.pixels = None
        self.palette = None
        self.num_clusters = 16
        self.pixel_size = 1
        self.color_space = color.ColorSpace.RGB

    def use_image(self, image: cv2.typing.MatLike, pixel_size: int = 1):
        self.original_shape = Shape.from_numpy(image.shape)
        # TODO Add a check for pixel_size being equal to 1
        image = cv2.resize(
            image,
            (0, 0),
            fx=1 / pixel_size,
            fy=1 / pixel_size,
            interpolation=cv2.INTER_AREA,
        )
        self.scaled_shape = Shape.from_numpy(image.shape)
        self.pixels = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).reshape(-1, 3)
        self.pixel_size = pixel_size

    def use_palette(self, palette: np.typing.NDArray):
        self.palette = palette
        self.num_clusters = len(palette)

    def auto_generate_palette(self, num_clusters: int):
        self.palette = None
        self.num_clusters = num_clusters

    def use_color_space(self, color_space: color.ColorSpace):
        self.color_space = color_space

    def create_color_mapper(self) -> Mapper:
        if self.palette is None:
            return lambda colors: colors
        return lambda colors: color.closest_color(
            colors, self.palette, self.create_space_mapper
        )

    def create_space_mapper(self) -> Mapper:
        match self.color_space:
            case color.ColorSpace.RGB:
                return lambda colors: colors
            case color.ColorSpace.LINEAR_RGB:
                return color.RGB.to_linear
            case color.ColorSpace.OKLAB:
                return lambda colors: color.Oklab.linear_triplet_to_lab_triplet(
                    color.RGB.to_linear(colors)
                )

    def create_space_unmapper(self) -> Mapper:
        match self.color_space:
            case color.ColorSpace.RGB:
                return lambda colors: colors
            case color.ColorSpace.LINEAR_RGB:
                return color.RGB.from_linear
            case color.ColorSpace.OKLAB:
                return lambda colors: color.RGB.from_linear(
                    color.Oklab.lab_triplet_to_linear_triplet(colors)
                )

    def create_pipeline(self) -> Pipeline | None:
        if self.pixels is None or self.original_shape is None or self.scaled_shape is None:
            return None

        return Pipeline(
            self.create_color_mapper(),
            self.create_space_mapper(),
            self.create_space_unmapper(),
            self.pixels,
            self.scaled_shape,
            self.original_shape,
            clusterer.MiniBatchKMeans(self.num_clusters),
        )
