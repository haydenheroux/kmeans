import numpy as np
import typing
from clusterer import Clusterer, MiniBatchKMeans
from color import ColorSpace, closest_color, RGB, Oklab
from image import Image

Mapper = typing.Callable[[np.typing.NDArray], np.typing.NDArray]


class Pipeline:
    color_mapper: Mapper
    space_mapper: Mapper
    space_unmapper: Mapper
    noiser: Mapper
    clusterer: Clusterer

    def __init__(
        self,
        color_mapper: Mapper,
        space_mapper: Mapper,
        space_unmapper: Mapper,
        noiser: Mapper,
        clusterer: Clusterer,
    ):
        self.color_mapper = color_mapper
        self.space_mapper = space_mapper
        self.space_unmapper = space_unmapper
        self.noiser = noiser
        self.clusterer = clusterer

    def run(self, image: Image) -> Image:
        mapped_pixels = self.space_mapper(image.pixels)
        transformed_pixels = self.cluster_and_map(mapped_pixels)
        noised_pixels = self.noiser(transformed_pixels)
        pixels = self.space_unmapper(noised_pixels)
        return Image(
            pixels=pixels,
            pixels_shape=image.pixels_shape,
            image_shape=image.image_shape,
        )

    def cluster_and_map(self, pixels: np.typing.NDArray) -> np.typing.NDArray:
        """
        Clusters the pixels then maps each cluster to a color.
        """
        centers = self.clusterer.fit(pixels)
        labels = self.clusterer.predict(pixels)
        centers = np.array(centers)
        # TODO Replace with a NumPy-friendly operation
        cluster_index_to_palette_color = np.array(
            [self.color_mapper(center) for center in centers]
        )
        new_pixels = cluster_index_to_palette_color[labels]
        return new_pixels


class PipelineConfig:
    palette: np.typing.NDArray | None
    num_clusters: int
    pixel_size: int
    color_space: ColorSpace
    noise_std: float

    def __init__(self):
        self.palette = None
        self.num_clusters = 16
        self.pixel_size = 1
        self.color_space = ColorSpace.RGB
        self.noise_std = 1

    def use_palette(self, palette: np.typing.NDArray):
        self.palette = palette
        self.num_clusters = len(palette)

    def auto_generate_palette(self, num_clusters: int):
        self.palette = None
        self.num_clusters = num_clusters

    def use_color_space(self, color_space: ColorSpace):
        self.color_space = color_space

    def use_noise_size(self, noise_size: float):
        self.noise_std = noise_size

    def create_color_mapper(self) -> Mapper:
        if self.palette is None:
            return lambda colors: colors
        return lambda colors: closest_color(
            colors, self.palette, self.create_space_mapper
        )

    def create_space_mapper(self) -> Mapper:
        match self.color_space:
            case ColorSpace.RGB:
                return lambda colors: colors
            case ColorSpace.LINEAR_RGB:
                return RGB.to_linear
            case ColorSpace.OKLAB:
                return lambda colors: Oklab.linear_triplet_to_lab_triplet(
                    RGB.to_linear(colors)
                )

    def create_space_unmapper(self) -> Mapper:
        match self.color_space:
            case ColorSpace.RGB:
                return lambda colors: colors
            case ColorSpace.LINEAR_RGB:
                return RGB.from_linear
            case ColorSpace.OKLAB:
                return lambda colors: RGB.from_linear(
                    Oklab.lab_triplet_to_linear_triplet(colors)
                )

    def create_noiser(self) -> Mapper:
        match self.color_space:
            case ColorSpace.RGB:
                return lambda colors: np.clip(
                    colors + np.random.normal(0, self.noise_std * 255, colors.shape),
                    0,
                    255,
                )
            case ColorSpace.LINEAR_RGB:
                return lambda colors: np.clip(
                    colors + np.random.normal(0, self.noise_std, colors.shape),
                    0,
                    1,
                )
            case ColorSpace.OKLAB:

                def add_oklab_noise(colors):
                    # NOTE In OKLAB, L-channel noise is twice as noticeable
                    noise_std_scale = 0.5
                    noisy = colors.copy()
                    L_noise = np.random.normal(0, self.noise_std * noise_std_scale, noisy[..., 0].shape)
                    noisy[..., 0] = np.clip(noisy[..., 0] + L_noise, 0, 1)
                    return noisy

                return add_oklab_noise

    def create_pipeline(self) -> Pipeline:
        return Pipeline(
            color_mapper=self.create_color_mapper(),
            space_mapper=self.create_space_mapper(),
            space_unmapper=self.create_space_unmapper(),
            noiser=self.create_noiser(),
            clusterer=MiniBatchKMeans(self.num_clusters),
        )
