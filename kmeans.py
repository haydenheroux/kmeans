import enum
from typing import Callable
import numpy as np
import cv2
from numpy.typing import NDArray
import sklearn.cluster
import scipy.spatial
import sys
import palette as palettes
import color
import enum


def closest_color(color, palette, space_mapper_getter):
    """
    Returns the color in the palette that is closest to the color.
    """
    space_mapper = space_mapper_getter()
    palette = space_mapper(palette)
    distances = [
        scipy.spatial.distance.euclidean(color, palette_color)
        for palette_color in palette
    ]
    return palette[np.argmin(distances)]


class ColorSpace(enum.StrEnum):
    RGB = "rgb"
    LINEAR_RGB = "linear"
    OKLAB = "oklab"

    def __contains__(self, key: str) -> bool:
        try:
            ColorSpace(key)
        except ValueError:
            return False
        return True


class KMeansApp:
    dimensions: cv2.typing.Size | None
    scaled_shape: cv2.typing.Size | None
    pixels: NDArray | None
    palette: NDArray | None
    num_clusters: int
    pixel_size: int
    color_space: ColorSpace

    def __init__(self):
        self.dimensions = None
        self.pixels = None
        self.palette = None
        self.num_clusters = 16
        self.pixel_size = 1
        self.color_space = ColorSpace.RGB

    def use_image(self, image: cv2.typing.MatLike, pixel_size: int = 1):
        height, width = image.shape[:2]
        self.dimensions = (width, height)
        # TODO Add a check for pixel_size being equal to 1
        image = cv2.resize(
            image,
            (0, 0),
            fx=1 / pixel_size,
            fy=1 / pixel_size,
            interpolation=cv2.INTER_AREA,
        )
        self.scaled_shape = image.shape
        self.pixels = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).reshape(-1, 3)
        self.pixel_size = pixel_size

    def use_palette(self, palette: NDArray):
        self.palette = palette
        self.num_clusters = len(palette)

    def auto_generate_palette(self, num_clusters: int):
        self.palette = None
        self.num_clusters = num_clusters

    def use_color_space(self, color_space: ColorSpace):
        self.color_space = color_space

    def color_mapper(self) -> Callable[[NDArray], NDArray]:
        if self.palette is None:
            return lambda color: color
        return lambda color: closest_color(color, self.palette, self.space_mapper)

    def space_mapper(self) -> Callable[[NDArray], NDArray]:
        match self.color_space:
            case ColorSpace.RGB:
                return lambda colors: colors
            case ColorSpace.LINEAR_RGB:
                return color.RGB.to_linear
            case ColorSpace.OKLAB:
                return lambda colors: color.Oklab.linear_triplet_to_lab_triplet(
                    color.RGB.to_linear(colors)
                )

    def space_unmapper(self) -> Callable[[NDArray], NDArray]:
        match self.color_space:
            case ColorSpace.RGB:
                return lambda colors: colors
            case ColorSpace.LINEAR_RGB:
                return color.RGB.from_linear
            case ColorSpace.OKLAB:
                return lambda colors: color.RGB.from_linear(
                    color.Oklab.lab_triplet_to_linear_triplet(colors)
                )

    def process(self) -> cv2.typing.MatLike:
        transformed_pixels = self.cluster_and_map(
            self.color_mapper(), self.space_mapper(), self.space_unmapper()
        )
        new_image = transformed_pixels.reshape(self.scaled_shape).astype(np.uint8)
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
        if self.pixel_size != 1:
            new_image = cv2.resize(
                new_image,
                self.dimensions,
                interpolation=cv2.INTER_NEAREST,
            )
        return new_image

    def cluster_and_map(
        self,
        color_mapper: Callable[[NDArray], NDArray],
        space_mapper: Callable[[NDArray], NDArray],
        space_unmapper: Callable[[NDArray], NDArray],
    ) -> NDArray:
        """
        Clusters the pixels into then maps each cluster to a color.
        """
        # TODO Ensure that pixels are set
        if self.pixels is None:
            return np.array([])
        pixels = space_mapper(self.pixels)
        cluster_index_per_pixel, centers = self.cluster_pixels(pixels)
        centers = np.array(centers)
        # TODO Replace with a NumPy-friendly operation
        cluster_index_to_palette_color = np.array(
            [color_mapper(center) for center in centers]
        )
        new_pixels = cluster_index_to_palette_color[cluster_index_per_pixel]
        return space_unmapper(new_pixels)

    def create_kmeans(self, mini_batch=True):
        if mini_batch:
            return sklearn.cluster.MiniBatchKMeans(n_clusters=self.num_clusters)
        return sklearn.cluster.KMeans(n_clusters=self.num_clusters)

    def cluster_pixels(self, pixels):
        """
        Clusters the pixels into k pixels.
        Returns the cluster index per pixel, and the cluster centers.
        """
        kmeans = self.create_kmeans()
        cluster_index_per_pixel = kmeans.fit_predict(pixels)
        return cluster_index_per_pixel, kmeans.cluster_centers_


if __name__ == "__main__":
    if len(sys.argv) == 1 or len(sys.argv) > 5:
        print(f"usage: {sys.argv[0]} <filename> <palette> <pixelsize> <space>")
        sys.exit(1)
    filename = sys.argv[1]
    image = cv2.imread(filename)
    app = KMeansApp()
    app.use_image(image)
    all_palettes = palettes.load("palettes.yaml")
    if len(sys.argv) >= 5:
        if (color_space := sys.argv[4].lower()) in ColorSpace:
            app.use_color_space(ColorSpace(color_space))
    if len(sys.argv) >= 4:
        if sys.argv[3].isdigit() and (pixel_size := int(sys.argv[3])) != 1:
            app.use_image(image, pixel_size)
    if len(sys.argv) >= 3:
        arg = sys.argv[2].lower()
        if arg in all_palettes:
            app.use_palette(all_palettes[arg])
        elif arg.isdigit():
            app.auto_generate_palette(int(arg))
    new_image = app.process()
    cv2.imshow("img", new_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
