import numpy as np
import cv2
from numpy.typing import NDArray
import sklearn.cluster
import scipy.spatial
import sys
import palette as palettes


def pixelate(image, pixel_size):
    """
    Returns the image pixelated down to a lower resolution.
    """
    return cv2.resize(
        image,
        (0, 0),
        fx=1 / pixel_size,
        fy=1 / pixel_size,
        interpolation=cv2.INTER_AREA,
    )


def closest_color(color, palette):
    """
    Returns the color in the palette that is closest to the color.
    """
    distances = [
        scipy.spatial.distance.euclidean(color, palette_color)
        for palette_color in palette
    ]
    return palette[np.argmin(distances)]


def create_kmeans(num_clusters, mini_batch=True):
    if mini_batch:
        return sklearn.cluster.MiniBatchKMeans(n_clusters=num_clusters)
    return sklearn.cluster.KMeans(n_clusters=num_clusters)


def cluster_pixels(image, num_clusters):
    """
    Clusters the pixels into k pixels.
    Returns the cluster index per pixel, and the cluster centers.
    """
    rgb_pixels = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).reshape(-1, 3)
    kmeans = create_kmeans(num_clusters)
    cluster_index_per_pixel = kmeans.fit_predict(rgb_pixels)
    return cluster_index_per_pixel, kmeans.cluster_centers_


def apply_palette(image, palette: int | NDArray):
    """
    Converts an image to the palette.
    """
    cluster_index_per_pixel = None
    centers = None
    cluster_index_to_palette_color = None
    if isinstance(palette, int):
        cluster_index_per_pixel, centers = cluster_pixels(image, num_clusters=palette)
        cluster_index_to_palette_color = np.array(centers)
    elif isinstance(palette, np.ndarray):
        cluster_index_per_pixel, centers = cluster_pixels(
            image, num_clusters=len(palette)
        )
        cluster_index_to_palette_color = np.array(
            [closest_color(center, palette) for center in centers]
        )
    new_image = (
        cluster_index_to_palette_color[cluster_index_per_pixel]
        .reshape(image.shape)
        .astype(np.uint8)
    )
    new_image_bgr = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    return new_image_bgr


if __name__ == "__main__":
    if len(sys.argv) == 1 or len(sys.argv) > 4:
        print(f"usage: {sys.argv[0]} <filename> <palette> <pixelsize>")
        sys.exit(1)
    filename = sys.argv[1]
    image = cv2.imread(filename)
    original_height, original_width = image.shape[:2]
    pixelated = False
    palette = 16
    all_palettes = palettes.load("palettes.yaml")
    if len(sys.argv) == 4:
        pixel_size = int(sys.argv[3])
        image = pixelate(image, pixel_size)
        pixelated = True
    if len(sys.argv) >= 3:
        palette_str = sys.argv[2].lower()
        if palette_str.isdigit():
            palette = int(palette_str)
        elif palette_str in all_palettes:
            palette = all_palettes[palette_str]
    transformed_image = apply_palette(image, palette)
    if pixelated:
        transformed_image = cv2.resize(
            transformed_image,
            (original_width, original_height),
            interpolation=cv2.INTER_NEAREST,
        )
    cv2.imshow("img", transformed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
