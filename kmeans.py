import numpy as np
import cv2
from numpy.typing import NDArray
import sklearn.cluster
import scipy.spatial
import sys


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
            cluster_index_per_pixel, centers = cluster_pixels(image, num_clusters=len(palette))
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


CATPPUCCIN_LATTE = np.array(
    [
        [220, 138, 120],  # ctp-rosewater
        [221, 120, 120],  # ctp-flamingo
        [234, 118, 203],  # ctp-pink
        [136, 57, 239],  # ctp-mauve
        [210, 15, 57],  # ctp-red
        [230, 69, 83],  # ctp-maroon
        [254, 100, 11],  # ctp-peach
        [223, 142, 29],  # ctp-yellow
        [64, 160, 43],  # ctp-green
        [23, 146, 153],  # ctp-teal
        [4, 165, 229],  # ctp-sky
        [32, 159, 181],  # ctp-sapphire
        [30, 102, 245],  # ctp-blue
        [114, 135, 253],  # ctp-lavender
        [76, 79, 105],  # ctp-text
        [92, 95, 119],  # ctp-subtext1
        [108, 111, 133],  # ctp-subtext0
        [124, 127, 147],  # ctp-overlay2
        [140, 143, 161],  # ctp-overlay1
        [156, 160, 176],  # ctp-overlay0
        [172, 176, 190],  # ctp-surface2
        [188, 192, 204],  # ctp-surface1
        [204, 208, 218],  # ctp-surface0
        [239, 241, 245],  # ctp-base
        [230, 233, 239],  # ctp-mantle
        [220, 224, 232],  # ctp-crust
        [234, 118, 203],  # hex-pink
        [136, 57, 239],  # hex-mauve
        [210, 15, 57],  # hex-red
        [254, 100, 11],  # hex-peach
        [223, 142, 29],  # hex-yellow
        [64, 160, 43],  # hex-green
        [23, 146, 153],  # hex-teal
        [4, 165, 229],  # hex-sky
        [32, 159, 181],  # hex-sapphire
        [30, 102, 245],  # hex-blue
        [114, 135, 253],  # hex-lavender
    ]
)

CATPPUCCIN_MACCHIATO = np.array(
    [
        [244, 219, 214],  # ctp-rosewater
        [240, 198, 198],  # ctp-flamingo
        [245, 189, 230],  # ctp-pink
        [198, 160, 246],  # ctp-mauve
        [237, 135, 150],  # ctp-red
        [238, 153, 160],  # ctp-maroon
        [245, 169, 127],  # ctp-peach
        [238, 212, 159],  # ctp-yellow
        [166, 218, 149],  # ctp-green
        [139, 213, 202],  # ctp-teal
        [145, 215, 227],  # ctp-sky
        [125, 196, 228],  # ctp-sapphire
        [138, 173, 244],  # ctp-blue
        [183, 189, 248],  # ctp-lavender
        [202, 211, 245],  # ctp-text
        [184, 192, 224],  # ctp-subtext1
        [165, 173, 203],  # ctp-subtext0
        [147, 154, 183],  # ctp-overlay2
        [128, 135, 162],  # ctp-overlay1
        [110, 115, 141],  # ctp-overlay0
        [91, 96, 120],  # ctp-surface2
        [73, 77, 100],  # ctp-surface1
        [54, 58, 79],  # ctp-surface0
        [36, 39, 58],  # ctp-base
        [30, 32, 48],  # ctp-mantle
        [24, 25, 38],  # ctp-crust
        [245, 189, 230],  # hex-pink
        [198, 160, 246],  # hex-mauve
        [237, 135, 150],  # hex-red
        [245, 169, 127],  # hex-peach
        [238, 212, 159],  # hex-yellow
        [166, 218, 149],  # hex-green
        [139, 213, 202],  # hex-teal
        [145, 215, 227],  # hex-sky
        [125, 196, 228],  # hex-sapphire
        [138, 173, 244],  # hex-blue
        [183, 189, 248],  # hex-lavender
    ]
)

if __name__ == "__main__":
    if len(sys.argv) == 1 or len(sys.argv) > 4:
        print(f"usage: {sys.argv[0]} <filename> <palette> <pixelsize>")
        sys.exit(1)
    filename = sys.argv[1]
    image = cv2.imread(filename)
    original_height, original_width = image.shape[:2]
    pixelated = False
    palette = 16
    if len(sys.argv) == 4:
        pixel_size = int(sys.argv[3])
        image = pixelate(image, pixel_size)
        pixelated = True
    if len(sys.argv) >= 3:
        palette_str = sys.argv[2].lower()
        if palette_str.isdigit():
            palette = int(palette_str)
        elif palette_str == "light" or palette_str == "latte":
            palette = CATPPUCCIN_LATTE
        elif palette_str == "dark" or palette_str == "macchiato":
            palette = CATPPUCCIN_MACCHIATO
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
