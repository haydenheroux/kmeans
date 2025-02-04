import enum
import numpy as np
import scipy

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


class RGB:
    class Const:
        A = 12.92
        C = 0.055
        Γ = 2.4
        U = 0.04045
        V = 0.0031308

    @classmethod
    def from_hex(cls, hex: str) -> list[int] | None:
        """Converts a hex string to an RGB tuple."""
        if hex[0] == "#":
            hex = hex[1:]
        if len(hex) != 6:
            return None
        return [int(hex[i : i + 2], 16) for i in (0, 2, 4)]

    @classmethod
    def to_hex(cls, rgb) -> str:
        """Converts an RGB tuple to a hex string."""
        return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"

    @classmethod
    def to_linear(cls, z: np.typing.NDArray, M=255) -> np.typing.NDArray:
        """Converts RGB values to linear RGB values."""
        u = z / M
        linear = np.where(
            u <= cls.Const.U,
            u / cls.Const.A,
            ((u + cls.Const.C) / (1 + cls.Const.C)) ** cls.Const.Γ,
        )
        return linear

    @classmethod
    def from_linear(cls, v: np.typing.NDArray, M=255) -> np.typing.NDArray:
        """Converts linear RGB values to RGB values."""
        E = np.where(
            v <= cls.Const.V,
            cls.Const.A * v,
            (1 + cls.Const.C) * (v ** (1 / cls.Const.Γ)) - cls.Const.C,
        )
        return np.round(M * E).astype(np.uint8)


class Oklab:
    class Const:
        M_1 = np.matrix(
            [
                [0.8189330101, 0.3618667424, -0.1288597137],
                [0.0329845436, 0.9293118715, 0.0361456387],
                [0.0482003018, 0.2643662691, 0.6338517070],
            ]
        )
        M_2 = np.matrix(
            [
                [0.2104542553, 0.7936177850, -0.0040720468],
                [1.9779984951, -2.4285922050, 0.4505937099],
                [0.0259040371, 0.7827717662, -0.8086757660],
            ]
        )
        RGB_LIN_TO_LMS = np.matrix(
            [
                [0.4122214708, 0.5363325363, 0.0514459929],
                [0.2119034982, 0.6806995451, 0.1073969566],
                [0.0883024619, 0.2817188376, 0.6299787005],
            ]
        )

    @classmethod
    def linear_triplet_to_lab_triplet(cls, rgb: np.typing.NDArray) -> np.typing.NDArray:
        """Converts linear RGB values to Oklab values."""
        lms = np.einsum("ij,...j->...i", cls.Const.RGB_LIN_TO_LMS, rgb)
        lms = np.cbrt(lms)
        return np.einsum("ij,...j->...i", cls.Const.M_2, lms)

    @classmethod
    def lab_triplet_to_linear_triplet(cls, lab: np.typing.NDArray) -> np.typing.NDArray:
        """Converts Oklab values to linear RGB values."""
        lms = np.einsum("ij,...j->...i", cls.Const.M_2.I, lab)
        lms = lms**3
        return np.einsum("ij,...j->...i", cls.Const.RGB_LIN_TO_LMS.I, lms)
