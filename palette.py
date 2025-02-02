import yaml
import numpy as np


def rgb_to_hex(rgb):
    """Converts an RGB tuple to a hex string."""
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def hex_to_rgb(hex) -> list[int] | None:
    """Converts a hex string to an RGB tuple."""
    if hex[0] == "#":
        hex = hex[1:]
    if len(hex) != 6:
        return None
    return [int(hex[i : i + 2], 16) for i in (0, 2, 4)]


def load(file_path):
    """Imports palettes from a YAML file."""
    with open(file_path, "r") as file:
        hex_palettes = yaml.load(file, Loader=yaml.FullLoader)

    # Convert hex palettes to RGB palettes
    palettes = {
        name: np.array([hex_to_rgb(hex_color) for hex_color in hex_palette])
        for name, hex_palette in hex_palettes.items()
    }

    return palettes


def save(file_path, palettes):
    """Exports palettes to a YAML file."""
    # Convert RGB palettes to hex palettes
    hex_palettes = {
        name: [rgb_to_hex(rgb) for rgb in palette] for name, palette in palettes.items()
    }

    with open(file_path, "w") as file:
        yaml.dump(hex_palettes, file)
