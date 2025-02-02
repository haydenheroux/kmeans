import yaml
import numpy as np
import color


def load(file_path):
    """Imports palettes from a YAML file."""
    with open(file_path, "r") as file:
        hex_palettes = yaml.load(file, Loader=yaml.FullLoader)

    palettes = {
        name: np.array([color.RGB.from_hex(hex_color) for hex_color in hex_palette])
        for name, hex_palette in hex_palettes.items()
    }

    return palettes


def save(file_path, palettes):
    """Exports palettes to a YAML file."""
    hex_palettes = {
        name: [color.RGB.to_hex(rgb) for rgb in palette]
        for name, palette in palettes.items()
    }

    with open(file_path, "w") as file:
        yaml.dump(hex_palettes, file)
