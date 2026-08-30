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
