import cv2
import sys
import palette
import color
import kmeans

def main():
    if len(sys.argv) == 1 or len(sys.argv) > 5:
        print(f"usage: {sys.argv[0]} <filename> <palette> <pixelsize> <space>")
        sys.exit(1)
    filename = sys.argv[1]
    image = cv2.imread(filename)
    config = kmeans.KMeansAppConfig()
    config.use_image(image)
    all_palettes = palette.load("palettes.yaml")
    if len(sys.argv) >= 5:
        if (color_space := sys.argv[4].lower()) in color.ColorSpace:
            config.use_color_space(color.ColorSpace(color_space))
    if len(sys.argv) >= 4:
        if sys.argv[3].isdigit() and (pixel_size := int(sys.argv[3])) != 1:
            config.use_image(image, pixel_size)
    if len(sys.argv) >= 3:
        arg = sys.argv[2].lower()
        if arg in all_palettes:
            config.use_palette(all_palettes[arg])
        elif arg.isdigit():
            config.auto_generate_palette(int(arg))
    pipeline = config.create_pipeline()
    if pipeline:
        new_image = pipeline.run()
        cv2.imshow("img", new_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
