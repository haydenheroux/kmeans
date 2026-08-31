use image::{DynamicImage, Rgb, RgbImage};
use kmeans_uni::KMeansBuilder;

fn kmeans(img: &DynamicImage) -> Result<DynamicImage, kmeans_uni::Error> {
    // TODO(hayden): Make generic; support arbitrary dimension, formats, etc.
    const RGB_DIMENSION: usize = 3;
    let pixels = get_rgb8_pixels(&img);
    let mut result = create_empty_rgb8(&img.to_rgb8());

    const K: usize = 8;
    const MAX_ITER: usize = 30;
    let pipeline = KMeansBuilder::new(K).iterations(MAX_ITER).cpu_simd().euclidean().build();

    let model = pipeline.fit(&pixels, RGB_DIMENSION)?;
    let labels = model.predict(&pixels)?;
    write_centroids(labels, model.centroids(), &mut result);

    Ok(image::DynamicImage::ImageRgb8(result))
}
fn get_rgb8_pixels(img: &DynamicImage) -> Vec<f32> {
    let rgb8 = img.to_rgb8();
    let (width, height) = rgb8.dimensions();
    let count = (width * height) as usize;
    const RGB_DIMENSION: usize = 3;
    let mut buffer: Vec<f32> = Vec::with_capacity(count * RGB_DIMENSION);
    for pixel in rgb8.pixels() {
        buffer.push(pixel[0] as f32);
        buffer.push(pixel[1] as f32);
        buffer.push(pixel[2] as f32);
    }
    buffer
}

fn create_empty_rgb8(img: &RgbImage) -> RgbImage {
    let (width, height) = img.dimensions();
    RgbImage::new(width, height)
}

fn get_centroid_rgb(centroids: &[f32], centroid_label: usize) -> Rgb<u8> {
    let base_idx = centroid_label * 3;
    // TODO(hayden): `cannot index into a value of type &[f32]`
    let r = centroids[base_idx + 0] as u8;
    let g = centroids[base_idx + 1] as u8;
    let b = centroids[base_idx + 2] as u8;
    Rgb([r, g, b])
}

fn write_centroids(labels: Vec<usize>, centroids: &[f32], result: &mut RgbImage) {
    let width = result.width();

    for (idx, &centroid_label) in labels.iter().enumerate() {
        let x = (idx as u32) % width;
        let y = (idx as u32) / width;
        let rgb = get_centroid_rgb(centroids, centroid_label);
        result.put_pixel(x, y, rgb);
    }
}

fn main() {
    let img: DynamicImage = image::open("input.png").expect("Reading must succeed");
    let result = kmeans(&img).expect("Kmeans must succeed");
    result.save("quantized_output.png").expect("Saving must succeed")
}
