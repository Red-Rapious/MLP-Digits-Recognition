use crate::neural_network::NeuralNetwork;
use crate::mnist_parser::*;
use crate::utility::sigmoid;
use show_image::{ImageView, ImageInfo, create_window};

pub mod neural_network;
pub mod tests;
pub mod utility;
pub mod mnist_parser;

#[show_image::main]
fn main() {
    let neural_network = NeuralNetwork::new(vec![28*28, 16, 16, 10]);
    let (training_images, training_labels, _, _) = load_data(5, 0);
    let result = neural_network.feed_forward(training_images[0].clone(), &sigmoid);
    //println!("result: {:?} \nlabel: {}", result, training_labels[0]);

    for i in 0..5 {
        let pixel_data: Vec<u8> = training_images[i].iter().map(|x| ((1.0-x)*256.0) as u8).collect();
        let image = ImageView::new(ImageInfo::mono8(28, 28), &pixel_data);
    
        // Create a window with default options and display the image.
        let window = create_window("image", Default::default()).unwrap();
        window.set_image("image-001", image).unwrap();
        window.wait_until_destroyed().unwrap();
    }
}