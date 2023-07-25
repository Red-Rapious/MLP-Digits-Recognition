use crate::neural_network::NeuralNetwork;
use crate::mnist_parser::*;
use crate::utility::{sigmoid, sigmoid_prime};
use show_image::{ImageView, ImageInfo, create_window};

pub mod neural_network;
pub mod tests;
pub mod utility;
pub mod mnist_parser;
pub mod evaluation_result;

//#[show_image::main]
fn main() {
    let mut nn = NeuralNetwork::new(vec![28*28, 16, 16, 10]);
    let (train_images, train_labels, validation_images, validation_labels) = load_data(300, 20);
    let mut training_data = train_images.into_iter().zip(train_labels.into_iter()).collect();
    let mut validation_data = validation_images.into_iter().zip(validation_labels.into_iter()).collect();
    nn.train(&mut training_data, 30, 10, 1.0, &mut validation_data, &sigmoid, &sigmoid_prime);
}

fn _preview_images() {
    let (training_images, _, _, _) = load_data(5, 0);
    //let _result = neural_network.feed_forward(training_images[0].clone(), &sigmoid);
    //println!("result: {:?} \nlabel: {}", result, training_labels[0]);

    for training_image in training_images.iter().take(5) {
        let pixel_data: Vec<u8> = training_image.iter().map(|x| ((1.0-x)*256.0) as u8).collect();
        let image = ImageView::new(ImageInfo::mono8(28, 28), &pixel_data);
    
        // Create a window with default options and display the image.
        let window = create_window("image", Default::default()).unwrap();
        window.set_image("image-001", image).unwrap();
        window.wait_until_destroyed().unwrap();
    }
}