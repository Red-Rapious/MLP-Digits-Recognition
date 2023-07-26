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
    // Initialise network
    let mut nn = NeuralNetwork::new(vec![28*28, 16, 16, 10]);
    // Get the data
    let (train_images, train_labels, 
        validation_images, validation_labels, 
        test_images, test_labels) = 
        load_data(500, 20, 100);

    // Parse it
    let mut training_data: Vec<(Vec<f64>, u8)> = train_images.into_iter().zip(train_labels.into_iter()).collect();
    let mut validation_data: Vec<(Vec<f64>, u8)> = validation_images.into_iter().zip(validation_labels.into_iter()).collect();
    let testing_data: Vec<(Vec<f64>, u8)> = test_images.into_iter().zip(test_labels.into_iter()).collect();

    // Train the network
    nn.train(&mut training_data, 10, 30, 3.0, &mut validation_data, &sigmoid, &sigmoid_prime);

    println!("\n\n");
    // Test the network
    let result = nn.evaluate(&testing_data, &sigmoid);
    print!("{}\n", result);
}

fn _preview_images() {
    let (training_images, _, _, _, _, _) = load_data(5, 0, 0);
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