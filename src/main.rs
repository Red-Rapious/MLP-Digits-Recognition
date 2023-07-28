use crate::neural_network::NeuralNetwork;
use crate::mnist_parser::*;
use crate::activation_function::ActivationFunction;
//use show_image::{ImageView, ImageInfo, create_window};
use std::time::Instant;

pub mod neural_network;
pub mod tests;
pub mod utility;
pub mod mnist_parser;
pub mod evaluation_result;
pub mod activation_function;

const TRAIN_LENGTH: u32 = 1_000;
const VALIDATION_LENGTH: u32 = 0;
const TEST_LENGTH: u32 = 100;

const BATCH_SIZE: usize = 10;
const EPOCHS: usize = 10;
const LEARNING_RATE: f64 = 3.0;

//#[show_image::main]
fn main() {
    let program_begining = Instant::now();
    println!("[INFO] Begining of the program.");
    println!("\n[PARAMETERS]\n  - Training length: {} ; Validation length: {} ; Test length: {}", TRAIN_LENGTH, VALIDATION_LENGTH, TEST_LENGTH);
    println!("  - Batch size: {}", BATCH_SIZE);
    println!("  - Epochs: {}", EPOCHS);
    println!("  - Learning rate: {}", LEARNING_RATE);
    println!("  - Activation: {}", "sigmoid"); // TODO: change

    // Initialise network
    print!("\nInitialising neural network... ");
    let now = Instant::now();
    let mut nn = NeuralNetwork::new(vec![28*28, 16, 16, 10], ActivationFunction::sigmoid());
    println!("done in {:.2?}.", now.elapsed());

    // Get the data
    print!("Loading training, validation, and test data... ");
    let now = Instant::now();
    let (mut training_data, 
        mut validation_data, 
        testing_data) = load_data(TRAIN_LENGTH, VALIDATION_LENGTH, TEST_LENGTH);
    println!("done in {:.2?}.", now.elapsed());
        
    // Train the network
    println!("\n[INFO] Starting to train the network.");
    let now = Instant::now();
    nn.train(&mut training_data, BATCH_SIZE, EPOCHS, LEARNING_RATE, &mut validation_data);
    println!("[INFO] Network trained. Total training time: {:.2?}", now.elapsed());

    print!("\n\nEvaluating the network... ");
    let now = Instant::now();
    // Test the network
    let result = nn.evaluate(&testing_data);
    println!("done in {:.2?}.", now.elapsed());
    print!("{}\n", result);

    println!("\n[INFO] End of program. Total execution time: {:.2?}.", program_begining.elapsed());
}

/*fn _preview_images() {
    let (training_data, _, _) = load_data(5, 0, 0);
    //let _result = neural_network.feed_forward(training_images[0].clone(), &sigmoid);
    //println!("result: {:?} \nlabel: {}", result, training_labels[0]);

    for (training_image, _) in training_data.iter().take(5) {
        let pixel_data: Vec<u8> = training_image.iter().map(|x| ((1.0-x)*256.0) as u8).collect();
        let image = ImageView::new(ImageInfo::mono8(28, 28), &pixel_data);
    
        // Create a window with default options and display the image.
        let window = create_window("image", Default::default()).unwrap();
        window.set_image("image-001", image).unwrap();
        window.wait_until_destroyed().unwrap();
    }
}*/