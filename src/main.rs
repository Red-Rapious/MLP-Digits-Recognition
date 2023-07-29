use activation_function::ActivationFunctionType;

use crate::neural_network::NeuralNetwork;
use crate::mnist_parser::*;
use crate::activation_function::ActivationFunction;
//use show_image::{ImageView, ImageInfo, create_window};
use std::time::Instant;
use std::io::Write;

pub mod neural_network;
pub mod tests;
pub mod utility;
pub mod mnist_parser;
pub mod evaluation_result;
pub mod activation_function;

// If `LOAD_NETWORK` is `true`, the network will be loaded from `FILE_NAME.json`
// Otherwise, the network will be trained using the parameters specified below
const LOAD_NETWORK: bool = false;
const FILE_NAME: &str = "src/saves/network"; // used both for saving and loading

// Parameters used in case of the training of the network
const LAYERS: &'static [usize] = &[28*28, 16, 16, 10];

const TRAIN_LENGTH: u32 = 1_000;
const VALIDATION_LENGTH: u32 = 100;
const TEST_LENGTH: u32 = 100;

const BATCH_SIZE: usize = 10;
const EPOCHS: usize = 3;
const LEARNING_RATE: f64 = 3.0;
const ACTIVATION_FUNCTION: ActivationFunctionType = ActivationFunctionType::Sigmoid;

//#[show_image::main]
fn main() {
    let program_begining = Instant::now();
    println!("[INFO] Begining of the program.");

    // Get the data
    print!("Loading training, validation, and test data... ");
    std::io::stdout().flush().unwrap();
    let now = Instant::now();
    let (mut training_data, 
        mut validation_data, 
        testing_data) = load_data(TRAIN_LENGTH, VALIDATION_LENGTH, TEST_LENGTH);
    println!("done in {:.2?}.", now.elapsed());


    let mut neural_network: NeuralNetwork;
    if LOAD_NETWORK {
        print!("Loading neural network from {}.json...", FILE_NAME);
        let now = Instant::now();
        neural_network = NeuralNetwork::load_network(FILE_NAME);
        println!("done in {:.2?}.", now.elapsed());
    } 
    else {
        println!("\n[PARAMETERS]\n  - Training length: {} ; Validation length: {} ; Test length: {}", TRAIN_LENGTH, VALIDATION_LENGTH, TEST_LENGTH);
        println!("  - Batch size: {}", BATCH_SIZE);
        println!("  - Epochs: {}", EPOCHS);
        println!("  - Learning rate: {}", LEARNING_RATE);
        println!("  - Activation: {}\n", ACTIVATION_FUNCTION);

        // Initialise network
        print!("Initialising neural network... ");
        let now = Instant::now();
        neural_network = NeuralNetwork::new(LAYERS.to_vec(), ActivationFunction::new(ACTIVATION_FUNCTION));
        println!("done in {:.2?}.", now.elapsed());

        // Train the network
        println!("\n[INFO] Starting to train the network.");
        let now = Instant::now();
        neural_network.train(&mut training_data, BATCH_SIZE, EPOCHS, LEARNING_RATE, &mut validation_data);
        println!("[INFO] Network trained. Total training time: {:.2?}", now.elapsed());

        // Save the network
        print!("\n\nSaving the network at {}.json... ", FILE_NAME);
        let now = Instant::now();
        neural_network.save_network(FILE_NAME);
        println!("done in {:.2?}.", now.elapsed());
    }

    // Test the network
    print!("\n\nEvaluating the network... ");
    let now = Instant::now();
    let result = neural_network.evaluate(&testing_data);
    println!("done in {:.2?}.", now.elapsed());
    println!("{}", result);

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