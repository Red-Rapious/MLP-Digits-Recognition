use crate::neural_network::NeuralNetwork;
use crate::mnist_parser::*;
use crate::utility::sigmoid;

pub mod neural_network;
pub mod tests;
pub mod utility;
pub mod mnist_parser;

fn main() {
    let neural_network = NeuralNetwork::new(vec![28*28, 16, 16, 10]);
    let (training_images, training_labels, _, _) = load_data(1, 0);
    let result = neural_network.feed_forward(training_images[0].clone(), &(|x| sigmoid(*x, 1.0)));
    println!("result: {:?} \nlabel: {}", result, training_labels[0]);
}