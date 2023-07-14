use crate::neural_network::NeuralNetwork;

pub mod neural_network;
pub mod tests;
pub mod utility;

fn main() {
    let _ = NeuralNetwork::new(vec![28*28, 16, 16, 10]);
}