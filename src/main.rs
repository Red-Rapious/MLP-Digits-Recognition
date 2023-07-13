use crate::neural_network::NeuralNetwork;

pub mod neural_network;
pub mod tests;

fn main() {
    let _ = NeuralNetwork::new(vec![28*28, 16, 16, 10]);
}