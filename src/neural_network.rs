use rand_distr::{Normal, Distribution};
use crate::utility::*;
use crate::evaluation_result::*;

/// A structure containing the actual neural network layers, weights, and biases.
pub struct NeuralNetwork {
    /// The number of neurons in each layer, including first and last layers.
    layers: Vec<usize>,
    /// The weight of each synapse. The shape of weights\[i\] is (layers\[i+1\], layers\[i\]).
    weights: Vec< Vec<Vec<f64>> >,
    /// The bias of each synaspse. biases\[0\] should always be a vector full of zeros.
    biases: Vec< Vec<f64> >
}

impl NeuralNetwork {
    /// Initialise a new Neural Network with random weights and biases.
    pub fn new(layers: Vec<usize>) -> Self {
        // Normal distribution sampler
        let normal = Normal::new(0.0, 1.0).unwrap();

        // Initially empty vectors
        let mut weights = vec![];
        let mut biases = vec![];

        // Generate a new random weight for each synapse
        for i in 0..(layers.len()-1) {
            let mut w_matrix = vec![];
            let mut b_vector = vec![];

            for _ in 0..layers[i+1] {
                // Push a random bias in the vector
                b_vector.push(
                    if i == 0 { 
                        0.0 // The first layer being an input layer, we will use biases of zero
                    } else {
                        normal.sample(&mut rand::thread_rng())
                    }
                );

                // Push a random line of weights in the matrix
                let mut line = vec![];
                for _ in 0..layers[i] {
                    line.push(normal.sample(&mut rand::thread_rng()));
                }
                w_matrix.push(line);
            }
            weights.push(w_matrix);
            biases.push(b_vector);
        }
        NeuralNetwork {
            layers: layers,
            weights: weights,
            biases: biases
        }
    }

    /// Computes the activations of the output layer, given the activations of the input layer.
    pub fn feed_forward(&self, input: Vec<f64>, activation_function: &dyn Fn(&f64) -> f64) -> Vec<f64> {
        let mut activation = input;
        for i in 0..self.layers.len()-1 {
            assert_eq!(activation.len(), self.layers[i]);
            activation = matrix_vector_product(&self.weights[i], &activation);
            activation = vectors_sum(&activation , &self.biases[i]);
            activation = activation.iter().map(activation_function).collect();
        }
        activation
    }

    /// Predicts the digit in a given image.
    pub fn predict(&self, input: Vec<f64>, activation_function: &dyn Fn(&f64) -> f64) -> u8 {
        let result = self.feed_forward(input, activation_function);
        let mut maxi = (result[0], 0);
        for i in 1..result.len() {
            if result[i] > maxi.0 {
                maxi = (result[i], i);
            }
        }
        maxi.1 as u8
    }

    /// Evaluates the neural network on a test set.
    pub fn evaluate(&self, test_images: Vec<Vec<f64>>, test_labels: Vec<u8>, activation_function: &dyn Fn(&f64) -> f64) -> EvaluationResult {
        assert_eq!(test_images.len(), test_labels.len());
        let (mut corrects, mut incorrects) = (0, 0);

        // Counts the number of corrects and incorrects classifications
        for i in 0..test_images.len() {
            let prediction = self.predict(test_images[i].clone(), activation_function);
            match prediction == test_labels[i] {
                true => corrects += 1,
                false => incorrects += 1
            }
        }

        EvaluationResult::new(corrects, incorrects)
    }
}