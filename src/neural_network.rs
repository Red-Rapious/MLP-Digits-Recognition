use std::io::Write;
use std::time::Instant;

use rand_distr::{Normal, Distribution};
use rand::thread_rng;
use rand::seq::SliceRandom;

use crate::activation_function::ActivationFunction;
use crate::utility::*;
use crate::evaluation_result::*;

use serde::{Serialize, Deserialize};
use std::fs::File;

use progress_bar::*;

/// A structure containing the actual neural network layers, weights, and biases.
#[derive(Serialize, Deserialize)]
pub struct NeuralNetwork {
    /// The number of neurons in each layer, including first and last layers.
    layers: Vec<usize>,
    /// The weight of each synapse. The shape of `weights\[i\]` is `(layers\[i+1\], layers\[i\])`.
    weights: Vec< Vec<Vec<f64>> >,
    /// The bias of each synaspse. The size of `biases\[i\]` is `(layers\[i+1\])`.
    biases: Vec< Vec<f64> >,
    /// The activation function and its derivative
    activation_function: ActivationFunction
}

impl NeuralNetwork {
    /// Initialise a new Neural Network with random weights and biases.
    pub fn new(layers: Vec<usize>, activation_function: ActivationFunction) -> Self {
        // Normal distribution sampler
        let normal = Normal::new(0.0, 1.0).unwrap();

        // Initially empty vectors
        let mut weights = vec![];
        let mut biases = vec![];

        // Generate a new random weight for each synapse
        for i in 0..layers.len()-1 {
            let mut w_matrix = vec![];
            let mut b_vector = vec![];

            for _ in 0..layers[i+1] {
                // Push a random bias in the vector
                b_vector.push(normal.sample(&mut rand::thread_rng()));

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
            layers,
            weights,
            biases,
            activation_function
        }
    }

    /// Computes the activations of the output layer, given the activations of the input layer.
    pub fn feed_forward(&self, mut activation: Vec<f64>) -> Vec<f64> {
        //let mut activation = input;
        for i in 0..self.layers.len()-1 {
            // Multiply the activation by the weights matrix
            activation = matrix_vector_product(&self.weights[i], &activation);
            // Add the biases
            vectors_sum(&mut activation , &self.biases[i]);
            
            // Apply the activation function to every coefficient
            // Here, I will not use `map` but instead a loop, to map in place
            for i in 0..activation.len() {
                activation[i] = self.activation_function.activation_function(activation[i]);
            }
            //activation = activation.iter().map(self.activation_function.activation_function).collect();
        }
        activation
    }

    /// Trains the network using stochastic gradient descend. 
    /// Each iteration is using a mini-batch of `batch_size` to pass backwards.
    /// If `validation_data` is provided, the network will test itself after each iteration.
    /// `learning_rate` is often written as η, or `eta`.
    pub fn train(&mut self, training_data: &mut Vec<(Vec<f64>, u8)>, batch_size: usize, epochs_nb: usize, learning_rate: f64, validation_data: &mut Vec<(Vec<f64>, u8)>) {
        assert!(batch_size != 0);
        if training_data.len() % batch_size != 0 {
            println!("[Warning]: the last batch is missing {} images to complete a full batch size.", training_data.len() % batch_size);
        }
        
        for epoch in 0..epochs_nb {
            std::io::stdout().flush().unwrap();
            let epoch_start_time = Instant::now();
            // Create new batches, different from the previous iteration
            training_data.shuffle(&mut thread_rng());
            let batches: Vec<&[(Vec<f64>, u8)]> = training_data.chunks(batch_size).collect();
            
            init_progress_bar(batches.len());
            set_progress_bar_action(format!("Epoch {}/{}", epoch+1, epochs_nb).as_str(), Color::White, Style::Normal);

            // Compute gradient and update weights and biases for each batch
            for batch in batches {
                self.learn(batch, learning_rate);
                inc_progress_bar();
            }
            finalize_progress_bar();

            // Display some update, and compute accuracy if validation is enabled.
            if !validation_data.is_empty() {
                let result = self.evaluate(validation_data);
                println!("      => Completed in {:.2?}, with validation accuracy of {:.1?}%.", epoch_start_time.elapsed(), result.accuracy().unwrap()*100.0);
            }
            else {
                println!("      => Completed in {:.2?}.", epoch_start_time.elapsed());
            }
        }
    }

    /// Updates the weights and biases using the gradient computed by backpropagation on one batch.
    fn learn(&mut self, batch: &[(Vec<f64>, u8)], learning_rate: f64) {
        // This is the base iterative algorithm, as described on the Wikipedia
        // article of 'Stochastic gradient descent'

        // Initialise gradients for weights and biases to zeros.
        let mut grad_weights: Vec<Vec<Vec<f64>>> = Vec::with_capacity(self.weights.len());
        for i in 0..self.weights.len() {
            grad_weights.push(vec![]);
            for _ in 0..self.weights[i].len() {
                grad_weights[i].push(vec![0.0; self.weights[i][0].len()]);
            }
        }
        let mut grad_biases: Vec<Vec<f64>> = Vec::with_capacity(self.biases.len());
        for biais in self.biases.iter() {
            grad_biases.push(vec![0.0; biais.len()])
        }

        // Compute the overall approximate gradient of the cost on all images of the batch.
        for (image, label) in batch {
            // Use backpropagation to find the small variation of gradient
            let (delta_grad_w, delta_grad_b) = self.backpropagation(image, label);
            // Update the gradient by adding the small variation
            tensor_sum(&mut grad_weights, &delta_grad_w);
            matrices_sum(&mut grad_biases, &delta_grad_b);
        }

        // Update the weights and biases based on the gradient and learning rate.
        for i in 0..self.weights.len() {
            for j in 0..self.weights[i].len() {
                for k in 0..self.weights[i][j].len() {
                    // We try to minimise the cost function, so we substract the gradient
                    self.weights[i][j][k] -= grad_weights[i][j][k] * learning_rate / (batch.len() as f64)
                }
            }
        }

        for i in 0..self.biases.len() {
            for j in 0..self.biases[i].len() {
                self.biases[i][j] -= grad_biases[i][j] * learning_rate / (batch.len() as f64)
            }
        }
    }

    /// Uses the Backpropagation algorithm to compute the approximate gradient 
    /// of the cost function with respect to the weights and the biases.
    fn backpropagation(&self, image: &[f64], label: &u8) -> (Vec< Vec<Vec<f64>> >, Vec< Vec<f64> >) {
        // Current activation, starting from the input layer
        let mut activation = image.to_owned();

        // Stores all the activations and weighted sums during the forward pass
        let mut layers_activations = Vec::with_capacity(self.layers.len() - 1);
        layers_activations.push(image.to_owned());
        let mut weighted_sums = Vec::with_capacity(self.layers.len() - 1);

        // Feed-forward pass
        for i in 0..self.layers.len()-1 {
            // Compute the weighted sum
            activation = matrix_vector_product(&self.weights[i], &activation);
            vectors_sum(&mut activation, &self.biases[i]);
            weighted_sums.push(activation.clone());

            // Compute the non-linear activation
            for i in 0..activation.len() {
                activation[i] = self.activation_function.activation_function(activation[i]);
            }
            layers_activations.push(activation.clone());
        }

        // Transforms the label into a vector made of 0 everywhere execept a -1 at the correct position
        let vec_label: Vec<f64> = (0..10).map(|x| if x == *label { -1.0 } else { 0.0 }).collect();

        // Define delta, the difference between the optimal activation for this image,
        // and the current output of the neural network
        // The initial expression of delta corresponds to the derivative of the cost function,
        // without the factor 2
        let mut delta = activation.clone();
        vectors_sum(&mut delta, &vec_label);

        let last_layer = weighted_sums.len() - 1;
        // Multiply `delta` by the last weighted sum, to which `activation_prime` is applied
        assert_eq!(delta.len(), weighted_sums[last_layer].len(), "Incompatible sizes for delta and the last layer's weighted sum.");
        for i in 0..delta.len() {
            delta[i] *= self.activation_function.activation_prime(weighted_sums[last_layer][i])
        }

        // Starts the construction of the gradients.
        let mut grad_biases = Vec::with_capacity(self.biases.len());
        let mut grad_weights = Vec::with_capacity(self.weights.len());
        // Note that the gradients are constructed in reverse and will be reversed at the end
        grad_biases.push(delta.clone());
        grad_weights.push(vectors_transpose_product(&delta, &layers_activations[layers_activations.len()-2].clone()));

        // Backpropagation
        for layer in 2..self.layers.len() {
            // Backpropagate: determine delta depending on the previous layer
            delta = matrix_vector_product(&transpose(&self.weights[self.weights.len() - layer + 1]), &delta);
            
            // Multiply `delta` by the weighted sum, to which `activation_prime` is applied
            assert_eq!(delta.len(), weighted_sums[weighted_sums.len() - layer].len(), "Incompatible sizes for delta and the layer's weighted sum, for layer number {}", self.layers.len() - layer - 1);
            for i in 0..delta.len() {
                delta[i] *= self.activation_function.activation_prime(weighted_sums[weighted_sums.len() - layer][i])
            }

            // Add the gradient of biases of the layer
            grad_biases.push(delta.clone());

            // Add the gradient of weights of the layer
            grad_weights.push(vectors_transpose_product(&delta, &layers_activations[layers_activations.len() - layer -1]));
        }

        // Reverse the grad vectors since they were build backwards
        grad_weights.reverse();
        grad_biases.reverse();
        (grad_weights, grad_biases)
    }

    /// Predicts the digit in a given image.
    pub fn predict(&self, input: Vec<f64>) -> u8 {
        let result = self.feed_forward(input);

        // Select the highest activation of the output layer.
        let mut maxi = (result[0], 0);
        for i in 1..result.len() {
            if result[i] > maxi.0 {
                maxi = (result[i], i);
            }
        }
        maxi.1 as u8
    }

    /// Evaluates the neural network on a test set.
    pub fn evaluate(&self, test_data: &Vec<(Vec<f64>, u8)>) -> EvaluationResult {
        let (mut corrects, mut incorrects) = (0, 0);

        // Counts the number of corrects and incorrects classifications
        for data in test_data.iter() {
            let prediction = self.predict(data.clone().0);
            match prediction == data.1 {
                true => corrects += 1,
                false => incorrects += 1
            }
        }

        EvaluationResult::new(corrects, incorrects)
    }
}

impl NeuralNetwork {
    /// Saves the trained network to a `JSON` file. Note that `file_name` should not contain the `.json` extension.
    pub fn save_network(&self, file_name: &str){
        // Convert the network to a string
        let serialized = serde_json::to_string(&self).unwrap();

        let mut file: File;
        // Save the string to a file
        if std::path::Path::new(format!("{}.json", file_name).as_str()).exists() {
            file = File::create(format!("{}_bis.json", file_name).as_str()).unwrap();
        } else {
            file = File::create(format!("{}.json", file_name).as_str()).unwrap();
        }
        file.write_all(&serialized.into_bytes()).unwrap();
    }

    /// Loads a pre-trained network from a `JSON` file. Note that `file_name` should not contain the `.json` extension.
    pub fn load_network(file_name: &str) -> Self {
        let file_path = std::fs::read_to_string(format!("{}.json", file_name).as_str()).unwrap();
        serde_json::from_str(file_path.as_str()).unwrap()
    }
}