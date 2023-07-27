use rand_distr::{Normal, Distribution};
use rand::thread_rng;
use rand::seq::SliceRandom;
use crate::utility::*;
use crate::evaluation_result::*;

/// A structure containing the actual neural network layers, weights, and biases.
pub struct NeuralNetwork {
    /// The number of neurons in each layer, including first and last layers.
    layers: Vec<usize>,
    /// The weight of each synapse. The shape of `weights\[i\]` is `(layers\[i+1\], layers\[i\])`.
    weights: Vec< Vec<Vec<f64>> >,
    /// The bias of each synaspse. The size of `biases\[i\]` is `(layers\[i+1\])`.
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
            biases
        }
    }

    /// Computes the activations of the output layer, given the activations of the input layer.
    pub fn feed_forward(&self, input: Vec<f64>, activation_function: &dyn Fn(&f64) -> f64) -> Vec<f64> {
        let mut activation = input;
        for i in 0..self.layers.len()-1 {
            // Multiply the activation by the weights matrix
            activation = matrix_vector_product(&self.weights[i], &activation);
            // Add the biases
            vectors_sum(&mut activation , &self.biases[i]);
            
            // Apply the activation function to every coefficient
            // Here, I will not use `map` but instead a loop, to map in place
            for i in 0..activation.len() {
                activation[i] = activation_function(&activation[i]);
            }
            //activation = activation.iter().map(activation_function).collect();
        }
        activation
    }

    /// Trains the network using stochastic gradient descend. 
    /// Each iteration is using a mini-batch of `batch_size` to pass backwards.
    /// If `validation_data` is provided, the network will test itself after each iteration.
    /// `learning_rate` is often written as η, or `eta`.
    pub fn train(&mut self, training_data: &mut Vec<(Vec<f64>, u8)>, batch_size: usize, epochs_nb: usize, learning_rate: f64, validation_data: &mut Vec<(Vec<f64>, u8)>, activation_function: &dyn Fn(&f64) -> f64, activation_prime: &dyn Fn(&f64) -> f64) {
        assert!(batch_size != 0);
        if training_data.len() % batch_size != 0 {
            println!("[Warning]: the last batch is missing {} images to complete a full batch size.", training_data.len() % batch_size);
        }
        
        for epoch in 0..epochs_nb {
            // Create new batches, different from the previous iteration
            training_data.shuffle(&mut thread_rng());
            let batches: Vec<&[(Vec<f64>, u8)]> = training_data.chunks(batch_size).collect();

            // Compute gradient and update weights and biases for each batch
            for batch in batches {
                self.learn(batch, learning_rate, activation_function, activation_prime);
            }

            // Display some update, and compute accuracy if validation is enabled.
            if !validation_data.is_empty() {
                let result = self.evaluate(validation_data, &sigmoid);
                println!("  [PROGRESS] Epoch {}/{}: validation accuracy of {}.", epoch+1, epochs_nb, result.accuracy().unwrap());
            }
            else {
                println!("  [PROGRESS] Epoch {}/{} completed.", epoch+1, epochs_nb);
            }
        }
    }

    /// Updates the weights and biases using the gradient computed by backpropagation on one batch.
    fn learn(&mut self, batch: &[(Vec<f64>, u8)], learning_rate: f64, activation_function: &dyn Fn(&f64) -> f64, activation_prime: &dyn Fn(&f64) -> f64) {
        // This is the base iterative algorithm, as described on the Wikipedia
        // article of 'Stochastic gradient descent'

        // Initialise gradients for weights and biases to zeros.
        let mut grad_weights: Vec<Vec<Vec<f64>>> = vec![];
        for i in 0..self.weights.len() {
            grad_weights.push(vec![]);
            for _ in 0..self.weights[i].len() {
                grad_weights[i].push(vec![0.0; self.weights[i][0].len()]);
            }
        }
        let mut grad_biases: Vec<Vec<f64>> = vec![];
        for biais in self.biases.iter() {
            grad_biases.push(vec![0.0; biais.len()])
        }

        // Compute the overall approximate gradient of the cost on all images of the batch.
        for (image, label) in batch {
            // Use backpropagation to find the small variation of gradient
            let (delta_grad_w, delta_grad_b) = self.backpropagation(image, &label, activation_function, activation_prime);
            // Update the gradient by adding the small variation
            tensor_sum(&mut grad_weights, &delta_grad_w);
            matrices_sum(&mut grad_biases, &delta_grad_b);
        }

        // Update the weights and biases based on the gradient and learning rate.
        for i in 0..self.weights.len() {
            for j in 0..self.weights[i].len() {
                for k in 0..self.weights[i][j].len() {
                    // We try to minimise the cost function, so we substract the gradient
                    self.weights[i][j][k] -= grad_weights[i][j][k] * learning_rate
                }
            }
        }

        for i in 0..self.biases.len() {
            for j in 0..self.biases[i].len() {
                self.biases[i][j] -= grad_biases[i][j] * learning_rate
            }
        }
    }

    /// Uses the Backpropagation algorithm to compute the approximate gradient 
    /// of the cost function with respect to the weights and the biases.
    fn backpropagation(&self, image: &Vec<f64>, label: &u8, activation_function: &dyn Fn(&f64) -> f64, activation_prime: &dyn Fn(&f64) -> f64) -> (Vec< Vec<Vec<f64>> >, Vec< Vec<f64> >) {
        // Initialise the gradients
        let mut grad_biases = vec![];
        let mut grad_weights = vec![];

        // Current activation, starting from the input layer
        let mut activation = image.clone();

        // Stores all the activations and weighted sums during the forward pass
        let mut layers_activations = vec![image.clone()];
        let mut weighted_sums = vec![];

        // Feed-forward pass
        for i in 0..self.weights.len() {
            // Compute the weighted sum
            activation = matrix_vector_product(&self.weights[i], &activation);
            vectors_sum(&mut activation, &self.biases[i]);
            weighted_sums.push(activation.clone());

            // Compute the non-linear activation
            for i in 0..activation.len() {
                activation[i] = activation_function(&activation[i]);
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

        // Multiply `delta` by the last weighted sum, to which `activation_prime` is applied
        let last_layer = weighted_sums.len() - 1;
        assert_eq!(delta.len(), weighted_sums[last_layer].len(), "Incompatible sizes for delta and the last layer's weighted sum.");
        for i in 0..delta.len() {
            delta[i] *= activation_prime(&weighted_sums[last_layer][i])
        }

        // Starts the construction of the gradients.
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
                delta[i] *= activation_prime(&weighted_sums[weighted_sums.len() - layer][i])
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
    pub fn predict(&self, input: Vec<f64>, activation_function: &dyn Fn(&f64) -> f64) -> u8 {
        let result = self.feed_forward(input, activation_function);

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
    pub fn evaluate(&self, test_data: &Vec<(Vec<f64>, u8)>, activation_function: &dyn Fn(&f64) -> f64) -> EvaluationResult {
        let (mut corrects, mut incorrects) = (0, 0);

        // Counts the number of corrects and incorrects classifications
        for data in test_data.iter() {
            let prediction = self.predict(data.clone().0, activation_function);
            match prediction == data.1 {
                true => corrects += 1,
                false => incorrects += 1
            }
        }

        EvaluationResult::new(corrects, incorrects)
    }
}