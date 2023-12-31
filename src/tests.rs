#[cfg(test)]
mod tests {
    use crate::evaluation_result::EvaluationResult;
    use crate::neural_network::NeuralNetwork;
    use crate::utility::*;
    use crate::mnist_parser::load_data;
    use crate::activation_function::ActivationFunction;

    #[test]
    fn test_nn_initialisation() {
        let _ = NeuralNetwork::new(vec![100; 10], ActivationFunction::sigmoid());
    }

    #[test]
    fn test_matrix_vector_product() {
        assert_eq!(matrix_vector_product(
            &vec![vec![1.0, 2.0, 3.0, 4.0], 
                 vec![5.0, 6.0, 7.0, 8.0], 
                 vec![9.0, 10.0, 11.0, 12.0]
                ], 
            &vec![13.0, 14.0, 15.0, 16.0]), 
            vec![150.0, 382.0, 614.0]
        );
    }

    // TODO: test matrices sum, tensor sum, vectors sum

    #[test]
    fn test_feed_forward_sigmoid() {
        let nn = NeuralNetwork::new(vec![3, 4, 5, 6], ActivationFunction::sigmoid());
        let input = vec![0.5, 0.2, 0.8];
        let output = nn.feed_forward(input);
        println!("{:?}", output);
    }

    #[test]
    
    fn test_feed_forward_relu() {
        let nn = NeuralNetwork::new(vec![3, 4, 5, 6], ActivationFunction::ReLU());
        let input = vec![0.5, 0.2, 0.8];
        let output = nn.feed_forward(input);
        println!("{:?}", output);
    }

    #[test]
    fn test_mnist_parser() {
        let (_, _, _) = load_data(100, 100, 100);
    }

    #[test]
    fn test_accuracy_null() {
        assert_eq!(EvaluationResult::new(0, 0).accuracy(), None);
    }

    #[test]
    fn test_accuracy_half() {
        assert_eq!(EvaluationResult::new(8, 8).accuracy(), Some(0.5));
    }

    #[test]
    fn test_train_sigmoid() {
        let mut nn = NeuralNetwork::new(vec![28*28, 16, 10], ActivationFunction::sigmoid());
        let (mut training_data, mut validation_data, _) = load_data(10, 10, 0);
        nn.train(&mut training_data, 2, 2, 1.0, &mut validation_data);
    }

    #[test]
    fn test_train_relu() {
        let mut nn = NeuralNetwork::new(vec![28*28, 16, 10], ActivationFunction::ReLU());
        let (mut training_data, mut validation_data, _) = load_data(10, 10, 0);
        nn.train(&mut training_data, 2, 2, 1.0, &mut validation_data);
    }
}