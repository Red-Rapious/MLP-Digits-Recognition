#[cfg(test)]
mod tests {
    use crate::evaluation_result::EvaluationResult;
    use crate::neural_network::NeuralNetwork;
    use crate::utility::*;
    use crate::mnist_parser::load_data;

    #[test]
    fn test_nn_initialisation() {
        let _ = NeuralNetwork::new(vec![100; 10]);
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

    #[test]
    fn test_euclidian_distance() {
        assert_eq!(
            euclidian_distance(
                &vec![1.0, 2.0, 3.0, 4.0], 
                &vec![5.0, 6.0, 7.0, 8.0]
            ),
            4.0*16.0
        );
    } 

    #[test]
    fn test_feed_forward() {
        let nn = NeuralNetwork::new(vec![3, 4, 5, 6]);
        let input = vec![0.5, 0.2, 0.8];
        let output = nn.feed_forward(input, &sigmoid);
        println!("{:?}", output);
    }

    #[test]
    fn test_mnist_parser() {
        load_data(100, 100);
    }

    #[test]
    fn test_accuracy_null() {
        assert_eq!(EvaluationResult::new(0, 0).accuracy(), None);
    }

    #[test]
    fn test_accuracy_half() {
        assert_eq!(EvaluationResult::new(8, 8).accuracy(), Some(0.5));
    }
}