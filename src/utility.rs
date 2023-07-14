use std::vec;

/// This file contains diverse utility functions,
/// such as linear algebra operations, the cost function, and sigmoid

/// Given a matrix A and a vector X, returns the vector AX.
pub fn matrix_vector_product(matrix: &Vec<Vec<f64>>, vector: &Vec<f64>) -> Vec<f64> {
    assert_eq!(matrix[0].len(), vector.len());

    let mut result = vec![];
    for j in 0..matrix.len() {
        result.push(0.0);
        for i in 0..vector.len() {
            result[j] += matrix[j][i] * vector[i]
        }
    }
    result
}

pub fn vectors_sum(vector1: &Vec<f64>, vector2: &Vec<f64>) -> Vec<f64> {
    assert_eq!(vector1.len(), vector2.len());
    let mut sum = vec![];
    for i in 0..vector1.len() {
        sum.push(vector1[i] + vector2[i]);
    }
    sum
}

/// Given two vectors X1 and X2, returns ||X1-X2||^2, where ||.|| is the euclidian norm.
pub fn euclidian_distance(vector1: &Vec<f64>, vector2: &Vec<f64>) -> f64 {
    assert_eq!(vector1.len(), vector2.len());
    let mut total = 0.0;

    for i in 0..vector1.len() {
        total += (vector1[i]-vector2[i])*(vector1[i]-vector2[i])
    }
    total
}

/// Logistic function evaluated in x.
pub fn sigmoid(x: &f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Splits a long vector in a vector of subvectors of length 'length'.
pub fn split_vector(vector: &Vec<u8>, length: usize) -> Vec<Vec<f64>> {
    assert!(vector.len() % length == 0);

    let mut data: Vec<Vec<f64>> = vec![];
    for i in 0..vector.len()/(length) {
        // Add the image to 'data', in the form of a vector of length IMAGE_SIDE*IMAGE_SIDE 
        data.push(vec![]);
        for y in 0..length {
            data[i].push(vector[i*length + y] as f64 / 256.0);
        }
    }
    data
}