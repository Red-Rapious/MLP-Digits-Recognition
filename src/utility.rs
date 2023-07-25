/// This file contains diverse utility functions,
/// such as linear algebra operations, the cost function, and sigmoid
use std::vec;

/// Given a matrix `A` and a vector `X`, returns the vector `AX`.
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

/// Given one mutable vector `X1` and another vector `X2` of the same size, adds `X2` to `X1`.
pub fn vectors_sum(vector1: &mut Vec<f64>, vector2: &Vec<f64>) {
    assert_eq!(vector1.len(), vector2.len());

    for i in 0..vector1.len() {
        vector1[i] += vector2[i];
    }
}

/// Given one mutable matrix `A1` and another matrix `A2` of the same size, adds `A2` to `A1`.
pub fn matrices_sum(matrix1: &mut Vec<Vec<f64>>, matrix2: &Vec<Vec<f64>>) {
    assert_eq!(matrix1.len(), matrix2.len());
    assert_eq!(matrix1[0].len(), matrix2[0].len());

    for i in 0..matrix1.len() {
        for j in 0..matrix1[0].len() {
            matrix1[i][j] += matrix2[i][j];
        }
    }
}

/// Given two vectors `X1` and `X2` of the same size, returns `||X1-X2||^2`, where `||.||` is the euclidian norm.
pub fn euclidian_distance(vector1: &Vec<f64>, vector2: &Vec<f64>) -> f64 {
    assert_eq!(vector1.len(), vector2.len());
    let mut total = 0.0;

    for i in 0..vector1.len() {
        total += (vector1[i]-vector2[i])*(vector1[i]-vector2[i])
    }
    total
}

/// Logistic function evaluated in `x`.
pub fn sigmoid(x: &f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Splits a long vector in a vector of subvectors of length `length`.
pub fn split_vector(vector: &Vec<u8>, length: usize) -> Vec<Vec<f64>> {
    assert!(vector.len() % length == 0);

    let mut data: Vec<Vec<f64>> = vec![];
    for i in 0..vector.len()/(length) {
        // Add the image to `data`, in the form of a vector of length `IMAGE_SIDE*IMAGE_SIDE`
        data.push(vec![]);
        for y in 0..length {
            data[i].push(vector[i*length + y] as f64 / 256.0);
        }
    }
    data
}