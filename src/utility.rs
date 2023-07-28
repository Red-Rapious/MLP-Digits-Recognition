// This file contains diverse utility functions,
// such as linear algebra operations, the cost function, and sigmoid


/// Given a matrix `A` and a vector `X`, returns the vector `AX`.
pub fn matrix_vector_product(matrix: &Vec<Vec<f64>>, vector: &Vec<f64>) -> Vec<f64> {
    assert_eq!(matrix[0].len(), vector.len(), "The matrix and vector shapes are incompatible.");

    let mut result = Vec::with_capacity(matrix.len());
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
    assert_eq!(vector1.len(), vector2.len(), "The two vectors have different sizes.");

    for i in 0..vector1.len() {
        vector1[i] += vector2[i];
    }
}

/// Given one mutable matrix `A1` and another matrix `A2` of the same size, adds `A2` to `A1`.
pub fn matrices_sum(matrix1: &mut Vec<Vec<f64>>, matrix2: &Vec<Vec<f64>>) {
    assert_eq!(matrix1.len(), matrix2.len(), "The two matrices have different heights.");
    assert_eq!(matrix1[0].len(), matrix2[0].len(), "The two matrices have difference widths.");

    for i in 0..matrix1.len() {
        for j in 0..matrix1[i].len() {
            matrix1[i][j] += matrix2[i][j];
        }
    }
}

/// Given one mutable tensor `T1` and another tensor `T2` of the same size, adds `T2` to `T1`.
pub fn tensor_sum(tensor1: &mut Vec<Vec<Vec<f64>>>, tensor2: &Vec<Vec<Vec<f64>>>) {
    assert_eq!(tensor1.len(), tensor2.len(), "The two tensors have different heights.");

    for i in 0..tensor1.len() {
        for j in 0..tensor1[i].len() {
            for k in 0..tensor1[i][j].len() {
                tensor1[i][j][k] += tensor2[i][j][k];
            }
        }
    }
}

/// Given two vectors `X1` and `X2`, return `X1 * X2^T`, where `X2^T` is the transpose of `X2`
pub fn vectors_transpose_product(vector1: &Vec<f64>, vector2: &Vec<f64>) -> Vec<Vec<f64>> {
    let mut matrix = Vec::with_capacity(vector1.len());
    for i in 0..vector1.len() {
        matrix.push(Vec::with_capacity(vector2.len()));
        for j in 0..vector2.len() {
            matrix[i].push(vector1[i] * vector2[j]);
        }
    }
    matrix
}

/// Given a matrix `A`, returns `A^T`, the transpose of `A`.
pub fn transpose(matrix: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let mut result = Vec::with_capacity(matrix[0].len());
    for i in 0..matrix[0].len() {
        result.push(Vec::with_capacity(matrix.len()));
        for j in 0..matrix.len() {
            result[i].push(matrix[j][i]);
        }
    }
    result
}

/// Splits a long vector in a vector of subvectors of length `length`.
pub fn split_vector(vector: &Vec<u8>, length: usize) -> Vec<Vec<f64>> {
    assert!(vector.len() % length == 0, "The vector length is not a multple of the length.");

    let mut data: Vec<Vec<f64>> = Vec::with_capacity(vector.len()/(length));
    for i in 0..vector.len()/(length) {
        // Add the image to `data`, in the form of a vector of length `length*length`
        data.push(Vec::with_capacity(length));
        for y in 0..length {
            data[i].push(vector[i*length + y] as f64 / 256.0);
        }
    }
    data
}