use mnist::*;
use crate::utility::split_vector;

const IMAGE_SIDE: usize = 28;

/// Uses the `mnist` crate to load the training and testing sets.
/// Returns `(training_images, training_labels, testing_images, testing_labels)`
pub fn load_data(training_length: u32, test_length: u32) -> (Vec<Vec<f64>>, Vec<u8>, Vec<Vec<f64>>, Vec<u8>){
    // Deconstruct the returned Mnist struct.
    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(training_length)
        //.validation_set_length(10_000)
        .test_set_length(test_length)
        .finalize();

    (split_vector(&trn_img, IMAGE_SIDE*IMAGE_SIDE), trn_lbl, 
     split_vector(&tst_img, IMAGE_SIDE*IMAGE_SIDE), tst_lbl)
}

/// Splits a long vector in a vector of 2D vectors representing each image.
fn _buffer_to_3d_vector(buffer: &Vec<u8>) -> Vec<Vec<Vec<f64>>> {
    assert!(buffer.len() % IMAGE_SIDE*IMAGE_SIDE == 0);
    
    let mut data: Vec<Vec<Vec<f64>>> = vec![];
    for i in 0..buffer.len()/(IMAGE_SIDE*IMAGE_SIDE) {
        data.push(vec![]);
        for y in 0..IMAGE_SIDE {
            data[i].push(vec![]);
            for x in 0..IMAGE_SIDE {
                data[i][y].push(buffer[i*IMAGE_SIDE*IMAGE_SIDE + y*IMAGE_SIDE + x] as f64 / 256.0);
            }
        }
    }

    data
}