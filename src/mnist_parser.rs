use mnist::*;
use crate::utility::split_vector;

const IMAGE_SIDE: usize = 28;

/// Uses the `mnist` crate to load the training and testing sets.
/// Returns `(training_data, validation_data, test_data)`
pub fn load_data(training_length: u32, validation_length: u32, test_length: u32) -> (Vec<(Vec<f64>, u8)>, Vec<(Vec<f64>, u8)>, Vec<(Vec<f64>, u8)>) {
    // Deconstruct the returned Mnist struct.
    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        val_img,
        val_lbl
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(training_length)
        .validation_set_length(validation_length)
        .test_set_length(test_length)
        .finalize();

     // Zip the images and labels
    let training_data: Vec<(Vec<f64>, u8)> = split_vector(&trn_img, IMAGE_SIDE*IMAGE_SIDE).into_iter().zip(trn_lbl.into_iter()).collect();
    let validation_data: Vec<(Vec<f64>, u8)> = split_vector(&val_img, IMAGE_SIDE*IMAGE_SIDE).into_iter().zip(val_lbl.into_iter()).collect();
    let testing_data: Vec<(Vec<f64>, u8)> = split_vector(&tst_img, IMAGE_SIDE*IMAGE_SIDE).into_iter().zip(tst_lbl.into_iter()).collect();

    (training_data, validation_data, testing_data)
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