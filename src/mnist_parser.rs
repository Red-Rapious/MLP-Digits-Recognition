use mnist::*;

const IMAGE_SIDE: usize = 28;

pub fn load_data(training_length: u32, test_lenght: u32) -> (Vec<Vec<f64>>, Vec<u8>, Vec<Vec<f64>>, Vec<u8>){

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
        .test_set_length(test_lenght)
        .finalize();

    (split_buffer(&trn_img),
    trn_lbl,
    split_buffer(&tst_img),
    tst_lbl)
}

fn split_buffer(buffer: &Vec<u8>) -> Vec<Vec<f64>> {
    assert!(buffer.len() % IMAGE_SIDE*IMAGE_SIDE == 0);
    let mut data: Vec<Vec<f64>> = vec![];
    for i in 0..buffer.len()/(IMAGE_SIDE*IMAGE_SIDE) {
        data.push(vec![]);
        for y in 0..IMAGE_SIDE*IMAGE_SIDE {
            data[i].push(buffer[i*IMAGE_SIDE*IMAGE_SIDE + y] as f64 / 256.0);
        }
    }

    data
}

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