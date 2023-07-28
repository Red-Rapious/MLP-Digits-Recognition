/// A structure containing an activation function and its derivative.
pub struct ActivationFunction {
    pub activation_function: Box<dyn Fn(f64) -> f64>,
    pub activation_prime: Box<dyn Fn(f64) -> f64>
}

impl ActivationFunction {
    pub fn new(activation_function: Box<dyn Fn(f64) -> f64>, activation_prime: Box<dyn Fn(f64) -> f64>) -> Self {
        ActivationFunction { 
            activation_function, 
            activation_prime 
        }
    }

    pub fn sigmoid() -> Self {
        ActivationFunction { 
            activation_function: Box::new(sigmoid), 
            activation_prime: Box::new(sigmoid_prime)
        }
    }

    #[allow(non_snake_case)]
    pub fn ReLU() -> Self {
        ActivationFunction { 
            activation_function: Box::new(ReLU), 
            activation_prime: Box::new(ReLU_prime)
        }
    }
}

/// Logistic function.
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Derivative of sigmoid.
fn sigmoid_prime(x: f64) -> f64 {
    let s = sigmoid(x);
    s * (1.0 - s)
}

/// Rectification function.
#[allow(non_snake_case)]
fn ReLU(x: f64) -> f64 {
    if x >= 0.0 { 
        x 
    } else {
        0.0
    }
}

/// Derivative of ReLU
#[allow(non_snake_case)]
fn ReLU_prime(x: f64) -> f64 {
    if x >= 0.0 {
        1.0
    } else {
        0.0
    }
}