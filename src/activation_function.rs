/// A structure containing an activation function and its derivative.
pub struct ActivationFunction {
    pub activation_function: Box<dyn Fn(&f64) -> f64>,
    pub activation_prime: Box<dyn Fn(&f64) -> f64>
}

impl ActivationFunction {
    pub fn new(activation_function: Box<dyn Fn(&f64) -> f64>, activation_prime: Box<dyn Fn(&f64) -> f64>) -> Self {
        ActivationFunction { 
            activation_function, 
            activation_prime 
        }
    }

    pub fn simoid() -> Self {
        ActivationFunction { 
            activation_function: Box::new(sigmoid), 
            activation_prime: Box::new(sigmoid_prime)
        }
    }
}

/// Logistic function evaluated in `x`.
fn sigmoid(x: &f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Derivative of sigmoid evaluated in `x`.
fn sigmoid_prime(x: &f64) -> f64 {
    let s = sigmoid(x);
    s * (1.0 - s)
}