pub enum ActivationFunctionType {
    Sigmoid,
    ReLU
}

pub struct ActivationFunction {
    function_type: ActivationFunctionType
}

impl ActivationFunction {
    pub fn new(function_type: ActivationFunctionType) -> Self {
        ActivationFunction { function_type }
    }

    pub fn sigmoid() -> Self {
        ActivationFunction::new(ActivationFunctionType::Sigmoid)
    }

    #[allow(non_snake_case)]
    pub fn ReLU() -> Self {
        ActivationFunction::new(ActivationFunctionType::Sigmoid)
    }

    pub fn activation_function(&self, x: f64) -> f64 {
        match self.function_type {
            ActivationFunctionType::Sigmoid => sigmoid(x),
            ActivationFunctionType::ReLU => ReLU(x)
        }
    }

    pub fn activation_prime(&self, x: f64) -> f64 {
        match self.function_type {
            ActivationFunctionType::Sigmoid => sigmoid_prime(x),
            ActivationFunctionType::ReLU => ReLU_prime(x)
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