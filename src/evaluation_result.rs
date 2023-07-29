use std::fmt;

/// Holds the results of an evaluation, i.e. the number of correct and incorrect outputs.
#[derive(Debug)]
pub struct EvaluationResult {
    corrects: usize,
    incorrects: usize
}

impl EvaluationResult {
    pub fn new(corrects: usize, incorrects: usize) -> Self {
        EvaluationResult { corrects, incorrects }
    }
    
    /// Total number of outputs, corrects or incorrects.
    pub fn test_data_length(&self) -> usize {
        self.corrects + self.incorrects
    }

    /// Computes the number of correct outputs divided by the total number of outputs.
    pub fn accuracy(&self) -> Option<f64> {
        let l = self.test_data_length();
        if l == 0 {
            None
        } else {
            Some (self.corrects as f64 / l as f64)
        }
    }
}

impl fmt::Display for EvaluationResult {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Evaluation result:\n   Corrects: {}/{}\n   Accuracy: {}", 
            self.corrects, 
            self.test_data_length(), 
            match self.accuracy() {
                None => String::from("undef"),
                Some(a) => format!("{:.1?}", a*100.0).to_string() + &String::from("%")
            })
    }
}