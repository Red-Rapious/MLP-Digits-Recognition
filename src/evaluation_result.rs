/// Holds the results of an evaluation, i.e. the number of correct and incorrect outputs.
#[derive(Debug)]
pub struct EvaluationResult {
    corrects: usize,
    incorrects: usize
}

impl EvaluationResult {
    pub fn new(corrects: usize, incorrects: usize) -> Self {
        EvaluationResult { corrects: corrects, incorrects: incorrects }
    }
    
    pub fn test_data_length(&self) -> usize {
        self.corrects + self.incorrects
    }

    /// Computes the number of correct outputs divided by the total number of outputs.
    pub fn accuracy(&self) -> Option<f64> {
        let l = self.test_data_length();
        if l == 0 {
            return None
        } else {
            return Some (self.corrects as f64 / l as f64)
        }
    }
}