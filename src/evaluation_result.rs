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

    pub fn accuracy(&self) -> Option<f64> {
        let l = self.test_data_length();
        if l == 0 {
            return None
        } else {
            return Some (self.corrects as f64 / l as f64)
        }
    }
}