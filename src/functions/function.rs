
use crate::Tensor;
use crate::tensor::DataType;

#[derive(Debug, Clone)]
pub struct Function<'a, T: DataType> {
    parents: Vec<&'a Tensor<'a, T>>,
}

impl<T: DataType> Function<'a, T> {
    pub fn none() -> Self {
        Function { parents: Vec::new() }
    }

    pub fn new(p: Vec<&'a ensor<T>>) -> Self {
        Function { parents: p }
    }

    pub fn requires_grad(&self) -> bool {
        self.parents.iter()
            .any(|&t| t.requires_grad())
    }
}

