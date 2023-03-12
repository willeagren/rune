use crate::datatype::DataType;
use crate::nn::parameter::Parameter;
use crate::tensor::Tensor;

use ndarray_rand::rand_distr::Distribution;
use ndarray_rand::rand_distr::Uniform;

pub struct Module<'a, T: DataType>
{
    parameters: Vec<Tensor<'a, T>>
}

impl<'a, T: DataType> Module<'a, T>
{
    pub fn affine(fan_in: usize, fan_out: usize) -> Self
    where Uniform<f32>: Distribution<T>
    {
        let mut weights = Parameter::uniform::<T>(&[fan_in, fan_out], -1.0, 1.0);
        let mut bias = Parameter::uniform(&[fan_out, 1], -1.0, 1.0);
        Module
        {
            parameters: vec![weights, bias],
        }
    }

    /*
    pub fn forward(&'a self, t: &'a Tensor<'a, T>) -> Tensor<'a, T>
    {
        Tensor::zeros(&[784, 1]).add(&self.parameters[1])

        // let r = t.dot(&self.parameters[0]);
        // r = r.add(&self.parameters[1]);
    }
    */
}

