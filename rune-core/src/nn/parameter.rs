//
// MIT License
// 
// Copyright (c) 2023 Wilhelm Ågren
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
// 
// File created: 2023-03-12
// Last updated: 2023-03-12
//

use crate::datatype::DataType;
use crate::tensor::Tensor;


use ndarray::ArrayD;

use ndarray_rand::rand_distr::Distribution;
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::rand_distr::Uniform;

pub struct Parameter {}

impl Parameter
{
    pub fn new<'a, T>(data: ArrayD<T>) -> Tensor<'a, T>
    where T: DataType
    {
        let mut parameter = Tensor::new(data);
        parameter.set_requires_grad(true);
        parameter
    }

    pub fn uniform<'a, T>(dims: &[usize], low: f32, high: f32) -> Tensor<'a, T>
    where T: DataType, Uniform<f32>: Distribution<T>
    {
        let mut parameter = Tensor::<T>::uniform(dims, low, high);
        parameter.set_requires_grad(true);
        parameter
    }

    pub fn normal<'a, T>(dims: &[usize], mu: f32, sigma: f32) -> Tensor<'a, T>
    where T: DataType, Normal<f32>: Distribution<T>
    {
        let mut parameter = Tensor::<T>::normal(dims, mu, sigma);
        parameter.set_requires_grad(true);
        parameter
    }
}

