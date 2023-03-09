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
// File created: 2023-03-09
// Last updated: 2023-03-09
//

use crate::datatype::DataType;
use crate::shape::Shape;

use ndarray::ArrayD;
use ndarray::IxDyn;

pub struct Tensor<'a, T: DataType>
{
    shape: Shape,
    data: ArrayD<T>,
    parents: Vec<&'a Tensor<'a, T>>,
    requires_grad: bool,
}

/// TODO how should we create a default ArrayD::<T>::?
impl<T: DataType> Default for Tensor<'_, T>
{
    fn default() -> Self
    {
        Tensor
        {
            shape: Shape::none(),
            data: ArrayD::<T>::zeros(IxDyn(&[1])),
            parents: Vec::new(),
            requires_grad: false,
        }
    }
}

impl<T: DataType> Tensor<'_, T>
{
    pub fn new(data: ArrayD<T>) -> Self
    {
        let shape = Shape::new(data.shape());
        Tensor
        {
            shape: shape,
            data: data,
            ..Default::default()
        }
    }

    pub fn zeros(dims: &[usize]) -> Self
    {
        let data = ArrayD::<T>::zeros(IxDyn(dims));
        Tensor
        {
            shape: Shape::new(dims),
            data: data,
            ..Default::default()
        }
    }

    pub fn ones(dims: &[usize]) -> Self
    {
        let data = ArrayD::<T>::ones(IxDyn(dims));
        Tensor
        {
            shape: Shape::new(dims),
            data: data,
            ..Default::default()
        }
    }
}

