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
// File created: 2023-03-06
// Last updated: 2023-03-06
//

use crate::shape::Shape;
use crate::datatype::DataType;

use ndarray::ArrayD;
use ndarray::IxDyn;

#[derive(Debug)]
pub struct Tensor<'a, T: DataType>
{
    shape: Shape,
    data: ArrayD<T>,
    parents: Vec<&'a Tensor<'a, T>>,
    requires_grad: bool,
}

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

    fn shape(&self) -> &Shape
    {
        &self.shape
    }

    fn data(&self) -> &ArrayD<T>
    {
        &self.data
    }

    fn parents(&self) -> &Vec<&'_ Tensor<'_, T>>
    {
        &self.parents
    }

    fn requires_grad(&self) -> bool
    {
        self.requires_grad
    }

    fn set_requires_grad(&mut self, requires: bool)
    {
        self.requires_grad = requires
    }
}

/// Return true if any of the tensors requires gradient.
pub fn any_requires_grad<T: DataType>
(tensors: Vec<&Tensor<T>>) -> bool
{
    tensors.iter().any(|&t| t.requires_grad())
}

/// binary ops
impl<'a, T: DataType> Tensor<'a, T>
{
    fn add(&'a self, other: &'a Tensor<T>) -> Tensor<'a, T>
    {
        let data = &self.data + &other.data;
        let dims = Shape::new(data.shape());
        let requires_grad = any_requires_grad(vec![self, other]);
        Tensor
        {
            shape: dims,
            data: data,
            parents: vec![self, other],
            requires_grad: requires_grad,
        }
    }
    fn sub(&'a self, other: &'a Tensor<T>) -> Tensor<'a, T>
    {
        let data = &self.data - &other.data;
        let dims = Shape::new(data.shape());
        let requires_grad = any_requires_grad(vec![self, other]);
        Tensor
        {
            shape: dims,
            data: data,
            parents: vec![self, other],
            requires_grad: requires_grad,
        }
    }

    fn mul(&'a self, other: &'a Tensor<T>) -> Tensor<'a, T>
    {
        let data = &self.data * &other.data;
        let dims = Shape::new(data.shape());
        let requires_grad = any_requires_grad(vec![self, other]);
        Tensor
        {
            shape: dims,
            data: data,
            parents: vec![self, other],
            requires_grad: requires_grad,
        }
    }
}

#[cfg(test)]
mod tests
{
    use super::*;
    use ndarray::ArrayD;
    use ndarray::IxDyn;

    #[test]
    fn ops()
    {
        let mut a = Tensor::<f32>::new(ArrayD::<f32>::ones(IxDyn(&[128, 1024])));
        let b = Tensor::<f32>::ones(&[128, 1]);
        let c = Tensor::<f32>::zeros(&[1, 1024]);

        let d = a.add(&b);
        let e = a.add(&c);

        let f = ArrayD::<f32>::ones(IxDyn(&[128, 1024]));
        assert_eq!(f, *e.data());

        let j = Tensor::<f32>::new(f);

        a.set_requires_grad(true);
        let mut g = a.sub(&b);

        assert_eq!(g.requires_grad(), true);
        g.set_requires_grad(false);
        assert_eq!(g.requires_grad(), false);

        let h = ArrayD::<f32>::zeros(IxDyn(&[128, 1024]));
        assert_eq!(h, *g.data());

        let i = j.mul(&c);
        assert_eq!(h, *i.data());
    }

    #[test]
    fn grad()
    {
        let mut a = Tensor::<f32>::new(ArrayD::<f32>::ones(IxDyn(&[128, 1024])));
        let mut b = Tensor::<f32>::ones(&[128, 1]);
        b.set_requires_grad(true);

        let c = a.add(&b);

        assert_eq!(c.requires_grad(), true);
        assert_eq!(c.parents().len(), 2);
    }
}

