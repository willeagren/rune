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

use numpy::PyReadonlyArrayDyn;
use numpy::PyArrayDyn;
use numpy::IntoPyArray;

use ndarray::Array;
use ndarray::ArrayD;
use ndarray::IxDyn;

use pyo3::prelude::*;
use pyo3::types::PyType;
use pyo3::types::PyList;

#[pyclass]
#[derive(Debug, Clone)]
pub struct Tensor
{
    shape: Shape,
    data: ArrayD<f32>,
    parents: Vec<Box<Tensor>>,
    requires_grad: bool,
}

impl Default for Tensor
{
    fn default() -> Self
    {
        Tensor
        {
            shape: Shape::none(),
            data: ArrayD::<f32>::zeros(IxDyn(&[1])),
            parents: Vec::new(),
            requires_grad: false,
        }
    }
}

#[pymethods]
impl Tensor
{
    #[new]
    pub fn new(np_arr: PyReadonlyArrayDyn<f32>) -> Self
    {
        let shape_slice = np_arr.shape();
        let data_vec = match np_arr.as_slice()
        {
            Ok(raw) => raw.to_vec(),
            Err(e) => panic!("Could not create immutable view of internal data, {}.", e),
        };
        let data: ArrayD<f32> = Array::from_vec(data_vec).into_dyn();
        let data = match data.into_shape(shape_slice)
        {
            Ok(shaped_arrayd) => shaped_arrayd,
            Err(e) => panic!("Could not reshape ArrayBase to {:?}, {}.", shape_slice, e),
        };
        let shape = Shape::new(shape_slice.to_vec());
        Tensor
        { 
            shape: shape,
            data: data,
            ..Default::default()
        }
    }

    #[classmethod]
    pub fn zeros(cls: &PyType, dims: &PyList) -> Self
    {
        let dims: Vec<usize> = dims.extract().expect("Could not extract Vec<T> from &PyList");
        let data = ArrayD::<f32>::zeros(IxDyn(&dims));
        Tensor
        {
            shape: Shape::new(dims),
            data: data,
            ..Default::default()
        }
    }

    #[classmethod]
    pub fn ones(cls: &PyType, dims: &PyList) -> Self
    {
        let dims: Vec<usize> = dims.extract().expect("Could not extract Vec<T> from &PyList");
        let data = ArrayD::<f32>::ones(IxDyn(&dims));
        Tensor
        {
            shape: Shape::new(dims),
            data: data,
            ..Default::default()
        }
    }

    fn detach<'py>(&self, py: Python<'py>) -> &'py PyArrayDyn<f32>
    {
        self.data.clone().into_pyarray(py)
    }

    /*
    fn shape(&self) -> &Shape
    {
        &self.shape
    }

    fn data(&self) -> &ArrayD<f32>
    {
        &self.data
    }

    fn parents(&self) -> &Vec<Box<Tensor>>
    {
        &self.parents
    }
*/

    fn requires_grad(&self) -> bool
    {
        self.requires_grad
    }

    /*
    fn sub(self, other: Tensor) -> Tensor
    {
        let data = &self.data - &other.data;
        let dims = Shape::new(data.shape().to_vec());
        let requires_grad = any_requires_grad(vec![&self, &other]);
        Tensor
        {
            shape: dims,
            data: data,
            parents: vec![Box::new(self), Box::new(other)],
            requires_grad: requires_grad,
        }
    }

    fn mul(self, other: Tensor) -> Tensor
    {
        let data = &self.data * &other.data;
        let dims = Shape::new(data.shape().to_vec());
        let requires_grad = any_requires_grad(vec![&self, &other]);
        Tensor
        {
            shape: dims,
            data: data,
            parents: vec![Box::new(self), Box::new(other)],
            requires_grad: requires_grad,
        }
    }

    fn div(self, other: Tensor) -> Tensor
    {
        let data = &self.data / &other.data;
        let dims = Shape::new(data.shape().to_vec());
        let requires_grad = any_requires_grad(vec![&self, &other]);
        Tensor
        {
            shape: dims,
            data: data,
            parents: vec![Box::new(self), Box::new(other)],
            requires_grad: requires_grad,
        }
    }
    */
}
/*

    fn set_requires_grad(&mut self, requires: bool)
    {
        self.requires_grad = requires
    }
}
*/
 
/// Return true if any of the tensors requires gradient.
pub fn any_requires_grad (tensors: Vec<&Tensor>) -> bool
{
    tensors.iter().any(|&t| t.requires_grad())
}

impl Tensor 
{
    pub fn add(self, other: Tensor) -> Self
    {
        let data = &self.data + &other.data;
        let dims = Shape::new(data.shape().to_vec());
        let requires_grad = any_requires_grad(vec![&self, &other]);
        Tensor
        {
            shape: dims,
            data: data,
            parents: vec![Box::new(self), Box::new(other)],
            requires_grad: requires_grad,
        }
    }
}

/*
#[cfg(test)]
mod tests
{
    use super::*;
    use ndarray::ArrayD;
    use ndarray::IxDyn;

    #[test]
    fn ops_add()
    {
        let mut a = Tensor::new(ArrayD::<f32>::ones(IxDyn(&[128, 1024])));
        let b = Tensor::ones(&[128, 1]);
        a.set_requires_grad(true);

        let c = a.add(b);
        let d = 2f32 * ArrayD::<f32>::ones(IxDyn(&[128, 1024]));
        assert_eq!(*c.data(), d);

        let e = c.add(Tensor::new(d));
        let f = 4f32 * ArrayD::<f32>::ones(IxDyn(&[128, 1024]));
        assert_eq!(*e.data(), f);
    }

    #[test]
    fn ops_sub()
    {
        let a = Tensor::new(ArrayD::<f32>::ones(IxDyn(&[32, 3, 256, 256])));
        let b = Tensor::ones(&[32, 3, 256, 1]);

        let c = a.sub(b);
        let d = Tensor::zeros(&[32, 3, 256, 256]);
        assert_eq!(*c.data(), *d.data());

    }

    #[test]
    fn ops_mul()
    {
        let a = Tensor::new(ArrayD::<f32>::ones(IxDyn(&[128, 1024])));
        let b = Tensor::ones(&[128, 1024]);

        let c = a.mul(b);
        let d = Tensor::ones(&[128, 1024]);
        assert_eq!(*c.data(), *d.data());
    }

    #[test]
    fn ops_div()
    {
        let mut a = Tensor::new(ArrayD::<f32>::ones(IxDyn(&[128, 1024])));
        let b = Tensor::ones(&[128, 1024]);
        a.set_requires_grad(true);

        let c = a.div(b);
        let d = ArrayD::<f32>::ones(IxDyn(&[128, 1024]));
        assert_eq!(c.requires_grad(), true);
        assert_eq!(*c.data(), d);
    }
}
*/

