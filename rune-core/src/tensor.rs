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
// Last updated: 2023-03-12
//

use crate::datatype::DataType;
use crate::shape::Shape;
use crate::utils::*;

use ndarray::Array0;
use ndarray::Array2;
use ndarray::ArrayD;
use ndarray::Ix0;
use ndarray::Ix2;
use ndarray::IxDyn;
use ndarray::prelude::s;

use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Distribution;
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::rand_distr::Uniform;

pub struct Tensor<'a, T: DataType>
{
    shape: Shape,
    data: ArrayD<T>,
    parents: Vec<&'a Tensor<'a, T>>,
    requires_grad: bool,
    dtype: T,
    grad: ArrayD<T>,
}

///
/// This will create a scalar tensor holding a single value. This value will 
/// be the deault value of the specified type T. We can not create an empty
/// ArrayD using the ndarray crate. The smallest tensor we can have is thus
/// the scalar values. 
///
/// # Example
///
/// let t = Tensor::<f32>::default();
/// println!("[{:?}]", *t.data());
///
/// >>> [0.0, shape=[], strides=[], layout=CFcf (0xf), dynamic ndim=0]
///
impl<T: DataType> Default for Tensor<'_, T>
{
    fn default() -> Self
    {
        Tensor
        {
            shape: Shape::none(),
            data: Array0::<T>::zeros(Ix0()).into_dyn(),
            parents: Vec::new(),
            requires_grad: false,
            dtype: T::default(),
            grad: Array0::<T>::zeros(Ix0()).into_dyn(),
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

    pub fn uniform(dims: &[usize], low: f32, high: f32) -> Self
    where Uniform<f32>: Distribution<T>
    {
        let dist = Uniform::new(low, high);
        let data = ArrayD::<T>::random(dims, dist);
        Tensor
        {
            shape: Shape::new(dims),
            data: data,
            ..Default::default()
        }
    }

    pub fn normal(dims: &[usize], mu: f32, sigma: f32) -> Self
    where Normal<f32>: Distribution<T>
    {
        let dist = match Normal::new(mu, sigma)
        {
            Ok(dist) => dist,
            Err(e) => panic!("Provided variance is not finite, {:?}", e),
        };
        let data = ArrayD::<T>::random(dims, dist);
        Tensor
        {
            shape: Shape::new(dims),
            data: data,
            ..Default::default()
        }
    }

    pub fn shape(&self) -> &Shape
    {
        &self.shape
    }

    pub fn data(&self) -> &ArrayD<T>
    {
        &self.data
    }

    pub fn requires_grad(&self) -> bool
    {
        self.requires_grad
    }

    pub fn set_requires_grad(&mut self, requires_grad: bool)
    {
        self.requires_grad = requires_grad;
        if requires_grad
        {
            let dims = self.shape.dims().as_slice();
            self.grad = ArrayD::<T>::zeros(IxDyn(dims));
        }
    }

    pub fn dtype(&self) -> T
    {
        self.dtype
    }
}

///
/// Lifetime dependent struct functions.
///
impl<'a, T: DataType> Tensor<'a, T>
{
    fn parents(&self) -> &Vec<&'a Tensor<'a, T>>
    {
        &self.parents
    }
}

///
/// Binary ops
///
impl<'a, T: DataType + 'static> Tensor<'a, T>
{
    pub fn add(&'a self, other: &'a Tensor<T>) -> Tensor<'a, T>
    {
        let data = &self.data + &other.data;
        let dims = Shape::new(data.shape());
        let requires_grad = any_requires_grad(vec![self, other]);
        let parents = vec![self, other];
        let grad = ArrayD::<T>::zeros(data.shape());
        Tensor
        {
            shape: dims,
            data: data,
            parents: parents,
            requires_grad: requires_grad,
            dtype: T::default(),
            grad: grad,
        }
    }

    pub fn sub(&'a self, other: &'a Tensor<T>) -> Tensor<'a, T>
    {
        let data = &self.data - &other.data;
        let dims = Shape::new(data.shape());
        let requires_grad = any_requires_grad(vec![self, other]);
        let parents = vec![self, other];
        let grad = ArrayD::<T>::zeros(data.shape());
        Tensor
        {
            shape: dims,
            data: data,
            parents: parents,
            requires_grad: requires_grad,
            dtype: T::default(),
            grad: grad,
        }
    }

    pub fn mul(&'a self, other: &'a Tensor<T>) -> Tensor<'a, T>
    {
        let data = &self.data * &other.data;
        let dims = Shape::new(data.shape());
        let requires_grad = any_requires_grad(vec![self, other]);
        let parents = vec![self, other];
        let grad = ArrayD::<T>::zeros(data.shape());
        Tensor
        {
            shape: dims,
            data: data,
            parents: parents,
            requires_grad: requires_grad,
            dtype: T::default(),
            grad: grad,
        }
    }

    pub fn div(&'a self, other: &'a Tensor<T>) -> Tensor<'a, T>
    {
        let data = &self.data / &other.data;
        let dims = Shape::new(data.shape());
        let requires_grad = any_requires_grad(vec![self, other]);
        let parents = vec![self, other];
        let grad = ArrayD::<T>::zeros(data.shape());
        Tensor
        {
            shape: dims,
            data: data,
            parents: parents,
            requires_grad: requires_grad,
            dtype: T::default(),
            grad: grad,
        }
    }

    ///
    /// output (i, k) = (i, j) @ (j, k)
    ///
    /// a = (i, j)
    /// b = (j, k)
    /// t = zeros(i, k)
    ///
    /// for i in range 0..i
    ///     for j in range 0..k
    ///         for k in range 0..j
    ///             t[i, j] += a[i, k] * b[k, j]
    /// return t
    ///
    pub fn matmul(&'a self, other: &'a Tensor<T>) -> Tensor<'a, T>
    {
        let lhs_dims = self.shape.dims();
        let rhs_dims = other.shape.dims();
        let dims = &[lhs_dims[0], rhs_dims[1]];
        let mut data = ArrayD::<T>::zeros(IxDyn(dims));

        for i in 0..lhs_dims[0]
        {
            for j in 0..rhs_dims[1] 
            {
                for k in 0..lhs_dims[1]
                {
                    data[[i, j]] += self.data()[[i, k]] + other.data()[[k, j]];
                }
            }
        }

        let dims = Shape::new(dims);
        let requires_grad = any_requires_grad(vec![self, other]);
        let parents = vec![self, other];
        let grad = ArrayD::<T>::zeros(data.shape());
        Tensor
        {
            shape: dims,
            data: data,
            parents: parents,
            requires_grad: requires_grad,
            dtype: T::default(),
            grad: grad,
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
    fn constructors()
    {
        let _a = Tensor::new(ArrayD::<f32>::zeros(IxDyn(&[128, 3, 256, 256])));
        let _b = Tensor::<f32>::default();
        let _c = Tensor::<f32>::ones(&[128, 3, 256, 256]);
        let _d = Tensor::<f64>::zeros(&[128, 784]);

        let _e = Tensor::uniform(&[128, 3, 256, 256], -1.0, 1.0);
        let _f = Tensor::normal(&[128, 3, 256, 256], 0.0, 3.1415);
    }

    #[test]
    fn shapes()
    {
        let a = Tensor::<f32>::default();
        let b = Tensor::<f32>::ones(&[3, 5, 19, 3]);
        let c = Tensor::<f64>::zeros(&[3, 5, 19, 3]);

        assert_eq!(b.shape(), c.shape());
        assert_eq!(*a.shape().dims(), Vec::<usize>::new());
    }

    #[test]
    fn requires_grad()
    {
        let a = Tensor::<f32>::default();
        let b = Tensor::<f32>::ones(&[128, 3, 256, 256]);
        let mut c = Tensor::<f32>::ones(&[128, 3, 256, 256]);
        c.set_requires_grad(true);

        assert_eq!(any_requires_grad(vec![&a, &b]), false);

        let d = b.add(&c);
        assert_eq!(any_requires_grad(d.parents().to_vec()), true);
    }

    #[test]
    fn dtypes()
    {
        let a = Tensor::<f32>::default();
        let b = Tensor::<f64>::default();

        assert_ne!(type_of(a.dtype()), type_of(b.dtype()));
    }
}
