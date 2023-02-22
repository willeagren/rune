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
// File created: 2023-02-16
// Last updated: 2023-02-22
//

use std::fmt::Debug;
use num_traits::{Float, ToPrimitive, NumCast};

use crate::functions::Function;
use crate::functions::BinaryOp;
use crate::Size;

pub trait DataType: Copy + Debug + Float + ToPrimitive + 'static{}

impl DataType for f32 {}
impl DataType for f64 {}

#[derive(Debug, Clone)]
pub struct Tensor<'a, T: DataType> {
    dims: Size,
    data: Vec<T>,
    ctx: Function<'a, T>,
    requires_grad: bool,
}

#[allow(dead_code)]
impl<T: DataType> Tensor<T> {
    pub fn zeros(d: &[usize]) -> Self {
        let shape = Size::new(d);
        let num_elements = shape.num_elements();
        let mut data = Vec::<T>::with_capacity(num_elements);

        for _ in 0..num_elements {
            let item = match NumCast::from(0.0) {
                Some(i) => i,
                None => panic!("Could not cast 0 to generic type <T: DataType>."),
            };
            data.push(item);
        }
        
        Tensor { 
            dims: shape,
            data: data,
            ctx: Function::none(),
            requires_grad: false,
        }
    }

    pub fn ones(d: &[usize]) -> Self {
        let shape = Size::new(d);
        let num_elements = shape.num_elements();
        let mut data = Vec::<T>::with_capacity(num_elements);

        for _ in 0..num_elements {
            let item = match NumCast::from(1.0) {
                Some(i) => i,
                None => panic!("Could not cast 1 to generic type <T: DataType>."),
            };
            data.push(item);
        }

        Tensor { 
            dims: shape,
            data: data,
            ctx: Function::none(),
            requires_grad: false,
        }
    }

    pub fn dims(&self) -> &Size {
        &self.dims
    }

    pub fn data(&self) -> &Vec<T> {
        &self.data
    }

    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }
}

impl<T: DataType> BinaryOp for Tensor<T> {
    type Other = Self;
    type Output = Self;

    fn add(&self, other: Self::Other) -> Self::Output {
        let shape = self.dims.clone();
        let data: Vec<T> = self.data.iter()
            .zip(other.data.iter())
            .map(|(&u, &v)| u + v)
            .collect();
        Tensor {
            dims: shape,
            data: data,
            ctx: Function::new(vec![&self, &other]),
            requires_grad: false,
        }
    }

    fn sub(&self, other: Self::Other) -> Self::Output {
        let shape = self.dims.clone();
        let data: Vec<T> = self.data.iter()
            .zip(other.data.iter())
            .map(|(&u, &v)| u - v)
            .collect();
        Tensor {
            dims: shape,
            data: data,
            ctx: Function::new(vec![&self, &other]),
            requires_grad: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add() {
        let a = Tensor::<f32>::ones(&[128, 4, 256, 256]);
        let b = Tensor::<f32>::zeros(&[128, 4, 256, 256]);
        let c = a.add(b);

        assert_eq!(
            c.dims().to_vec(),
            Size::new(&[128, 4, 256, 256]).to_vec(),
        );
    }
}

