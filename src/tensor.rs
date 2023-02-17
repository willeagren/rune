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
// Last updated: 2023-02-17
//

use ndarray::ArrayD;
use ndarray::IxDyn;
use crate::Context;

#[allow(dead_code)]
#[derive(Debug)]
pub struct Tensor<'a> {
    dims: &'a [usize], 
    data: ArrayD<f32>,
    ctx: Context<'a>,
    requires_grad: bool,
}

#[allow(dead_code)]
impl<'a> Tensor<'a> {

    pub fn from_numpy(
        data: &'a ArrayD<f32>,
    ) -> Self {
        Tensor {
            dims: data.shape(),
            data: data.clone(),
            ctx: Context::new(),
            requires_grad: false,
        }
    }

    pub fn zeros(
        dims: &'a [usize], 
    ) -> Self {
        Tensor {
            dims: &dims,
            data: ArrayD::<f32>::zeros(IxDyn(&dims)),
            ctx: Context::new(),
            requires_grad: false,
        }
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construct_tensor() {
        let t: Tensor = Tensor::from_numpy(
            &ArrayD::<f32>::zeros(IxDyn(&[3, 2])),
        );

        let z: Tensor = Tensor::zeros(
            &[2, 3],
        );
    }

}

