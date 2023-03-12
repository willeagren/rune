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

pub mod datatype;
pub mod nn;
pub mod shape;
pub mod tensor;
pub mod utils;

#[cfg(test)]
mod tests
{
    use crate::tensor::Tensor;
    use crate::nn::module::Module;

    #[test]
    fn backward()
    {
        let x = Tensor::<f32>::ones(&[128, 128]);
        let w = Tensor::<f32>::uniform(&[128, 500], -1.0, 1.0);
        let b = Tensor::<f32>::uniform(&[1, 500], -1.0, 1.0);
        let r = x.matmul(&w);
        println!("{:?} = {:?} @ {:?}", r.shape().dims(), x.shape().dims(), w.shape().dims());
        let y = r.add(&b);
    }
}

