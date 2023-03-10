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

use std::any::type_name;

pub fn any_requires_grad<T>(tensors: Vec<&Tensor<T>>) -> bool
where T: DataType
{
    tensors.iter().any(|&t| t.requires_grad())
}

pub fn type_of<T>(_: T) -> &'static str
where T: DataType
{
    type_name::<T>()
}

pub fn vec_to_array<T, const N: usize>(v: Vec<T>) -> [T; N]
{
    v.try_into().unwrap_or_else(
        |v: Vec<T>| panic!("Expected Vec<T> of length {} but got {}", N, v.len())
    )
}

