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
// File created: 2023-03-03
// Last updated: 2023-03-04
//

use numpy::PyReadonlyArrayDyn;

use pyo3::prelude::*;
use pyo3::PyNumberProtocol;

#[derive(Clone)]
#[pyclass(module = "rune")]
pub struct Buffer {
    shape: Vec<usize>,
    data: Vec<f32>,
}

#[pymethods]
impl Buffer {
    #[new]
    fn new(arr: PyReadonlyArrayDyn<f32>) -> Self {
        let shape = arr.shape().to_vec();
        let data = match arr.as_slice() {
            Ok(raw) => raw.to_vec(),
            Err(_) => panic!("Could not create immutable view of internal data."),
        };
        Buffer { shape: shape, data: data }
    }
}

#[pyproto]
impl PyNumberProtocol for Buffer {
    fn __add__(lhs: Buffer, rhs: Buffer) -> PyResult<Buffer> {
        let shape: Vec<usize> = lhs.shape.clone();
        let data: Vec<f32> = lhs.data.iter()
            .zip(rhs.data.iter())
            .map(|(&a, &b)| a + b)
            .collect();
        Ok(Buffer { shape: shape, data: data })
    }
}

