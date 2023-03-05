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
// File created: 2023-03-05
// Last updated: 2023-03-05
//

use ndarray::ArrayD;
use ndarray::Array;

use numpy::PyReadonlyArrayDyn;
use numpy::PyArrayDyn;
use numpy::IntoPyArray;

use pyo3::prelude::*;
use pyo3::PyNumberProtocol;
use pyo3::PyObjectProtocol;
use pyo3::Python;

#[pyclass]
#[derive(Default, Clone, Debug)]
pub struct RBuffer {
    shape: Vec<usize>,
    data: ArrayD<f32>,
}

#[pymethods]
impl RBuffer {
    #[new]
    fn new(arr: PyReadonlyArrayDyn<f32>) -> Self {
        let slice_shape = arr.shape();
        let shape = slice_shape.to_vec();
        let data_vec = match arr.as_slice() {
            Ok(raw) => raw.to_vec(),
            Err(_) => panic!("Could not create immutable view of internal data."),
        };
        let data: ArrayD<f32> = Array::from_vec(data_vec).into_dyn();
        let data = match data.into_shape(slice_shape) {
            Ok(shaped) => shaped,
            Err(e) => panic!("Could not reshape ArrayBase to {:?}, {}", slice_shape, e),
        };
        RBuffer { shape: shape, data: data }
    }

    fn detach<'py>(&self, py: Python<'py>) -> &'py PyArrayDyn<f32> {
        self.data.clone().into_pyarray(py)
    }
}

#[pyproto]
impl PyNumberProtocol for RBuffer {
    fn __add__(lhs: RBuffer, rhs: RBuffer) -> PyResult<RBuffer> {
        let shape: Vec<usize> = lhs.shape.clone();
        let data: ArrayD<f32> = lhs.data + rhs.data;
        Ok(RBuffer { shape: shape, data: data })
    }

    fn __sub__(lhs: RBuffer, rhs: RBuffer) -> PyResult<RBuffer> {
        let shape: Vec<usize> = lhs.shape.clone();
        let data: ArrayD<f32> = lhs.data - rhs.data;
        Ok(RBuffer { shape: shape, data: data })
    }

    fn __truediv__(lhs: RBuffer, rhs: RBuffer) -> PyResult<RBuffer> {
        let shape: Vec<usize> = lhs.shape.clone();
        let data: ArrayD<f32> = lhs.data / rhs.data;
        Ok(RBuffer { shape: shape, data: data })
    }

    fn __mul__(lhs: RBuffer, rhs: RBuffer) -> PyResult<RBuffer> {
        let shape: Vec<usize> = lhs.shape.clone();
        let data: ArrayD<f32> = lhs.data * rhs.data;
        Ok(RBuffer { shape: shape, data: data })
    }
}

#[pyproto]
impl PyObjectProtocol for RBuffer {
    fn __str__(&self) -> PyResult<String> {
        Ok(format!("{:?}", self.data))
    }
}

