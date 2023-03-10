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
// Last updated: 2023-03-06
//

mod datatype;
mod rbuffer;
mod utils;
mod shape;
mod tensor;

use numpy::PyReadonlyArrayDyn;
use numpy::PyArrayDyn;
use numpy::IntoPyArray;

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::Python;

#[pyfunction]
pub fn add<'py>(
    py: Python<'py>, 
    x: PyReadonlyArrayDyn<f32>, 
    y: PyReadonlyArrayDyn<f32>,
) -> &'py PyArrayDyn<f32>
{
    let x = &x.as_array();
    let y = &y.as_array();
    (x + y).into_pyarray(py)
}

#[pyfunction]
pub fn sub<'py>(
    py: Python<'py>, 
    x: PyReadonlyArrayDyn<f32>, 
    y: PyReadonlyArrayDyn<f32>,
) -> &'py PyArrayDyn<f32>
{
    let x = &x.as_array();
    let y = &y.as_array();
    (x - y).into_pyarray(py)
}

#[pyfunction]
pub fn mul<'py>(
    py: Python<'py>,
    x: PyReadonlyArrayDyn<f32>,
    y: PyReadonlyArrayDyn<f32>,
) -> &'py PyArrayDyn<f32>
{
    let x = &x.as_array();
    let y = &y.as_array();
    (x * y).into_pyarray(py)
}

#[pyfunction]
pub fn tadd<'py>(py: Python<'py>, x: tensor::Tensor, y: tensor::Tensor)
-> tensor::Tensor
{
    x.add(y)
}

#[pymodule]
fn rune(_py: Python<'_>, m: &PyModule) -> PyResult<()>
{
    m.add_wrapped(wrap_pyfunction!(add))?;
    m.add_wrapped(wrap_pyfunction!(sub))?;
    m.add_wrapped(wrap_pyfunction!(mul))?;
    m.add_wrapped(wrap_pyfunction!(tadd))?;
    m.add_class::<rbuffer::RBuffer>()?;
    m.add_class::<tensor::Tensor>()?;
    Ok(())
}

