
use opencl3::command_queue::CommandQueue;
use opencl3::context::Context;
use opencl3::device::Device;
use opencl3::kernel::Kernel;
use opencl3::memory::Buffer;
use opencl3::program::Program;
use opencl3::types::cl_float;

use pyo3::prelude::*;

const EXAMPLE_PROGRAM: &str = r#"
kernel void __scalarsum (global float* z,
    global float const* x,
    global float const* y,
    float a)
{
    const size_t i = get_global_id(0);
    z[i] = a * x[i] + y[i];
}"#;

#[pyclass]
pub struct CLBuffer {
    device: Device,
    context: Context,
    queue: CommandQueue,
    program: Program,
    kernel: Kernel,
}

