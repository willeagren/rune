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
// File created: 2023-02-17
// Last updated: 2023-02-17
//

use std::vec::Vec;

use crate::Tensor;
use crate::function::Function;

#[allow(dead_code)]
#[derive(Debug)]
pub struct Context<'a> {
    parents: Vec<&'a Tensor<'a>>,
    saved_tensors: Vec<&'a Tensor<'a>>,
    grad_fn: Function,
}

impl<'a> Default for Context<'a> {
    fn default() -> Self {
        Context {
            parents: vec![],
            saved_tensors: vec![],
            grad_fn: Function::new(),
        }
    }
}

impl<'a> Context<'a> {

    pub fn new() -> Self {
        Context { ..Default::default() }
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new() {
        let ctx: Context = Context::new();
    }

}
