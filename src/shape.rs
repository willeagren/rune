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
// File created: 2023-03-06
// Last updated: 2023-03-06
//

#[derive(Clone, Debug, Default)]
pub struct Shape
{
    dims: Vec<usize>,
}

impl Shape
{
    pub fn new(dims: &[usize]) -> Self
    {
        Shape { dims: dims.to_vec() }
    }

    pub fn none() -> Self
    {
        Shape { ..Default::default() }
    }

    fn dims(&self) -> Vec<usize>
    {
        self.dims.clone()
    }

    fn set_dims(&mut self, dims: &[usize])
    {
        self.dims = dims.to_vec();
    }
}

impl PartialEq for Shape
{
    fn eq(&self, other: &Self) -> bool
    {
        self.dims == other.dims
    }
}

impl Eq for Shape {}

#[cfg(test)]
mod tests
{
    use super::*;

    #[test]
    fn constructors()
    {
        let a = Shape::new(&[1024, 3, 256, 256]);
        let b = Shape::new(&[2048, 64, 32, 32]);
        let c = Shape::none();
        let d = Shape::new(&[1024, 3, 256, 256]);

        assert_eq!(a, d);
        assert_eq!(b.dims().len(), 4);
        assert_eq!(*c.dims(), Vec::new());
    }

    #[test]
    fn setter()
    {
        let mut a = Shape::none();
        a.set_dims(&[10, 40, 29]);

        assert_eq!(*a.dims(), vec![10, 40, 29]);
        assert_ne!(*a.dims(), Vec::new());
    }
}

