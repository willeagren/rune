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
// Last updated: 2023-03-09
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

    fn dims(&self) -> &Vec<usize>
    {
        &self.dims
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

/// 
/// Iterate over the dim vector in the Shape struct.
/// Moves the .dims Vec<usize> to the caller.
///
/// let shape = Shape::new(&[128, 3, 6, 129, 129]);
/// let mut num_dims = 0;
/// for dim in shape.into_iter()
/// {
///     num_dims += dim;
/// }
///
impl IntoIterator for Shape
{
    type Item = usize;
    type IntoIter = <Vec<usize> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter 
    {
        self.dims.into_iter()
    }
}

///
/// FUNCTIONAL UNIT TESTING
///
#[cfg(test)]
mod tests
{
    use super::*;
    
    #[test]
    fn constructor()
    {
        let a = Shape::new(&[1024, 3, 128, 128]);
        let b = Shape::new(&[1024, 3, 128, 128]);
        let c = Shape::new(&[128, 784]);
        let d = Shape::none();

        assert_eq!(a, b);
        assert_eq!(c.dims().len(), 2);
        assert_eq!(*d.dims(), Vec::new());
    }

    #[test]
    fn setter()
    {
        let mut a = Shape::new(&[16, 256, 256]);
        assert_eq!(*a.dims(), vec![16, 256, 256]);

        a.set_dims(&[128, 16, 256, 256]);
        assert_eq!(a.dims().len(), 4);
    }

    #[test]
    fn iter()
    {
        let mut a = Shape::new(&[4, 3, 5, 10, 239]);
        let dims = [128, 3, 256, 256];
        a.set_dims(&dims);

        // Because we move the dims vec to the caller when calling
        // into_iter() we need to clone if we want to use it after.
        for (x, y) in a.clone().into_iter().zip(dims.into_iter())
        {
            assert_eq!(x, y);
        }

        let mut num_dims = 0;
        for d in a.into_iter() 
        {
            num_dims += d;
        }
        assert_eq!(num_dims, 128 + 3 + 256 + 256);
    }
}

