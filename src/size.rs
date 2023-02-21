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
// File created: 2023-02-21
// Last updated: 2023-02-21
//

#[derive(Debug, Clone)]
pub struct Size {
    s: Vec<usize>,
}

impl Size {
    pub fn new(s: &[usize]) -> Self {
        Size { s: s.to_vec() }
    }

    pub fn num_elements(&self) -> usize {
        self.s.iter().product()
    }

    pub fn to_vec(&self) -> &Vec<usize> {
        &self.s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construct() {
        let s = Size::new(&[12478, 8174, 1489, 2]);
        let c = Size::new(&[192, 4, 511, 3, 4, 5, 6]);

        assert_ne!(
            *s.to_vec(),
            *c.to_vec(),
        );
    }

    #[test]
    fn to_vec() {
        let s = Size::new(&[128, 64, 32, 32]);
        let v: &Vec<usize> = s.to_vec();

        assert_eq!(*v, vec![128, 64, 32, 32]);
    }

}

