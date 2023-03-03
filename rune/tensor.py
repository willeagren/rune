"""
MIT License

Copyright (c) 2023 Wilhelm Ågren

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

File created: 2023-02-28
Last updated: 2023-03-04
"""

from __future__ import annotations

import logging
import numpy as np

from typing import (
    Dict,
    Union, 
    List,
    Tuple,
    Optional, 
    Sequence,
    NoReturn,
)
from rune.typing import ArrayD
from rune import Buffer 

logger = logging.getLogger(__name__)

__all__ = (
    'Tensor',
)

class Tensor(object):
    """
    """

    def __init__(self, data: Union[ArrayD, List, Sequence], *args: Tuple,
        requires_grad: Boolean = False, **kwargs: Dict) -> NoReturn:

        # TODO convert data to ndarray
        self.data = Buffer(data)
        self.shape = data.shape
        self.grad = None
        self._ctx = None
        self.requires_grad = requires_grad

    @property
    def shape(self) -> Tuple:
        return self.shape

    @property
    def requires_grad(self) -> Boolean:
        return self.requires_grad
    
