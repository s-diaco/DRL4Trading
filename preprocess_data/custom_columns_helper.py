# The MIT License (MIT)

# Copyright 2021 Diaco Soltanpour

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""A collection of tools needed by costom columns"""

import numpy as np


def divide_array(col1, col2, col_out, zeros_or_ons):
    """
    Gets two arrays and divides using numpy.divide

    Parameters:
            col1 (array): Array to divide
            col2 (array): Array to divide to
            col_out (array): shape of the result
            zeros_or_ons (bool): If true, numpy.zeros_like and
                if false numpy.ones_like is used for 'Nan's

    Returns:
            pd.Series: The result of the division
    """
    if zeros_or_ons:
        divided_array = np.divide(
            col1,
            col2,
            out=np.zeros_like(col_out),
            where=col2 != 0)
    else:
        divided_array = np.divide(
            col1,
            col2,
            out=np.ones_like(col_out),
            where=col2 != 0)
    return divided_array
