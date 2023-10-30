# Numpy

## Introduction

Numpy is a library for scientific computing in Python. It provides a multidimensional array object, the `ndarray`
(often known as just an array) that can be thought of as a generalization of a matrix. It also provides a 
large collection of high-level and very efficient mathematical functions to operate on these arrays, including:

* Generic mathematical and logical operations
* Array shape manipulation
* Sorting
* Linear algebra
* Fourier transforms
* Basic statistical operations (including random numbers)

and much more.

## Installation

Numpy is installed by default in Anaconda. If you are using a different Python distribution, you can 
install it using pip:

```bash
pip install numpy
```

## Importing

To import numpy, use the following command:

```python
import numpy as np
```

!!!note
    It is a convention to import numpy as `np`. You can, of course, use any other name.

## Arrays

The ndarray is a multidimensional array of elements of the same type, usually of numbers either integers or floats.
The number of dimensions and items in an array is defined by its shape, which is a tuple of N non-negative integers.

### Creating Arrays

There are several ways to create numpy arrays. The most common way is to create an array from a Python list,
using the `array` function:

```python
a = np.array([1, 2, 3])
print(a)

# Output:
[1 2 3]
```

You can also create a multidimensional array by passing a list of lists:

```python
a = np.array([[1, 2, 3], [4, 5, 6]])
```

The array type can be explicitly specified at creation time:

```python
a = np.array([1, 2, 3], dtype=float)
```

You can also create arrays filled with zeros or ones:

```python
a = np.zeros((3, 3))
b = np.ones((3, 3))
```

To create sequences of numbers, NumPy provides the `arange` function which is analogous to the Python built-in 
range, but returns an array.
    
```python
a = np.arange(10) # 0 .. n-1  (!)
b = np.arange(1, 9, 2) # start, end (exclusive), step
```

For creating arrays with evenly spaced numbers, there is also the `linspace` function:

```python
a = np.linspace(0, 1, 6)   # start, end, num-points
print(a)
# Output:
[0. 0.2 0.4 0.6 0.8 1.]
```

### Array attributes

The most important attributes of an ndarray object are:

* `ndarray.ndim`: the number of axes (dimensions) of the array.
* `ndarray.shape`: the dimensions of the array. This is a tuple of integers indicating the size of the array 
in each dimension. For a matrix with `n` rows and `m` columns, `shape` will be `(n,m)`. The length of 
the `shape` tuple is therefore the number of axes, `ndim`.
* `ndarray.size`: the total number of elements of the array. This is equal to the product of the elements of `shape`.
* `ndarray.dtype`: an object describing the type of the elements in the array. One can create or specify
dtype’s using standard Python types. Additionally NumPy provides types of its own (e.g. `numpy.int32` or `numpy.int16`).
* `ndarray.itemsize`: the size in bytes of each element of the array. For example, an array of elements of type
`float64` has `itemsize` 8 (=64/8), while one of type `complex32` has `itemsize` 4 (=32/8).

### Basic operations

Python arithmetic operators on arrays apply elementwise. A new array is created and filled with the result.

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

c = a + b
print(c)
# Output:
[5 7 9]
```
Unlike in many matrix languages, the product operator `*` operates elementwise in NumPy arrays. 
The matrix product can be performed using the `@` operator (in python >=3.5):

```python
a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([[1, 2], [3, 4], [5, 6]])

c = a @ b
print(c)
# Output:
[[22 28]
 [49 64]]
```

Some operations, such as `+=` and `*=`, act in place to modify an existing array rather than create a new one.
When operating with arrays of different types, the type of the resulting array corresponds to the more 
general or precise one (a behavior known as **upcasting**).

### Methods

Many unary operations, such as computing the sum of all the elements in the array, are implemented as 
methods of the ndarray class
    
```python
a = np.array([[1, 2, 3], [4, 5, 6]])
print(a.sum())
# Output:
21
```

By default, these operations apply to the array as though it were a list of numbers, regardless of its shape. 
However, by specifying the axis parameter you can apply an operation along the specified axis of an array:
```python
b = np.arange(12).reshape(3, 4)

print(b)
# Output:
[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]]

print(b.sum(axis=0))     # sum of each column
# Output:
[12 15 18 21]

print(b.min(axis=1))     # min of each row
# Output:
[0 4 8]

print(b.cumsum(axis=1))  # cumulative sum along each row
# Output:
[[ 0  1  3  6]
 [ 4  9 15 22]
 [ 8 17 27 38]]
```

### Universal functions

NumPy provides familiar mathematical functions such as sin, cos, and exp. In NumPy, these are called
**universal functions**. Within NumPy, these functions operate elementwise on an array, producing an array
as output.

```python
a = np.arange(3)
print(np.exp(a))
# Output:
[1.         2.71828183 7.3890561 ]
```

### Indexing, Slicing and Iterating

One-dimensional arrays can be indexed, sliced and iterated over, much like lists and other Python sequences.

```python
a = np.arange(10)**3
print(a)
# Output:
[  0   1   8  27  64 125 216 343 512 729]

print(a[2:5])
# Output:
[ 8 27 64]
```

Multidimensional arrays can have one index per axis. These indices are given in a tuple separated by commas:

```python
def f(x, y):
    return 10*x+y

b = np.fromfunction(f, (5, 4), dtype=int)
print(b)
# Output:
[[ 0  1  2  3]
 [10 11 12 13]
 [20 21 22 23]
 [30 31 32 33]
 [40 41 42 43]]

print(b[2, 3])
# Output:
23

print(b[0:5, 1])  # each row in the second column of b
# Output:
[ 1 11 21 31 41]
```

### Stacking together different arrays

Several arrays can be stacked together along different axes:

```python
a = np.floor(10*np.random.random((2,2)))
b = np.floor(10*np.random.random((2,2)))

print(np.vstack((a,b)))
# Output:
[[5. 9.]
 [0. 0.]
 [1. 7.]
 [8. 2.]]

print(np.hstack((a,b)))
# Output:
[[5. 9. 1. 7.]
 [0. 0. 8. 2.]]
```

### Linear algebra

NumPy provides the `linalg` package to perform linear algebra operations. To compute the inverse of a matrix:

```python
a = np.array([[1., 2.], [3., 4.]])
print(np.linalg.inv(a))
# Output:
[[-2.   1. ]
 [ 1.5 -0.5]]
```

To compute the eigenvalues and eigenvectors of a square matrix:

```python
a = np.array([[1., 2.], [3., 4.]])
w, v = np.linalg.eig(a)

print(w)
# Output:
[-0.37228132  5.37228132]

print(v)
# Output:
[[-0.82456484 -0.41597356]
 [ 0.56576746 -0.90937671]]
```

### Random numbers

NumPy has powerful random number generating capabilities. It uses a particular algorithm called the Mersenne Twister
to generate pseudorandom numbers. The `random` module provides tools for making random selections. For example,
to pick a random number from a uniform distribution:

```python
print(np.random.rand())
# Output:
0.47108547995356098
```

To pick a random number from a normal distribution:

```python
print(np.random.randn())
# Output:
-0.72487283708301885
```
