# Functions

## Functions

### Introduction

You may be familiar with the mathematical concept of a **function**. A function is a relationship or 
mapping between one or more inputs and a set of outputs. In mathematics, a function is typically represented like

\begin{align}
z = f(x,y)
\end{align}

Here, $f$ is a function that operates on the inputs $x$ and $y$. The output of the function is $z$. 
However, programming functions are much more generalized and versatile than this mathematical definition. 
In fact, appropriate function definition and use is so critical to proper software development that virtually 
all modern programming languages support both built-in and user-defined functions.

In programming, a function is a self-contained block of code that encapsulates a specific task or related group 
of tasks. We have already seen some of the built-in functions provided by Python: `print()`, `type()` or `len()`. 
For example, `len()` returns the length of the argument passed to it:

```python
>>> a = ['foo', 'bar', 'baz', 'qux']
>>> len(a)
4
```

These functions are part of the **Python Standard Library**, a collection of modules accessible to Python programs
that require no installation of external code. The Python standard library can be used to 
simplify the programming process, removing the need to reinvent the wheel and rewrite commonly used commands.

Most functions from the standard library can be used by calling `import package_name` at the beginning of a script,
where `package_name` is the name of the precise library that we want to use. We'll see a couple of important 
Python modules in the next sections.

!!!note
    Not all functions need to be imported to be used. `print()`, `type()` or `len()` are so general
    that we can use them without the need to import anything.

### The math library

For straightforward mathematical calculations in Python, you can use the built-in mathematical operators, 
such as addition (`+`), subtraction (`-`), division (`/`), and multiplication (`*`). But more advanced operations, 
such as exponential, logarithmic, trigonometric, or power functions, are not built in. Does that mean you need 
to implement all of these functions from scratch? 

Fortunately, no. Python provides a module specifically designed for higher-level mathematical operations: the math 
module. The math module comes packaged with the Python release, so you don’t have to install it separately. 
Using it is just a matter of importing the module:

```python
>>> import math
```
You can import the Python math module using the above command. After importing, you can use it straightaway.
For instance, imagine that we want to use the cosine function, $f(x) = \cos(x)$. Then we would do

```python
x = 3.14
y = math.cos(y)
```
and similarly for all other functions and parameters of the math module. 

!!!note
    If we don't want to write `math.XXX` in front of every import of the math module, we can also
    just import the specific parts of the module that we need, as in

    ```python
    from math import cos

    x = 3.14
    y = cos(y)
    ```

### Numbers, math functions

The math module provides many functions and important "named" numbers. This is a list of some of the
most important:

- `ceil(x)`: returns the smallest integer greater than or equal to x.
- `trunc(x)`: returns the truncated integer value of x.
- `factorial(x)`: returns the factorial of x
- `pow(x, y)`: returns x raised to the power y
- `cos(x)`, `sin(y)`, `tan(y)`: trigonometric functions
- `pi`: mathematical constant 3.1415...
- `e`: mathematical constant 2.7182...

### Random numbers

Python provides the `random` module to generate random numbers. This is also a built-in module that requires
no installation. `random` provides a number of useful tools for generating what we call pseudo-random data.

!!!note
    **Disclaimer**: most random data generated with Python is not fully random in the scientific sense of the word. 
    Rather, it is pseudorandom: generated with a pseudorandom number generator (PRNG), which is essentially any 
    algorithm for generating seemingly random but still reproducible data.

#### Random floats

The `random.random()` function returns a random float in the interval $[0.0, 1.0)$:

```python
>>> # Don't call `random.seed()` yet
>>> import random
>>> random.random()
0.35553263284394376
>>> random.random()
0.6101992345575074
```

If you run this code yourself, the numbers returned on your machine will be different. 
The default when you don’t seed the generator is to use your current system time or a “randomness source” from 
your OS if one is available.

With `random.seed()`, you can make results reproducible, and the chain of calls after `random.seed()` will 
produce the same trail of data:

```python
>>> random.seed(444)
>>> random.random()
0.3088946587429545
>>> random.random()
0.01323751590501987

>>> random.seed(444)  # Re-seed
>>> random.random()
0.3088946587429545
>>> random.random()
0.01323751590501987
```

#### Random integers

You can generate a random integer between two endpoints in Python with the `random.randint()` function. This spans 
the full $[x, y]$ interval and may include both endpoints:

```python
>>> random.randint(0, 10)
7
>>> random.randint(500, 50000)
18601
```

!!!note
    If we wanted to simulate a dice, we could run `random.randint(1, 6)`.

### Custom function definitions

If no function from an already existing package fits our needs, we can always define our own 
custom function.

When you define your own Python function, it works just the same as with built-in functions. 
From somewhere in your code, you’ll call your Python function and program execution will 
transfer to the body of code that makes up the function.

!!!note
    Functions are **really really important** in programming, since they allow code **reusability**.

When the function is finished, execution returns to the location where the function was called. 
Depending on how you designed the function’s interface, data may be passed in when the function is called, 
and return values may be passed back when it finishes.

The usual syntax for defining a Python function is as follows:

```python
def <function_name>([<arguments>]):
    """Docstring."""
    <statement(s)>
```

where the components are:

- `def`: the keyword that informs Python that a function is being defined
- `<function_name>`: A valid Python identifier that names the function
- `<arguments>`: An optional, comma-separated list of parameters that may be passed to the function
- `Docstring`: information on how the function works, what it does, its arguments and return types.
- `:`: Punctuation that denotes the end of the Python function header (the name and parameter list)
- `<statement(s)>`: A block of valid Python code that does something with the passed parameters


Here’s an example that defines and calls f():
```python
def f():    
    s = '-- Inside f()'    
    print(s)

print('Before calling f()')
f()
print('After calling f()')
```

Here’s how this code works:

- **Line 1** uses the def keyword to indicate that a function is being defined. Execution of the def 
  statement merely creates the definition of `f()`. All the following lines that are indented (lines 2 to 3) 
  become part of the body of `f()` and are stored as its definition, but they aren’t executed yet.

- **Line 4** is a bit of whitespace between the function definition and the first line of the main program. 
  While it isn’t syntactically necessary, it is nice to have. 

- **Line 5** is the first statement that isn’t indented because it isn’t a part of the definition of `f()`. 
  It’s the start of the main program. When the main program executes, this statement is executed first.

- **Line 6** is a call to `f()`. Note that empty parentheses are always required in both a function definition 
  and a function call, even when there are no parameters or arguments. Execution proceeds to `f()` and the statements 
  in the body of `f()` are executed.

- **Line 7** is the next line to execute once the body of `f()` has finished. Execution returns to this `print()` statement.

#### Return statement

To use a function, first you need to call it. As we have seen, a function call consists of the function's name 
followed by the function’s arguments in parentheses:

```python
function_name(arg1, arg2, ..., argN)
```

You’ll need to pass arguments to a function call only if the function requires them. The parentheses, on the other 
hand, are always required in a function call. If you forget them, then you won’t be calling the function but 
referencing it as a function object.

But, how do we make the function return a value? For that we need to use the Python `return` statement. 

```python
def mean(sample):
    return sum(sample) / len(sample)
```

we can also return more than one value if we put the results in a list, tuple or dictionary:

```python
def sum_and_diff(var1, var2):
    return var1 + var2, var1 - var2
```

#### Definition, arguments and type annotations

Consider the following function definition:

```python
def duplicate(msg):
    """Returns a string containing two copies of `msg`"""
    return msg + msg
```
The argument of the function is the parameter `msg`: this function is intended to duplicate the passed message.
For example, if called with the value `"Hello"`, it returns the value `"HelloHello"`. If called with other types of 
data, however, it will not work as expected. 

!!!note
    What will the function do if given an `int` or a `float` value?

Python allows you to indicate the intended type of the function parameters and the type of the function 
return value in a function definition using a special notation demonstrated in this example:

```python
def duplicate(msg: str) -> str:
    """Returns a string containing two copies of `msg`"""
    return msg + msg

result = duplicate('Hello')
print(result)
```
This definition of `duplicate` makes use of type annotations that indicate the function’s parameter type 
and return type (the return type is what comes after the `->`). A type annotation, sometimes called a type hint, 
is an optional notation that specifies the type of a parameter or function result. 
It tells the programmer using the function what kind of data to 
pass to the function, and what kind of data to expect when the function returns a value.

!!!note
    It’s important to understand that adding type annotations to a function definition does not cause the 
    Python interpreter to check that the values passed to a function are the expected types, or cause the 
    returned value to be converted to the expected type! This is only an indication for the programmer
    and, if you are using one, for the IDE.

For example, consider the following function:

```python
def add(x: int, y: int) -> int:
    """Returns the sum of `x` and `y`"""
    return x + y
```
If the function `add` in the example above is called like this:
```python
result = add('5', '15')
```
the function will receive two string values, concatenate them, and return the resulting string `"515"`. 
The `int` annotations are completely ignored by the Python interpreter. 

!!!note
    You should always try to use type annotations! Code looks much better with them, and it is easier
    to understand.

#### Mutability and arguments

In Python, arguments of functions can be of two types: immutable (`int`, `float`, `str`, `tuples`...) or mutable
(mostly `lst` and `dict`). 

If you pass a mutable object into a function, the function gets a reference to that same object: this means
that the function can **modify the value of the outer variable**. However, with immutable objects, the rest
of the script will remain unchanged. 

Example of a function **_that modifies_** a list:

```python
def try_to_change_list_contents(the_list):
    print('got', the_list)
    the_list.append('four')
    print('changed to', the_list)

outer_list = ['one', 'two', 'three']

print('before, outer_list =', outer_list)
try_to_change_list_contents(outer_list)
print('after, outer_list =', outer_list)

# Output:

before, outer_list = ['one', 'two', 'three']
got ['one', 'two', 'three']
changed to ['one', 'two', 'three', 'four']
after, outer_list = ['one', 'two', 'three', 'four']
```

Example of a function **_not_** **_modifying_** a string:

```python
def try_to_change_string_reference(the_string):
    print('got', the_string)
    the_string = 'In a kingdom by the sea'
    print('set to', the_string)

outer_string = 'It was many and many a year ago'

print('before, outer_string =', outer_string)
try_to_change_string_reference(outer_string)
print('after, outer_string =', outer_string)

# Output:

before, outer_string = It was many and many a year ago
got It was many and many a year ago
set to In a kingdom by the sea
after, outer_string = It was many and many a year ago
```

### Lambda functions

**Lambda functions** are small anonymous functions. They work as normal python functions, but are
defined within a single line and a little bit differently, as:

```python
lambda arguments : expression 
```

A lambda function can take any number of arguments, but can only have one expression. For instance,
a function that adds 10 to an argument a, and returns the result would be written as:

```python
x = lambda a : a + 10
print(x(5)) 
```