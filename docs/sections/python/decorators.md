# Decorators

## Introduction

Decorators are a way to modify or extend the behavior of functions or methods, without modifying the 
original function or method. This sounds confusing, but it’ll make more sense after you’ve seen a few examples 
of how decorators work. 

## Decorator syntax

In Python, decorators are implemented using the `@` symbol, followed by the name of the decorator function. 
They always come before the definition of a function, and are applied to the function below. The syntax looks 
like this:

```python
@some_decorator
def some_function():
    print("This is the inside of some_function.")
```

but even though the syntax is a bit strange, it is just a shorthand (i.e., "syntactic sugar") for the following:

```python
def some_function():
    print("This is the inside of some_function.")

some_function = some_decorator(some_function)
```

In other words, the `@some_decorator` syntax is just a function that takes another function as input, and returns
a new function, modified, version of the input function.

## Decorator example

Python libraries are often full of decorators, and you’ve probably used them without even realizing it.
Let's make a simple example of a decorator that prints the time it takes to run a function. We can use the
`time` module to get the current time, and then subtract the start time from the end time to get the time it
took to run the function. 

```python
import time

# The timer function (which we will use as a decorator) takes a function as input, 
# and returns a new function (the "wrapper" function) that does what the previous function did,
# plus a little extra (in this case, it prints the time it took to run the function).
def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Elapsed time: {end - start} seconds")
        return result
    return wrapper

# Now we can use the `@timer` decorator to print the time it takes to run a function.
@timer
def some_function():
    print("This is the inside of some_function.")
    time.sleep(1)

some_function()

# Output:
This is the inside of some_function.
Elapsed time: 1.000000238418579 seconds
```