# Exceptions

## Basic exception handling

**Exceptions** are a way to keep Python errors under control, once a program is running. 
Exceptions errors arise when _correct_ Python programs (i.e., syntactically correct code) produce an error.

Python creates an exception object whenever such errors occur. When we write code that deals with the 
exception, our programs will continue running, even if an error is thrown. If we don't, then our programs will 
stop executing and show a trace-back, which is sometimes hard for a user to understand.

#### Example 1

Let's run a program that divides a number by zero. We know (or we should) that you cannot divide by zero,
but let’s see what Python does:
```python
print(6/0)
```
When we run the above code, Python gives the following traceback:

```
Traceback (most recent call last):
  File “C:\Users\ADMIN\PycharmProject\pythonProject\main.py”, line 1, in <module>
 print(6/0)
ZeroDivisionError: division by zero
```
Since Python cannot divide a number by zero, it reports an error in the trace-back as `ZeroDivisionError`, 
which is an exception object, and then the execution **stops**. 
This kind of object responds to a scenario where Python can't do what we asked it to.

!!!note
    There are many different types of exceptions in Python. You will probably find some of them soon,
    if you haven't yet. You can even define your own exceptions.

If you think an error might occur in your code, use the try-except block to control the exception 
that may be raised.

To handle the `ZeroDivisionError` exception, use a try-except block like this:

```python
try:
    print(6/0)
except ZeroDivisionError:
    print("You can’t divide by zero!") # You can’t divide by zero!
```
When you runt it, you will see the following output:
```python
You can’t divide by zero!
```

#### Example 2

Errors arise often when working with files that are missing. Python may fail to retrieve a file, if 
you have written the wrong spelling of the filename, or the file does not exist.

We handle this situation like before: by making use of the try-except block. For example, imagine the program 
below tries to read a file that doesn't exist on your computer:

```python
filename = 'some_nonexistent_file.txt'
with open(filename) as file:
    contents = file.read()
```

Since Python cannot read a file that does not exist, it raises an exception:

```shell
Traceback (most recent call last):
  File “C:\Users\ADMIN\PycharmProject\pythonProject\main.py”, line 2, in <module>
 with open(filename) as f_obj:
FileNotFoundError: [Errno 2] No such file or directory: ‘some_nonexistent_file.txt’
```

This is the `FileNotFoundError` exception. In this example, the `open()` function creates the error. 
To solve this error, use the try block just before the line, which involves the `open()` function:

```python
filename = 'some_nonexistent_file.txt'
try:
    with open(filename) as f_obj:
        contents = f_obj.read()
except FileNotFoundError:
    msg = "Sorry, the file "+ filename + "does not exist."
    print(msg) # Sorry, the file some_nonexistent_file.txt does not exist.
```

Now the code works correctly. This is known as _catching_ the exception.

#### Try-except structure

The full exception handling in Python has this structure:

```python
try:
   # Some Code.... 

except:
   # optional block
   # Handling of exception (if required)

else:
   # execute if no exception

finally:
  # Some code .....(always executed)
```
!!!note
    Note that in the try-except block above we didn't specify what exception we are 
    catching. It is not mandatory to do so, but it is very good practice to do it **always**.
    Otherwise, we could be having a different error in the code (that we have not foreseen)
    and we wouldn't notice!

However, in practice we often only use it like this:
```python
try:
    # some code
except SomeException:
    # what to do when the exception is raised
```

#### How to manually raise an exception in Python

How do we raise an exception in Python so that it can later be caught via an except block?
We should always use the most specific Exception constructor that semantically fits your issue.

Some common rules:

* Be specific in your message, e.g.:
    ```python
    raise ValueError('A very specific bad thing happened.')
    ```
* Don't raise generic exceptions: avoid raising a generic Exception. To catch it, you'll have to 
catch all other more specific exceptions that subclass it.