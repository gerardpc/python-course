# IO, files and exceptions

## IO and files

### Introduction 

One of the most common tasks that you can do with Python is reading and writing files. 
Whether it’s writing to a simple text file, reading a complicated server log, or even analyzing raw 
byte data, all of these situations require reading or writing a file.

Before we can go into how to work with files in Python, it’s important to understand what exactly a 
file is and how modern operating systems handle some of their aspects.

At its core, a file is a contiguous set of bytes used to store data. This data is organized 
in a specific format and can be anything as simple as a text file or as complicated as a 
program executable. In the end, these byte files are then translated into binary 1 and 0 for 
easier processing by the computer.

Files on most modern file systems are composed of three main parts:

- **Header**: metadata about the contents of the file (file name, size, type, and so on)
- **Data**: contents of the file as written by the creator or editor
- **End of file (EOF)**: special character that indicates the end of the file

What this data represents depends on the format specification used, which is typically represented by an extension.

### File paths

When you access a file on an operating system, a file path is required. The file path is a string 
that represents the location of a file. It’s broken up into three major parts:

- **Folder Path**: the file folder location on the file system where subsequent folders are separated 
  by a forward slash / (Unix) or backslash \ (Windows)
- **File Name**: the actual name of the file
- **Extension**: the end of the file path pre-pended with a period (.) used to indicate the file type

Here’s a quick example. Let’s say you have a file located within a file structure like this:

    /
    │
    ├── path/
    |   │
    │   ├── to/
    │   │   └── cats.gif
    │   │
    │   └── dog_breeds.txt
    |
    └── animals.csv

Let’s say you wanted to access the cats.gif file, and your current location was in the same folder as path. 
In order to access the file, you need to go through the path folder and then the to folder, finally 
arriving at the `cats.gif` file. The Folder Path is `path/to/`. The File Name is cats. The File 
Extension is `.gif`. So the full path is `path/to/cats.gif`.

### Opening and Closing a File in Python

When you want to work with a file, the first thing to do is to open it. This is done by invoking 
the `open()` built-in function. `open()` has a single required argument that is the path to the file. 
`open()` has a single return, the file object. Once we are done with the file, we need to close it:

```python
file = open('dog_breeds.txt')
# do something with file
file.close()
```

It’s important to remember that it’s your responsibility to close the file! This is why it's always
recommended to use the `with` statement when dealing with files:

```python
with open('dog_breeds.txt') as reader:
    # Further file processing goes here
    ...
```

When the `with` statement is finished, everything goes back to normal (and we don't need to remember
to close anything). 

When opening a file, we are (directly or indirectly) using one of the different modes provided by Python.
The most commonly used modes are the following:

- `r`: Open for reading (default mode if nothing is specified)
- `w`: Open for writing, truncating (overwriting) the file first
- `a`: Open for writing, appending to the end of the file
- `rb` or `wb`: Open in binary mode (read/write using byte data)

Reading example:

```python
with open('dog_breeds.txt', 'r') as file:
    # Read & print the entire file
    print(file.read())

# Output
Pug
Jack Russell Terrier
English Springer Spaniel
German Shepherd
Staffordshire Bull Terrier
Cavalier King Charles Spaniel
Golden Retriever
West Highland White Terrier
Boxer
Border Terrier
```

Writing example:
```python
with open('dog_breeds_reversed.txt', 'w') as file:
    # Write the dog breeds to the file in reversed order
    for line in reversed(file):
        file.write(line)
```

Appending example:
```python
with open("test.txt", "a") as file:
    file.write("appended text")
```

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

```bash
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
FileNotFoundError: [Errno 2] No such file or directory: ‘john.txt’
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
    print(msg) # Sorry, the file John.txt does not exist.
```

Now the code works correctly. This is known as _catching_ the exception.

#### Try-except structure

The full exception handling in Pythong has this structure:

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