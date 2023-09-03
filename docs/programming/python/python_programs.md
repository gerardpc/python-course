# Python programs

## Python programs

If you quit from the Python interpreter and enter it again, the definitions you have made 
(functions and variables) are lost. Therefore, if you want to write a somewhat longer program, 
you are better off using a text editor to prepare the input for the interpreter and running it 
with that file as input instead. This is known as creating a **script**. As your program gets longer, 
you may want to split it into several files for easier maintenance. You may also want to use a 
handy function that you’ve written in several programs without copying its definition into each program.

What we want to make is a Python **module**: a file containing Python definitions and statements. 
The file name is the module name 
with the suffix `.py` appended. Within a module, the module’s name (as a string) is available as 
the value of the global variable `__name__`. 

### The import Statement

Python modules start by importing code from other modules, if necessary. 
The import statement takes many different forms, but the simplest form is the one already shown above:
```python
import some_module
```

We can then access the contents of the module (for example, an imaginary `some_function` from `some_module`)
like so:
```python
a = some_module.some_function()
```

An alternate form of the import statement allows individual objects from the module to be imported 
directly into the caller’s symbol table:
```python
from <module_name> import <name(s)>
```

Following execution of the above statement, <name(s)> can be referenced in the caller’s environment 
without the <module_name> prefix:
```python
>>> from mod import s
>>> s
'If Comrade Napoleon says it, it must be right.'

>>> from math import e
>>> e
2.718281828459045
```
Because this form of import places the object names directly into the caller’s symbol table, 
any objects that already exist with the same name will be overwritten:
```python
>>> a = ['foo', 'bar', 'baz']
>>> a
['foo', 'bar', 'baz']

>>> from mod import a
>>> a
[100, 200, 300]
```
It is also possible to import individual objects but enter them into the local symbol 
table with alternate names:
```python
from <module_name> import <name> as <alt_name>[, <name> as <alt_name> …]
```
This makes it possible to place names directly into the local symbol table but avoid conflicts with previously existing names:
```python
>>> s = 'foo'
>>> a = ['foo', 'bar', 'baz']

>>> from mod import s as string, a as alist
>>> s
'foo'
>>> string
'If Comrade Napoleon says it, it must be right.'
>>> a
['foo', 'bar', 'baz']
>>> alist
[100, 200, 300]
```

Finally, you can also import an entire module under an alternate name:

```python
import <module_name> as <alt_name>

>>> import mod as my_module
>>> my_module.a
[100, 200, 300]
>>> my_module.foo('qux')
arg = qux
```

### Definitions

The central part of the Python module are function and class definitions. Hence, so far a module could 
look like this

```python
import math as m

def silly_function(a: int) -> int:
    """Some description."""
    b = m.cos(a)**2
    return b
```

### Script

Our module, so far, only has imports and definitions. This means that it does not do anything. We can
add some function calls (anything that is not inside a definition of a function or a class gets *actually*
executed) in the last section of the file: 

```python
import math as m

def silly_function(a: int) -> int:
    """Some description."""
    b = m.cos(a)**2
    return b

print("This will be executed.")
var = silly_function(2)
print(f"The result is {var}.")
```

!!!note
    Even though the structure above is *the standard*, in principle you can order your code however you want.
    Just bear in mind that code starts executing from the **beginning of the file** and then goes down line
    by line. So if there is a script section in the middle of the file that makes use of something that hasn't 
    been defined yet, it will not work.


### Running scripts from the terminal

Any `.py` file that contains a module is essentially a Python script. Therefore, let's save our newly
created module as `test.py` inside a folder. To run the script, we need to open a terminal, go to the 
folder where the file is located and then run

```bash
/home/gerard/documents> python3 test.py
```
or, in Windows,
```dos
C:\Users\gerard\Documents> python test.py
```