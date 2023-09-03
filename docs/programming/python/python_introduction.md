# Introduction to Python

## Language Introduction

Python is an interpreted language. It uses **variables** to store information: 
whether information is a number, text or a list of names, it will always be saved in a variable. 
Information in variables can later be processed, or kept as is. 

When we declare variables in Python, we are not forced to tell the interpreter of what sort they 
will be; Python will infer it at runtime. This gives the programmer some flexibility, but also 
the possibility of making mistakes.

An excellent way to see how Python code works is to run the Python interpreter and type code 
right into it. If you ever have a question like, "What happens if I do this or that?" just 
typing it into the Python interpreter is a fast and likely the best way to see what happens. 

```python
$ python3        ## Run the Python interpreter
Python 3.X.X (XXX, XXX XX XXXX, XX:XX:XX) [XXX] on XXX
Type "help", "copyright", "credits" or "license" for more information.
>>> a = 6       ## set a variable in this interpreter session
>>> a           ## entering an expression prints its value
6
>>> a + 2
8
>>> a = 'hi'    ## 'a' can hold a string just as well
>>> a
'hi'
>>> len(a)      ## call the len() function on a string
2
>>> a + len(a)  ## try something that doesn't work
Traceback (most recent call last):
  File "", line 1, in 
TypeError: can only concatenate str (not "int") to str
>>> a + str(len(a))  ## probably what you really wanted
'hi2'
>>> foo         ## try something else that doesn't work
Traceback (most recent call last):
  File "", line 1, in 
NameError: name 'foo' is not defined
>>> ^D          ## type CTRL-d to exit (CTRL-z in Windows/DOS terminal)
```

!!!note
    If you don't understand everything that is happening in the previous code snippet, 
    don't worry! We'll see it on the next sessions.

As you can see above, it's easy to experiment with variables and operators. 
Also, the interpreter throws, or "raises" in Python parlance, a runtime error if the 
code tries to read a variable that has not been assigned a value. Like other programming languages, 
Python is case sensitive so "`a`" and "`A`" are different variables. The end of a line marks the end 
of a statement, so Python does not require a semicolon at the end of each statement. 
Comments begin with a `#` and extend to the end of the line.

## Basic variables and types

### Numbers

The interpreter acts as a simple calculator: you can type an expression at it and it will write the value. 
Expression syntax is straightforward: the operators +, -, * and / can be used to perform arithmetic; 
parentheses (()) can be used for grouping. For example:

```python
>>>

2 + 2
4

50 - 5*6
20

(50 - 5*6) / 4
5.0

8 / 5  # division always returns a floating point number
1.6
```

The **integer** numbers (e.g. 2, 4, 20) have type `int`, the ones with a **fractional part** 
(e.g. 5.0, 1.6) have type `float`. Finally, **boolean** numbers have type `bool` and represent
logical values of `True` or `False`.

!!!note
    Booleans can't be operated as normal numbers, with additions or powers. Rather, they should
    be used in logical expressions such as "do this if this and that, or do that if this and that".
    We'll see more about it in another section.

Division (/) always returns a float. To do floor division and get an integer result you can 
use the // operator; to calculate the remainder you can use %:
```python
>>>

17 / 3  # classic division returns a float
5.666666666666667
>>>

17 // 3  # floor division discards the fractional part
5

17 % 3  # the % operator returns the remainder of the division
2

5 * 3 + 2  # floored quotient * divisor + remainder
17

```

With Python, it is possible to use the ** operator to calculate powers:

```python
>>>

5 ** 2  # 5 squared
25

2 ** 7  # 2 to the power of 7
128
```

The equal sign `=` is used to assign a value to a variable. Afterwards, no result is displayed 
before the next interactive prompt:
```python
>>>

width = 20

height = 5 * 9

width * height
900
```

If a variable is not “defined” (assigned a value), trying to use it will give you an error:

```python
>>>

n  # try to access an undefined variable
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'n' is not defined
```

There is full support for floating point; operators with mixed type operands convert the integer operand 
to floating point:
```python
>>>

4 * 3.75 - 1
14.0
```
In interactive mode, the last printed expression is assigned to the variable _. This means that when you are using 
Python as a desk calculator, it is somewhat easier to continue calculations, for example:
```python
>>>

tax = 12.5 / 100

price = 100.50

price * tax
12.5625

price + _
113.0625

round(_, 2)
113.06

```

This variable should be treated as read-only by the user. Don’t explicitly assign a value to it — you would 
create an independent local variable with the same name masking the built-in variable with its magic behavior.

### Text

Python can manipulate text (represented by type `str`, so-called “strings”) as well as numbers. 
This includes characters “!”, words “rabbit”, names “Paris”, sentences “Got your back.”, etc. “Yay! :)”. 
They can be enclosed in single quotes ('...') or double quotes ("...") with the same result.
```python
>>>

'spam eggs'  # single quotes
'spam eggs'

"Paris rabbit got your back :)! Yay!"  # double quotes
'Paris rabbit got your back :)! Yay!'

'1975'  # digits and numerals enclosed in quotes are also strings
'1975'
```

We should always use double quotes `"..."` as quotation marks (since they allo the use of `'` inside):
```python
>>>
"doesn't need to"  # ...use double quotes instead
"doesn't need to"
```

In the Python shell, the string definition and output string can look different. The print() 
function produces a more readable output, by omitting the enclosing quotes and by 
printing escaped and special characters:
```python
>>>

s = 'First line.\nSecond line.'  # \n means newline

s  # without print(), special characters are included in the string
'First line.\nSecond line.'

print(s)  # with print(), special characters are interpreted, so \n produces new line
First line.
Second line.
```

If you don’t want characters prefaced by \ to be interpreted as special characters, you can use raw 
strings by adding an r before the first quote:

```python
>>>

print('C:\some\name')  # here \n means newline!
C:\some
ame

print(r'C:\some\name')  # note the r before the quote
C:\some\name
```

**String literals** can span multiple lines. One way is using triple-quotes: `"""..."""`. 
End of lines are automatically included in the string:

```python
print("""\
Usage: thingy [OPTIONS]
     -h                        Display this usage message
     -H hostname               Hostname to connect to
""")
```

produces the following output (note that the initial newline is not included):

```python
Usage: thingy [OPTIONS]
     -h                        Display this usage message
     -H hostname               Hostname to connect to
```

Strings can be concatenated (glued together) with the + operator, and repeated with *:

```python
>>>

# 3 times 'un', followed by 'ium'

3 * 'un' + 'ium'
'unununium'
```

## Basic Python functions

We can find the complete set of Python built-in functions [here](https://docs.python.org/3/library/functions.html).
In this section we will only describe the most common and basic:

### Type

The `type()` function is mostly used for debugging purposes. If a single argument `type(obj)` is passed, 
it returns the type of the given object. 

```python
x = 10
print(type(x))
# Output: <class 'int'>
```

### Length

The `len()` function returns the length of a data structure passed to it. We'll see more about it in coming
sessions:

```python
>>> a = ['foo', 'bar', 'baz', 'qux']
>>> len(a)
4
```

### Python Output

In Python, we can simply use the `print()` function to print output on the "standard output". For example,
```python
print('Python is powerful')

# Output: Python is powerful
```
Here, the `print()` function displays the string enclosed inside the single quotation.
In the above code, the `print()` function is taking a single parameter (the actual syntax of 
the print function accepts as much as 5 parameters, but if we really need them we can google it).

We can also use the `print()` function to display the values of Python variables. We can do it
in several ways, but the most convenient is the use of the `f-string`:
1. Prefix the string that we want to print with the letter `f`, as in `print(f"some text here")`. 
2. Now, imagine that we have defined a variable before, as in
    ```python
    dog_name = "Ruff"
    ```
3. To use this variable with `print()` in an f-string we just need to call it like
    ```python
    print(f"My dog name is {dog_name}")
    # Output: My dog name is Ruff
    ```
4. We can insert as many variables as we like in an f-string:
    ```python
    age = 15
    hair_color = purple
    name = "Amy"
    print(f"My name is {name}, I'm {age} and have {hair_color} hair.")
    # Output: My name is Amy, I'm 15 and have purple hair.
    ```

We can also join two strings together inside the `print()` statement. For example,
```python
print('This class is ' + 'awesome.')
# Output: This class is awesome.
```

The `print()` function normally prints out one or more python items followed by a newline
(but the ending character, which by default is `\n`, can be changed.)

### Python input

While programming, we might want to take the input from the user. In Python, we can use the `input()` function:
```python
input(prompt)
```
Here, `prompt` is the string we wish to display on the screen and is optional.

An example:
```python
# using input() to take user input
num = input('Enter a number: ')

print('You Entered:', num)

print('Data type of num:', type(num))
```
**Output**:
```python
Output

Enter a number: 10
You Entered: 10
Data type of num: <class 'str'>
```
In the above example, we have used the `input()` function to take input from the user 
and stored the user input in the num variable.

It is important to note that the entered value 10 is a `string`, not a `number`. So, type(num) 
returns <class 'str'>. If we want to convert the string to a number, we should do it explicitly,
like:

```python
age = input("Enter your age: ")
age = int(age)
```

## Basic Python operators

### Python Arithmetic Operators

Arithmetic operators are used to perform mathematical operations like addition, subtraction, 
multiplication, etc. For example,

```python
a = 7
b = 2

# addition
print ('Sum: ', a + b)  

# subtraction
print ('Subtraction: ', a - b)   

# multiplication
print ('Multiplication: ', a * b)  

# division
print ('Division: ', a / b) 

# floor division
print ('Floor Division: ', a // b)

# modulo
print ('Modulo: ', a % b)  

# a to the power b
print ('Power: ', a ** b)   
```

**Output**:
```python
Sum: 9
Subtraction: 5
Multiplication: 14
Division: 3.5
Floor Division: 3
Modulo: 1
Power: 49
```

### Assignment operators

Assignment operators are used to assign values to variables. For example,

```python
# assign 5 to x 
var x = 5
```
Here, `=` is an assignment operator that assigns 5 to x. Some extra examples:
```python
# assign 10 to a
a = 10

# assign 5 to b
b = 5 

# assign the sum of a and b to a
a += b      # a = a + b

print(a)

# Output: 15
```

### Python Comparison Operators

Comparison operators compare two values/variables and return a boolean result: `True` or `False`. 
For example,

```python
a = 5

b = 2

# equal to operator
print('a == b =', a == b)

# not equal to operator
print('a != b =', a != b)

# greater than operator
print('a > b =', a > b)

# less than operator
print('a < b =', a < b)

# greater than or equal to operator
print('a >= b =', a >= b)

# less than or equal to operator
print('a <= b =', a <= b)
```

### Python Logical Operators

Logical operators are used to check whether an expression is `True` or `False`. 
They are used in decision-making. For example,
```python
# logical AND
print(True and True)     # True
print(True and False)    # False

# logical OR
print(True or False)     # True

# logical NOT
print(not True)          # False
```
Python also offers some special types of operators, like the identity operator and the membership operator. 
In Python, `is` and `is not` are used to check if two values are located on the same part of the memory:
```python
x1 = 5
y1 = 5
x2 = 'Hello'
y2 = 'Hello'
x3 = [1,2,3]
y3 = [1,2,3]

print(x1 is not y1)  # prints False

print(x2 is y2)  # prints True

print(x3 is y3)  # prints False
```
`in` and `not in` are the membership operators. They are used to test whether a value or 
variable is found in a sequence (string, list, tuple, set and dictionary):

```python
x = 'Hello world'
y = {1:'a', 2:'b'}

# check if 'H' is present in x string
print('H' in x)  # prints True

# check if 'hello' is present in x string
print('hello' not in x)  # prints True

# check if '1' key is present in y
print(1 in y)  # prints True

# check if 'a' key is present in y
print('a' in y)  # prints False
```

