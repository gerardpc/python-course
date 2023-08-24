# Python

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

## Conditional statements

Python uses boolean logic to evaluate conditions. For example:
```python
x = 2
print(x == 2) # prints out True
print(x == 3) # prints out False
print(x < 3) # prints out True
```

### if/else statements

Conditional statements are used to control the flow of the program with `if`/`else` statements. 
We use the `if` statement to run a block code only when a certain condition is met.

For example, assigning grades (A, B, C) based on marks obtained by a student.

    if the percentage is above 90, assign grade A
    if the percentage is above 75, assign grade B
    if the percentage is above 65, assign grade C

In Python, there are three forms of the if...else statement.

    if statement
    if...else statement
    if...elif...else statement

The syntax of if statement in Python is:

```python
if condition:
    # body of if statement
```
The `if` statement evaluates `condition`.

    If condition is evaluated to True, the code inside the body of if is executed.
    If condition is evaluated to False, the code inside the body of if is skipped.

```python
number = 10

# check if number is greater than 0
if number > 0:
    print('Number is positive.')

print('The if statement is easy')
```
**Output**
```python
Number is positive.
The if statement is easy
```
An `if` statement can have an optional `else` clause.

The syntax of` if...else` statement is:
```python
if condition:
    # block of code if condition is True
else:
    # block of code if condition is False
```
The `if...else` statement is used to execute a block of code among two alternatives.

However, if we need to make a choice between more than two alternatives, we can use
the `if...elif...else` statement.

The syntax of the if...elif...else statement is:
```python
if condition1:
    # code block 1
elif condition2:
    # code block 2
else: 
    # code block 3
```

We can also use an `if` statement inside of an `if` statement. This is known as a nested if statement:
```python
number = 5

# outer if statement
if number > 0:
    # inner if statement
    if number > 100000:
        print('Number is very big!')    
    # inner else statement
    else:
        print('Number is positive')
elif number == 0:
    print("Number is zero.")
# outer else statement
else:
    print('Number is negative')
# Output: Number is positive
```

Any value can be used as an "if-test". The "zero" values all count as false: `None`, `0`, empty string, 
empty list, empty dictionary. Each block of if/else statements starts with a `:` and the statements are 
grouped by their indentation:

```python
if time_hour >= 0 and time_hour <= 24:
    print('Suggesting a drink option...')
if mood == 'sleepy' and time_hour < 10:
    print('coffee')
elif mood == 'thirsty' or time_hour < 2:
    print('lemonade')
else:
    print('water')
```


### Match case

Python 3.10 offers a simple and effective way to test multiple values and perform conditional 
actions: the match-case statement. In case you’re familiar with C++, it works similarly to the switch case.

For our example, let’s say you’re building a program to check a computer’s processor. 
Based on the result, the program will let the gamer know if their processor is compatible 
with a certain video game. Here’s how our program would look:

```python
# First, ask the player about their CPU
cpu_model = input("Please enter your CPU model: ")
 
# The match statement evaluates the variable's value
match cpu_model:
    case "celeron": # We test for different values and print different messages
            print ("Forget about it and play Minesweeper instead...")
    case "core i3":
            print ("Good luck with that ;)")
    case "core i5":
            print ("Yeah, you should be fine.")
    case "core i7":
            print ("Have fun!")
    case "core i9":
            print ("Our team designed nice loading screens… Too bad you won't see them...")
    case _: # the underscore character is used as a catch-all.
            print ("Is that even a thing?")
```

The above code checks a few possible values for the cpuModel variable. 
If the CPU model the user enters doesn’t match our expected options, the final 
case statement prints an error message. Here’s a possible output:
```python
Please enter your CPU model: core i9
Our teams designed nice loading screens... Too bad you won't see them...
```

- Types of programs, stages in programming and expressions.
- Using conditional statements to satisfy program specifications

## Strings

Python has a built-in string class named `str` with many handy features. String literals can be 
enclosed by either double or single quotes. Backslash escapes work the usual way within both 
single and double quoted literals -- e.g. `\n` `\'` `\"`. A double quoted string literal can 
contain single quotes without any fuss (e.g. `"I didn't do it"`) and likewise single quoted string 
can contain double quotes. 

String literals inside triple quotes, """ or ''', can span multiple lines of text.
Python strings are "immutable", which means they cannot be changed after they are created.
Since strings can't be changed, we construct *new* strings as we go to represent computed values. 
So for example the expression (`'hello'` + `'there'`) takes in the 2 strings `'hello'` and `'there'` and 
builds a new string `'hellothere'`.

Characters in a string can be accessed using the bracket `[ ]` syntax. 

!!!note
    Like other languages, Python uses **zero-based indexing**, so if `s` is `'hello'` `s[1]` is `'e'`. 
    If the index is out of bounds for the string, Python raises an error. 

The handy "slice" syntax (below) also works to extract any substring from a string. 
The `len(string)` function returns the length of a string. The `[ ]` syntax and the `len()` 
function actually work on any sequence type -- strings, lists, etc. 
Python tries to make its operations work consistently across different types. 

!!!note
    Python newbie gotcha: don't use "`len`" as a variable name to avoid blocking out the `len()` function. 

The `+` operator can concatenate two strings. Notice in the code below that variables are not pre-declared -- 
just assign to them and go.

```python
s = 'hi'
print(s[1])          ## i
print(len(s))        ## 2
print(s + ' there')  ## hi there
```

The '`+`' symbol does not automatically convert numbers or other types to string form. The `str()` function 
converts values to a string form so they can be combined with other strings.

```python
pi = 3.14
##text = 'The value of pi is ' + pi      ## NO, does not work
text = 'The value of pi is '  + str(pi)  ## yes
```

A "raw" string literal is prefixed by an '`r`' and passes all the chars through without special treatment of 
backslashes, so `r'x\nx'` evaluates to the length-4 string `'x\nx'`.

```python
raw = r'this\t\n and that'

# this\t\n and that
print(raw)

multi = """It was the best of times.
It was the worst of times."""

# It was the best of times.
#   It was the worst of times.
print(multi)
```


### String Methods

Here are some of the most common string methods. A method is like a function, but it runs 
"on" an object. If the variable `s` is a string, then the code `s.lower()` runs the `lower()` method on 
that string object and returns the result (this idea is one of the basic ideas that make up 
**Object Oriented Programming**, OOP). Here are some of the most common string methods:

```python
s.lower(), s.upper() -- returns the lowercase or uppercase version of the string
s.strip() -- returns a string with whitespace removed from the start and end
s.isalpha()/s.isdigit()/s.isspace()... -- tests if all the string chars are in the various character classes
s.startswith('other'), s.endswith('other') -- tests if the string starts or ends with the given other string
s.find('other') -- searches for the given other string (not a regular expression) within s, and returns the first index where it begins or -1 if not found
s.replace('old', 'new') -- returns a string where all occurrences of 'old' have been replaced by 'new'
s.split('delim') -- returns a list of substrings separated by the given delimiter. The delimiter is not a regular expression, it's just text. 'aaa,bbb,ccc'.split(',') -> ['aaa', 'bbb', 'ccc']. As a convenient special case s.split() (with no arguments) splits on all whitespace chars.
s.join(list) -- opposite of split(), joins the elements in the given list together using the string as the delimiter. e.g. '---'.join(['aaa', 'bbb', 'ccc']) -> aaa---bbb---ccc
```

A google search for "python str" should lead you to the official python.org string methods which lists all the `str` methods.

The "slice" syntax is a handy way to refer to sub-parts of sequences -- typically strings and lists. 
The slice `s[start:end]` is the elements beginning at start and extending up to but not including end. 
Suppose we have `s = "Hello"`:

```python
s[1:4] is 'ell' -- chars starting at index 1 and extending up to but not including index 4
s[1:] is 'ello' -- omitting either index defaults to the start or end of the string
s[:] is 'Hello' -- omitting both always gives us a copy of the whole thing (this is the pythonic way to copy a sequence like a string or list)
s[1:100] is 'ello' -- an index that is too big is truncated down to the string length
```
The standard zero-based index numbers give easy access to chars near the start of the string. 
As an alternative, Python uses negative numbers to give easy access to the chars at the end of 
the string: `s[-1]` is the last char `'o'`, `s[-2]` is `'l'` the next-to-last char, and so on. 
Negative index numbers count back from the end of the string:

```python
s[-1] is 'o' -- last char (1st from the end)
s[-4] is 'e' -- 4th from the end
s[:-3] is 'He' -- going up to but not including the last 3 chars.
s[-3:] is 'llo' -- starting with the 3rd char from the end and extending to the end of the string. 
```

It is a neat thing of slices that for any index n, `s[:n] + s[n:] == s`. This works 
even for n negative or out of bounds. 

One neat thing python can do is automatically convert objects into a string suitable for printing. 
Two built-in ways to do this are formatted string literals, also called "`f-strings`", and invoking `str.format()`.

You'll often see formatted string literals used in situations like:

```python
value = 2.791514
print(f'approximate value = {value:.2f}')  # approximate value = 2.79

car = {'tires':4, 'doors':2}
print(f'car = {car}') # car = {'tires': 4, 'doors': 2}
```

A formatted literal string is prefixed with `'f'` (like the `'r'` prefix used for raw strings). Any 
text outside of curly braces `'{}'` is printed out directly. Expressions contained in `'{}'` are are 
printed out using the format specification described in the format spec. There are lots of neat things 
you can do with the formatting including truncation and conversion to scientific notation and left/right/center alignment.

`f-strings` are very useful when you'd like to print out a table of objects and would like the columns 
representing different object attributes to be aligned like

```python
address_book = [{'name':'N.X.', 'addr':'15 Jones St', 'bonus': 70},
  {'name':'J.P.', 'addr':'1005 5th St', 'bonus': 400},
  {'name':'A.A.', 'addr':'200001 Bdwy', 'bonus': 5},]

for person in address_book:
    print(f'{person["name"]:8} || {person["addr"]:20} || {person["bonus"]:>5}')

# N.X.     || 15 Jones St          ||    70
# J.P.     || 1005 5th St          ||   400
# A.A.     || 200001 Bdwy          ||     5
```

!!!note
    We'll see about the `for` loop in the next section!

## Loops

### Python lists

Python has a built-in ordered list type named "`list`". Lists are written within square brackets `[ ]`. 
Lists work similarly to strings -- use the `len()` function and square brackets `[ ]` to access data, with 
the first element at index 0. (See the official python.org list docs.)

```python
colors = ['red', 'blue', 'green']
print(colors[0])    # red
print(colors[2])    # green
print(len(colors))  # 3
```

!!!note
    You can put any sort of variable you want inside a list! Numbers, strings or other (more exotic)
    objects will work.

Assignment with an `=` on lists **does not make a copy**. Instead, assignment makes the two variables 
point to the same one list in memory.

```python
b = colors   # Does not copy the list, just reuses it!
```

This means that if we now change `b`, we will also be changing the contents of the list `colors`.
To make a _different_ (bud identical) copy of a list, that can be modified without affecting the 
original list, wee need to call the `copy()` method

```python
b = colors.copy()   # b is now a proper copy of colors
```

The "empty list" is just an empty pair of brackets `[ ]`. The '`+`' works to append two lists, 
so `[1, 2] + [3, 4]` yields `[1, 2, 3, 4]` (this is just like `+` with strings).

### For loop

Python's *for* and *in* constructs are extremely useful, and the first use of them we'll see is 
with lists. The `for` construct -- `for var in list` -- is an easy way to look at each element 
in a list (or other collection). 

```python
squares = [1, 4, 9, 16]
sum = 0
for num in squares:
    sum += num
print(sum)  # 30
```

!!!note
    **Do not** add or remove items from the list during iterations! It will give you plenty of headaches.

If you know what sort of thing is in the list, use a variable name in the loop that captures that 
information (such as "num", or "name", or "url") to improve readability. 

The *in* construct on its own is an easy way to test if an element appears in a list or another collection:
`value in collection` tests if the value is in the collection, returning `True`/`False`.

```python
list = ['larry', 'curly', 'moe']
if 'curly' in list:
    print('yay')
```

The for/in constructs are very commonly used in Python code and work on data types other than list, 
so you should just memorize their syntax. You may have habits from other languages where you start 
manually iterating over a collection, where in Python you should just use for/in.

You can also use for/in to work on a string. The string acts like a list of its chars, so 

```python
for ch in s: 
    print(ch)
```

prints all the chars in a string.

### Range

The `range(n)` function yields the numbers 0, 1, ... n-1, and `range(a, b)` returns a, a+1, ... b-1 -- up to **but 
not including** the last number. The combination of the for-loop and the **range()** function allow you to build 
a traditional numeric for loop:

```python
# print the numbers from 0 through 99
for i in range(100):
    print(i)
```

### While Loop

Python also has the standard while-loop. The above for/in loops solves the common case of iterating over 
every element in a list, but the while loop gives you total control over the index numbers.

Its syntaxis is quite simple:

```python
while "some boolean condition":
    # block of code that gets executed in each iteration
    ...
```

Here's a while loop which accesses every 3rd element in a list:

```python
# Access every 3rd element in a list
i = 0
while i < len(a):
    print(a[i])
    i = i + 3
```

As in other languages, we have the `break` and `continue` statements:

- `break` finishes the while loop
- `continue` moves on to the next iteration

```python
a = [1, 2, 3, 4, 5, 6, 7, 8]
i = 0
while i < len(a):
    print(a[i])
    if i > 4:
        break # finish while loop
    elif i == 2:
        print("i is 2!")
        i += 2
        continue # go back to the beginning of the while loop
    i += 1    
```

##  Data structures

Python has some built-in data structures that are very convenient to use. 

* **Lists**: an ordered collection of elements
* **Dictionaries**: a collection of elements, indexed by keys
* **Sets**: a collection of elements with no order, indexes or repeated elements
* **Tuples**: an ordered collection of elements that, unlike lists, cannot be modified

### Lists

We have seen an introduction to lists in the previous loops section. We will see a few other details here:

#### List methods

Here are some common list methods.

- `list.append(elem)` -- adds a single element to the end of the list. Common error: does not return the new list, just modifies the original.
- `list.insert(index, elem)` -- inserts the element at the given index, shifting elements to the right.
- `list.extend(list2)` adds the elements in list2 to the end of the list. Using + or += on a list is similar to using extend().
- `list.index(elem)` -- searches for the given element from the start of the list and returns its index. Throws a ValueError if the element does not appear (use "in" to check without a ValueError).
- `list.remove(elem)` -- searches for the first instance of the given element and removes it (throws ValueError if not present)
- `list.sort()` -- sorts the list in place (does not return it). (The sorted() function shown later is preferred.)
- `list.reverse()` -- reverses the list in place (does not return it)
- `list.pop(index)` -- removes and returns the element at the given index. Returns the rightmost element if index is omitted (roughly the opposite of append()).

Notice that these are *methods* on a list object, while `len()` is a function that takes the list 
(or string or whatever) as an argument.

```python
list = ['larry', 'curly', 'moe']
list.append('shemp')         ## append elem at end
list.insert(0, 'xxx')        ## insert elem at index 0
list.extend(['yyy', 'zzz'])  ## add list of elems at end
print(list)  ## ['xxx', 'larry', 'curly', 'moe', 'shemp', 'yyy', 'zzz']
print(list.index('curly'))    ## 2

list.remove('curly')         ## search and remove that element
list.pop(1)                  ## removes and returns 'larry'
print(list)  ## ['xxx', 'moe', 'shemp', 'yyy', 'zzz']
```

!!!note
    **Common error**: note that the above methods do not *return* the modified list, they just 
    modify the original list.

```python
list = [1, 2, 3]
print(list.append(4))   ## NO, does not work, append() returns None
## Correct pattern:
list.append(4)
print(list)  ## [1, 2, 3, 4]
```

#### Building up a list

One common pattern is to start a list as the empty list [], then use append() or extend() to add elements to it:

```python
list = []          ## Start as the empty list
list.append('a')   ## Use append() to add elements
list.append('b')
```

#### List Slices

Slices work on lists just as with strings, and can also be used to change sub-parts of the list.

```python
list = ['a', 'b', 'c', 'd']
print(list[1:-1])   ## ['b', 'c']
list[0:2] = 'z'    ## replace ['a', 'b'] with ['z']
print(list)         ## ['z', 'c', 'd']
```

### Dictionaries

Python provides another composite data type called a dictionary, which is similar to a list in that it is a 
collection of objects.

Dictionaries and lists share the following characteristics:

- Both are mutable.
- Both are dynamic. They can grow and shrink as needed.
- Both can be nested. A list can contain another list. A dictionary can contain another dictionary. 
A dictionary can also contain a list, and vice versa.

Dictionaries differ from lists primarily in how elements are accessed:

- List elements are accessed by their position in the list, via indexing. 
- Dictionary elements are accessed via keys.

Defining a Dictionary

A dictionary consists of a collection of key-value pairs. Each key-value pair maps the key to its associated value.
You can define a dictionary by enclosing a comma-separated list of key-value pairs in curly braces (`{}`). A colon 
(`:`) separates each key from its associated value:

```python
d = {
    <key>: <value>,
    <key>: <value>,
      .
      .
      .
    <key>: <value>
}
```
For example:
```python
football_teams = {
    "bilbao": "athletic",
    "barcelona": "barça",
    "madrid": "real madrid",
    "munich": "bayern",
    "paris": "psg"
}
```

Of course, dictionary elements must be accessible somehow. If you don’t get them by index, then how do you get them?
A value is retrieved from a dictionary by specifying its corresponding key in square brackets (`[]`):

```python
>>> football_teams['bilbao']
'athletic'
>>> football_teams['paris']
'psg'
```
If you refer to a key that is not in the dictionary, Python raises an exception:
```python
>>> football_teams['toronto']
KeyError: 'Toronto'
```

Defining a dictionary using curly braces and a list of key-value pairs, as shown above, is fine 
if you know all the keys and values in advance. But what if you want to build a dictionary on the fly?

You can start by creating an empty dictionary, which is specified by empty curly braces. Then you can 
add new keys and values one at a time:

```python
>>> person = {}
>>> type(person)
<class 'dict'>

person['fname'] = 'Joe'
person['lname'] = 'Fonebone'
person['age'] = 51
person['spouse'] = 'Edna'
person['children'] = ['Ralph', 'Betty', 'Joey']
person['pets'] = {'dog': 'Fido', 'cat': 'Sox'}
```

!!!note
    In dictionaries, a given key can appear in **only once**. 
    Duplicate keys are not allowed. A dictionary maps each key to a corresponding value, 
    so it doesn’t make sense to map a particular key more than once. 

#### Iterating over dictionaries

When you’re working with dictionaries, it’s likely that you’ll want to work with both the keys 
and the values. One of the most useful ways to iterate through a dictionary in Python is by 
using `.items()`, which is a method that returns a new view of the dictionary’s items:

```python
>>> a_dict = {'color': 'blue', 'fruit': 'apple', 'pet': 'dog'}
>>> d_items = a_dict.items()
>>> d_items  # Here d_items is a view of items
dict_items([('color', 'blue'), ('fruit', 'apple'), ('pet', 'dog')])
```

To iterate through the keys and values of the dictionary, we will just need to "unpack" the elements of 
the dict like so:

```python
for key, value in a_dict.items():
    print(key, '->', value)

# Output
color -> blue
fruit -> apple
pet -> dog
```

#### Dict important methods

- `clear()`	Removes all the elements from the dictionary
- `copy()`	Returns a copy of the dictionary
- `fromkeys()`	Returns a dictionary with the specified keys and value
- `get()`	Returns the value of the specified key
- `items()`	Returns a list containing a tuple for each key value pair
- `keys()`	Returns a list containing the dictionary's keys
- `pop()`	Removes the element with the specified key
- `popitem()`	Removes the last inserted key-value pair
- `setdefault()`	Returns the value of the specified key. If the key does not exist: insert the key, with the specified value
- `update()`	Updates the dictionary with the specified key-value pairs
- `values()`	Returns a list of all the values in the dictionary

### Sets and tuples

**Sets** and **tuples** are the last of Python main data structures. 

In Python, we create **sets** by placing all the elements inside curly braces {}, separated by comma.

A set can have any number of items and they may be of different types (integer, float, tuple, string etc.). 
But a set cannot have mutable elements like lists, sets or dictionaries as its elements.

```python
# create a set of integer type
student_id = {112, 114, 116, 118, 115}
print('Student ID:', student_id)

# create a set of string type
vowel_letters = {'a', 'e', 'i', 'o', 'u'}
print('Vowel Letters:', vowel_letters)

# create a set of mixed data types
mixed_set = {'Hello', 101, -2, 'Bye'}
print('Set of mixed data types:', mixed_set)

# Output
Student ID: {112, 114, 115, 116, 118}
Vowel Letters: {'u', 'a', 'e', 'i', 'o'}
Set of mixed data types: {'Hello', 'Bye', 101, -2}
```

Sets cannot have duplicate elements. Let's see what will happen if we try to include duplicate items in a set:
```python
numbers = {2, 4, 6, 6, 2, 8}
print(numbers)   

# Output
{8, 2, 4, 6}
```

A tuple is created by placing all the items (elements) inside parentheses (), separated by commas. 
A tuple can have any number of items and they may be of different types (integer, float, list, string, etc.).

```python
# Different types of tuples

# Empty tuple
my_tuple = ()
print(my_tuple)

# Tuple having integers
my_tuple = (1, 2, 3)
print(my_tuple)

# tuple with mixed datatypes
my_tuple = (1, "Hello", 3.4)
print(my_tuple)

# nested tuple
my_tuple = ("mouse", [8, 4, 6], (1, 2, 3))
print(my_tuple)

# Output

()
(1, 2, 3)
(1, 'Hello', 3.4)
('mouse', [8, 4, 6], (1, 2, 3))
```

!!!note
    When defining a tuple, parentheses are optional (although it is good practice to use them)

Like lists, tuples allow slicing and indexing. However, unlike lists, tuples are **immutable**: once 
defined, they cannot be changed.

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

### Basic exception handling

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

## Classes and Object-Oriented-Programming

Objects and classes are a way to group up a bunch of function and variables into a single "thing". When you get 
all the way down to it, this is simply a way of organizing everything into groups that make sense. 
There are benefits down the road for making things easier to understand, debug, extend, or maintain, but 
basically it is just a way to make code easy to understand and develop.

This might sound very abstract, but in practice it is not: everything is an object in Python! 
```int`, `dict`, `list```... etc. are all different types of objects.

To be more precise, ```int`, `dict`, `list```... are known as **classes**, and objects are particular instances
of such classes. In the next example: 
```python
a = 2
```
`int` would be the class, and `a` would be the object (or instance, they are used interchangeably). So, 
in other words, classes are the blueprint from which objects are created.

!!!note 
    It is often helpful to make analogies of classes with animals: for instance, if `dog` is a class,
    then `Max = dog()` (one dog in particular) is the instance.

Making use of classes in programming is what is known as **Object-Oriented-Programming** (OOP). OOP is highly 
concerned with code organization, reusability, and encapsulation. OOP is partially in contrast to **Functional 
Programming**, which is a different paradigm used a lot in Python. Not everyone who programs in Python 
uses OOP. 

!!!note
    So, should we use functions or classes when programming? As a general rule of thumb, we can think of classes/objects 
    as _nouns_ and functions/methods as _verbs_. In other words, functions **do** specific things, classes **are** 
    specific things. With a language like Python we are not forced to choose: we can use just a little bit of each,
    when it is most convenient.

### Defining classes

Python makes it very easy to define our own classes. A very basic class would look something like this:

```python
class MyClass:
    variable = "blah"

    def function(self):
        print("This is a message inside the class.")
```

!!!note
    Variables in classes are also known as _attributes_, and functions in classes are also known as _methods_.
    These words are complete synonyms.

We'll explain why you have to include that `self` as a parameter a little bit later. 
First, to assign the above class to an object you would do the following:

```python
class MyClass:
    variable = "blah"

    def function(self):
        print("This is a message inside the class.")

myobjectx = MyClass()
```

Now the variable `myobjectx` holds an object of the class `MyClass` that contains the variable 
and the function defined within the class called `MyClass`.
Accessing Object Variables

### Accessing object variables

To access the variable inside the newly created object `myobjectx` you would do the following:
```python
class MyClass:
    variable = "blah"

    def function(self):
        print("This is a message inside the class.")

myobjectx = MyClass()

myobjectx.variable
```

So for instance the below would output the string `blah`:
```python
class MyClass:
    variable = "blah"

    def function(self):
        print("This is a message inside the class.")

myobjectx = MyClass()

print(myobjectx.variable)
```

You can create multiple different objects that are of the same class (i.e., have the same variables and functions 
defined). However, each object contains independent copies of the variables defined in the class. For instance, 
if we were to define another object with the `MyClass` class and then change the string in the variable above:

```python
class MyClass:
    variable = "blah"

    def function(self):
        print("This is a message inside the class.")

myobjectx = MyClass()
myobjecty = MyClass()

myobjecty.variable = "yackity"

# Then print out both values
print(myobjectx.variable)
print(myobjecty.variable)
```
nothing would happen to the first object.

### Accessing object functions

To access a function inside of an object you use notation similar to accessing a variable:

```python
class MyClass:
    variable = "blah"

    def function(self):
        print("This is a message inside the class.")

myobjectx = MyClass()

myobjectx.function()
```

The above would print out the message `This is a message inside the class.`

#### `init()` function

The `__init__()` function is a special function that is called when the class is being instantiated. 
`__init__` doesn't _initialize_ a class, it initializes an instance of a class or an object. 

The difference between variables assigned inside the `__init__()` method versus variables assigned
in the class definition is that in the first case we are only defining values for one particular
instance, whereas if we define a value in the class definition, all objects from that class will have
the same value. 

!!!note
    In other words, each dog has colour, but dogs as a class don't: hence the color for one particular dog
    should be defined inside the `__init__()` method. 

    The class is a concept of an object. When you see Fido and Spot, you recognise their similarity, 
    their doghood. That's the class. However, when you say
    ```python
    class Dog:
        def __init__(self, legs, colour):
            self.legs = legs
            self.colour = colour
    
    fido = Dog(4, "brown")
    spot = Dog(3, "mostly yellow")
    ```
    
    You're saying, Fido is a brown dog with 4 legs while Spot is a bit of a cripple and is mostly yellow. 

### Class inheritance

**Inheritance** is a mechanism that allows you to create a hierarchy of classes that share a set of 
properties and methods by deriving a class from another class. This is extremely useful because it allows
us to write _less code_ thanks to code reuse.

For instance, imagine that we need to define a class for a dogs and cats (and potentially many more). Instead of 
writing everything every time for each other animal that we define, we could make them inherit from another parent
class `Animal`:

```python
class Animal:
    def __init__(self, age, height):
        self.age = age
        self.height = height
    def print_height(self):
        print(self.height)
                
class Dog(Animal):
    def __init__(self, age, height, race):
        super().__init__(age, height)
        self.race = race        
        
class Cat(Animal):
    def __init__(self, age, height, meowness):
        super().__init__(age, height)
        self.meowness = meowness
        
my_cat = Cat(age=2, height=35, meowness=0.5)
some_dog = Dog(age=1, height=50, race="beagle")

my_cat.print_height()
print(some_dog.race)

# Output
35
beagle
```

## Package distribution

A `.whl` (**wheel**) file is a distribution package file saved in Python’s wheel format. It is a standard format 
installation of Python distributions and contains all the files and metadata required for installation. 
The WHL file also contains information about the Python versions and platforms supported by this wheel file. 
WHL file format is a ready-to-install format that allows running the installation package without building the 
source distribution.

!!!note
    * All else being equal, wheels are typically smaller in size than source distributions.
    * Installing from wheels directly avoids the intermediate step of building packages off of 
    the source distribution.

A `.whl` file is essentially a zip archive with a specially crafted filename that tells installers what 
Python versions and platforms the wheel will support.
