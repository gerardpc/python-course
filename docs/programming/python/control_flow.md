# Control flow

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
