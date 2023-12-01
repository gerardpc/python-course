# Comprehensions

## List Comprehensions

List comprehensions provide a concise way to create lists. Common applications are to make new 
lists where each element is the result of some operations applied to each member of another sequence 
or iterable, or to create a subsequence of those elements that satisfy a certain condition.

For example, assume we want to create a list of squares, like:

```python
squares = []
for x in range(10):
    squares.append(x**2)
```

We can obtain the same result with:
```python
squares = [x**2 for x in range(10)]
```
This last snippet is an example of a list comprehension. 

List comprehensions always returns a result list. It consists of brackets containing an expression
followed by a `for` clause, then zero or more `for` or `if` clauses. The expressions can be anything,
meaning you can put in all kinds of objects in lists.

!!!note
    **Warning**: the comprehension syntax can be a bit confusing at first. If the comprehension is too long,
    it is recommended to use the normal `for` loop syntax instead, which is more readable.

### Examples

For example, this combines the elements of two lists if they are not equal:

```python
[(x, y) for x in [1,2,3] for y in [3,1,4] if x != y]
```

and it’s equivalent to:

```python
combs = []
for x in [1,2,3]:
    for y in [3,1,4]:
        if x != y:
            combs.append((x, y))
```

Some other examples:

```python
vec = [-4, -2, 0, 2, 4]
# create a new list with the values doubled
[x*2 for x in vec]
# [-8, -4, 0, 4, 8]
# filter the list to exclude negative numbers
[x for x in vec if x >= 0]
# [0, 2, 4]
# apply a function to all the elements
[abs(x) for x in vec]
# [4, 2, 0, 2, 4]
```

## Dictionary Comprehensions

Dictionary comprehensions are similar, but allow you to easily construct dictionaries. For example:

```python
{x: x**2 for x in (2, 4, 6)}
```

### Examples

Beyond the basic usage above, dictionary comprehensions can also be used to create dictionaries from 
arbitrary key and value expressions. These are some examples:

* Create a dictionary with only pairs for odd numbers:    
    ```python
    {x: x**2 for x in range(10) if x % 2 == 1}
    ```
* An example that also uses `if`:    
    ```python
    {x: x**2 for x in range(10) if x % 2 == 1}
    ```
* Create a dictionary from two lists:    
    ```python
    {x: y for x, y in zip(['a', 'b'], [1, 2])}
    ```


## Set Comprehensions

Set comprehensions are similar to list comprehensions, but return a set and not a list. Syntactically,
set comprehensions are the same as list comprehensions except that they use curly braces `{}` instead.

For example:

```python
{x for x in 'abracadabra' if x not in 'abc'}
```