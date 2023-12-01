# Data structures

## Data structures

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

### Sets 

In Python, we create **sets** by placing all the elements inside curly braces `{}`, separated by commas.

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

#### Set methods

Some of the important methods of Python sets are listed below:

* `add()`: Adds an element to the set
* `clear()`: Removes all the elements from the set
* `copy()`:	Returns a copy of the set

Typical mathematical operations on sets:

* `difference()`: Returns a set containing the difference between two or more sets
* `intersection()`: Returns a set, that is the intersection of two other sets
* `union()`: Return a set containing the union of sets. Can also be used with the `|` operator between sets
* `symmetric_difference()`: Returns a set with the symmetric differences of two sets.


### Tuples

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
    When defining a tuple, parentheses are optional (although it is good practice to use them). 
    A tuple containing a single value must be defined with a comma, otherwise Python will not
    recognize it as a tuple.

Like lists, tuples allow slicing and indexing:

```python  
my_tuple = ('p','e','r','m','i','t')

print(my_tuple[0])
# Output: 'p'
```

However, unlike lists, tuples are **immutable**: once defined, they cannot be changed. For example:

```python
my_tuple = (1, 2, 3)
my_tuple[0] = 4

# Output
TypeError: 'tuple' object does not support item assignment
```