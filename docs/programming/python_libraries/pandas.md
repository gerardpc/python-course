# Pandas

## Introduction

Pandas is a high-level data manipulation tool built with the Numpy package. Its key data structure is called the 
DataFrame, which allow us to store and manipulate tabular data (we can think of the rows as different "observations" 
and the columns as the variables).

!!!note
    Intuitively, pandas DataFrames can be thought of as a way to hold Excel spreadsheets data in a
    Python object.

Since Pandas **is not** part of the standard Python library, we will need to install it first in the virtual 
environment, for instance with `pip`:

```bash
pip install pandas
```

Once we have it installed, to use it in the code will need to import it with

```python
import pandas

# Code here
...
```

!!!note
    Often, pandas is renamed to `pd` in the import, as:
    ```python
    import pandas as pd
    ```
    This is of course not mandatory, but very common.    

After importing the package, we can start using its most important object, the DataFrame. 
There are several ways to create a DataFrame, for example:

```python
dict = {"country": ["Brazil", "Russia", "India", "China", "South Africa"],
       "capital": ["Brasilia", "Moscow", "New Dehli", "Beijing", "Pretoria"],
       "area": [8.516, 17.10, 3.286, 9.597, 1.221],
       "population": [200.4, 143.5, 1252, 1357, 52.98] }

import pandas as pd
brics = pd.DataFrame(dict)
print(brics)
```
which outputs
```
     area    capital       country  population
0   8.516   Brasilia        Brazil      200.40
1  17.100     Moscow        Russia      143.50
2   3.286  New Dehli         India     1252.00
3   9.597    Beijing         China     1357.00
4   1.221   Pretoria  South Africa       52.98
```

!!!note
    If you want a nice, complete introduction and walkthrough to Pandas, it is recommended
    to check the official documentation's own [guide](https://pandas.pydata.org/docs/user_guide/10min.html).
    The notes in this page are not comprehensive.

### Pandas vs. Excel

Tasks such as data cleaning, data normalization, visualization, and statistical analysis can be performed on 
both Excel and Pandas. That being said, Pandas has some major benefits over Excel:

* Limitation by size: Excel can handle around 1 million rows, while Python can handle millions and millions of rows 
(the limitation is on PC computing power and memory).
* Complex data transformation: Memory-intensive computations in Excel can crash a workbook. 
Python can handle complex computations without major problems.
* Automation: Excel was not designed to automate tasks. You can create a macro or use VBA to simplify some tasks, 
but Python is a general programming language (meaning we can program almost anything).

### Basic objects: Series and Dataframes

Fundamentally, Pandas has two main objects that we will be using: 

* **Pandas Series** are one-dimensional labeled arrays capable of holding any data 
type (integers, strings, floating point numbers, Python objects, etc.). We can think of a Pandas Series
as a single "column" of a table.
* **Pandas Dataframes** are 2-dimensional labeled data structures with columns of potentially different 
types. We can think of it like a spreadsheet or SQL table, or a dict of Series objects. 

!!!note
    In general we will only be using Pandas Dataframes, which are the most commonly used pandas object.
    However, since Pandas Dataframes are a collection of Pandas Series, Series will appear often as the
    result of using some of the methods of Dataframes.

### Series creation

In a Series object, the axis labels are collectively referred to as the **index**. The basic method 
to create a Series is to call:

```python
s = pd.Series(data, index=index)
```

Here, data can be many different things, for example:

* A Python dict or list
* A scalar value
* A numpy array

If `data` is a numpy array, index must be the same length as data. If no index is passed, 
one will be created having values `[0, ..., len(data) - 1]`.

```python
s = pd.Series(np.random.randn(5), index=["a", "b", "c", "d", "e"])

s
Out[4]: 
a    0.469112
b   -0.282863
c   -1.509059
d   -1.135632
e    1.212112
dtype: float64

s.index
Out[5]: Index(['a', 'b', 'c', 'd', 'e'], dtype='object')

pd.Series(np.random.randn(5))
Out[6]: 
0   -0.173215
1    0.119209
2   -1.044236
3   -0.861849
4   -2.104569
dtype: float64
```

### Dataframe creation

DataFrames are, in practice, a dictionary of columns. That's why **column labels must be unique**. 
Each column corresponds to a `Series`, and different columns can be of different type. Rows are 
identified by an index, shared by all the columns (and may be non-unique). 

To instantiate a DataFrame we may use different kinds of input, such as:

* A `dict` of 1D objects: Numpy arrays, lists, dicts, or Series. In any case, the arrays must all be the same length. 
If an index is passed, it must also be the same length as the arrays. If no index is passed, the result will be 
`range(n)`, where `n` is the array length:

    ```python
    d = {"one": [1.0, 2.0, 3.0, 4.0], "two": [4.0, 3.0, 2.0, 1.0]}
    
    pd.DataFrame(d)
    Out[46]: 
       one  two
    0  1.0  4.0
    1  2.0  3.0
    2  3.0  2.0
    3  4.0  1.0
    
    pd.DataFrame(d, index=["a", "b", "c", "d"])
    Out[47]: 
       one  two
    a  1.0  4.0
    b  2.0  3.0
    c  3.0  2.0
    d  4.0  1.0
    ```
  
* From a list of dicts (where each dict has the same keys, which correspond to the DataFrame-to-be columns):

    ```python
    In [53]: data2 = [{"a": 1, "b": 2}, {"a": 5, "b": 10, "c": 20}]
    
    In [54]: pd.DataFrame(data2)
    Out[54]: 
       a   b     c
    0  1   2   NaN
    1  5  10  20.0
    
    In [55]: pd.DataFrame(data2, index=["first", "second"])
    Out[55]: 
            a   b     c
    first   1   2   NaN
    second  5  10  20.0
    
    In [56]: pd.DataFrame(data2, columns=["a", "b"])
    Out[56]: 
       a   b
    0  1   2
    1  5  10
    ```
  
* Tables from a `.csv` file or Excel file (e.g., `.xlsx`)
* A single Pandas Series
* 2-D numpy.ndarray

In any case, the DataFrame construction always follows the same syntax:
```python
import pandas as pd

# suppose "a" is a Python object type that can be converted to a DataFrame
a = ...

# DataFrame object creation
df = pd.DataFrame(a)
```

!!!note
    Although Series are 1D, nothing prevents us from converting a Pandas Series into a Pandas DataFrame.
    This is often very convenient since DataFrames and Series do not have the same methods available.

### Reading/writing dataframes

Dataframes are very useful, but how can we transfer tabular information between different Python sessions?

!!!note
    Remember that information stored in Python objects disappears once the current execution is finished.

The most common way to achieve this is to save the tables in files. These files can be reused in future executions,
or transferred through the internet (this is mostly how we download datasets from the internet).

A simple way to store big data sets is to use `.csv` files (comma separated files). CSV files contain
plain text and is a well known format that can be read by everyone (including Pandas).

#### Reading from a CSV

Reading from a CSV is as easy as:

```python
import pandas as pd

df = pd.read_csv('data.csv')

print(df.to_string()) 

# Output

     Duration  Pulse  Maxpulse  Calories
0          60    110       130     409.1
1          60    117       145     479.0
2          60    103       135     340.0
3          45    109       175     282.4
```

If the file has a header, or uses different characters (instead of commas), the pandas documentation has
several examples on how to use the `.read_csv()` method.

!!!note
    In the previous example, the `to_string()` method is not really necessary. This method is used
    to make sure that we are printing the whole DataFrame (otherwise, if it has many columns and rows, 
    pandas might shorten it when we try to print).

#### Saving to a CSV

Saving an already existing DataFrame `df` into a `.csv` can be accomplished with:

```python
df.to_csv('file_name.csv')
```

If you want to export without the index, simply add index=False;

```python
df.to_csv('file_name.csv', index=False)
```

If you get a `UnicodeEncodeError`, simply add encoding='utf-8':

```python
df.to_csv('file_name.csv', encoding='utf-8')
```

### Viewing DataFrames

To view the contents of a DataFrame, we can use one of several options:

* Using the `print(some_dataframe)` function
* To view the top of a dataframe, use `DataFrame.head()`
* To view the bottom of a dataframe, use `DataFrame.tail()`

```python
In [13]: df.head()
Out[13]: 
                   A         B         C         D
2013-01-01  0.469112 -0.282863 -1.509059 -1.135632
2013-01-02  1.212112 -0.173215  0.119209 -1.044236
2013-01-03 -0.861849 -2.104569 -0.494929  1.071804
2013-01-04  0.721555 -0.706771 -1.039575  0.271860
2013-01-05 -0.424972  0.567020  0.276232 -1.087401

In [14]: df.tail(3)
Out[14]: 
                   A         B         C         D
2013-01-04  0.721555 -0.706771 -1.039575  0.271860
2013-01-05 -0.424972  0.567020  0.276232 -1.087401
2013-01-06 -0.673690  0.113648 -1.478427  0.524988
```

We can also explore the data with an IDE such as PyCharm, or we can export the DataFrame to a `.csv` file 
and open it with a spreadsheet editor.


### Accessing data in a DataFrame

There are 3 options to access the data of a pandas object:

* `[]`, the "standard getter" which is highly overloaded
* `.loc[]`, label based selector
* `.iloc[],` integer location based selector

!!!note
    The `[]` operator is highly overloaded and its use is discouraged. Use `.loc[]` and `.iloc[]` whenever possible.

#### Accessing data in one ore more columns

We can either use the standard getter `[]` or `.loc[]`.

With the standard getter it works like this (the return type is another DataFrame):

```python
my_dataframe = pd.DataFrame(
  {"colA": [10, 20, 30], 4.5: ["Alice", "Eve", "Bob"]},
  index=["a", "b", "b"],
)
>>> my_dataframe[[4.5]]
     4.5
a  Alice
b    Eve
b    Bob
>>> my_dataframe[[4.5, "colA"]]
     4.5  colA
a  Alice    10
b    Eve    20
b    Bob    30
```

To select columns with the `.loc[]` operator we use the syntax `my_dataframe.loc[:, col_selector]`. 
The `:` indicates that we want to select all rows (more on that later), and `col_selector` should be a
list of the columns that we want to select:

```python
my_dataframe = pd.DataFrame(
  {"colA": [10, 20, 30, 40], 4.5: ["Alice", "Eve", "Bob", "Charlie"]},
  index=["a", "b", "b", "c"],
)
>>> my_dataframe.loc[:, [4.5]]  # returns a DataFrame
     4.5
a  Alice
b    Eve
b    Bob
c    Charlie
>>> my_dataframe.loc[:, [4.5, "colA"]]  # returns a DataFrame
     4.5  colA
a  Alice     10
b    Eve     20
b    Bob     30
c    Charlie 40
```

!!!note
    In both examples we could have used a column name, instead of a list of column names, 
    as a selector. In this case, the return type, instead of a DataFrame, would be a Pandas
    Series. In general, **try to avoid working with Series whenever possible** and only use
    methods that return DataFrames.

#### Slicing

Slices act on row index, and are a way to access data from row X to row Y following the same 
logic as explained before: a slice `[1:3]` will return all members from 1 to 3, except for the last one
(i.e., 3 will be excluded). The return is a DataFrame:

```python
my_dataframe = pd.DataFrame(
  {"colA": [10, 20, 30, 40], 4.5: ["Alice", "Eve", "Bob", "Charlie"]},
  index=["a", "b", "b", "c"],
)
>>> my_dataframe[1:3]
   colA  4.5
b    20  Eve
b    30  Bob
>>> my_dataframe[:"b"]
   colA    4.5
a    10  Alice
b    20    Eve
b    30    Bob
```

#### Selecting rows and columns at the same time

To select rows with the `.loc[]` operator we use the syntax `my_dataframe.loc[row_selector, column_selector]`. 
As before, `column_selector` can be `:`, which indicates that we want to select all columns (but we could also 
put in here any list of columns we want).

!!!note
    `my_dataframe.loc[row_selector, :]` can also be written as `my_dataframe.loc[row_selector]`, without the
    `:`. However, the first version is more explicit, and so its use is recommended.

We can select rows based on row name, row subset (by using a list, as with columns), a slice or, interestingly,
based on boolean conditions:

```python
my_dataframe = pd.DataFrame(
  {"colA": [10, 20, 30, 40], 4.5: ["Alice", "Eve", "Bob", "Charlie"]},
  index=["a", "b", "b", "c"],
)

>>> my_dataframe.loc[["c", "a"], :]
   colA      4.5
c    40  Charlie
a    10    Alice


>>> my_dataframe.loc[3:2, :]
   colA      4.5
3    20      Eve
1    30      Bob
2    40  Charlie

>>> my_dataframe.loc[my_dataframe["colA"] > 30, 4.5]
       4.5
c  Charlie 
```

!!!note 
    In the last example we have used a simple "greater than" boolean condition to select data in the DataFrame,
    but we can make this boolean condition as sophisticated or complex as we want, for example
    ```python
    df.loc[(df["a"] > 0) & (df["b"] == df["d"]), ["a", "c", "g"]
    ```

#### Accessing data from its location on the DataFrame

The `.iloc[]` operator works the same as `.loc[]`, but on integer location of rows and also columns. 
The syntax is the same, `my_dataframe[row_selector, column_selector]` and you can use `:` to indicate all rows or columns.

##### Column selectors

* Selecting a single column (`my_dataframe.iloc[:, 1]`) always returns a Series.
* Selecting a list of columns (`my_dataframe.iloc[:, [1]]`) always returns a DataFrame.

##### Row selectors

* Selecting a single integer position (`my_dataframe.iloc[1, :]`) always returns a Series.
* Selecting with a list of integer positions (`my_dataframe.iloc[[1, 5], :]`) always returns a DataFrame.
* Selecting with a slice of integer positions (`my_dataframe.iloc[1:4, :]`) always returns a DataFrame. 
Recall that the upper limit **is excluded**.
* Selecting with a list of boolean indexes (`my_dataframe.iloc[1:4, :]`) always returns a DataFrame

### Setting new values on a DataFrame

We can use the same methods that have been shown to access data _to set_ new values to the DataFrame.
We just need to put the dataframe cell selection on the **left side** of a Python assignment, for example:

```python
# this assigns the value 10 to all cells in the column "D"
df.loc[:, "D"] = 10

# this assigns the value 0 to all cells with negative values
df.loc[df < 0, :] = 0
```

### Working with DataFrames

This section describes very common methods or tasks that you will be performing with pandas DataFrames.
Before we start, however, one word of caution: DataFrames are **objects**, that derive from the 
`pandas.DataFrame` class. As objects, they have a lot of available methods that have been implemented
for the class, and we may think that when we call one of this methods, the original DataFrame might be
modified (since, after all, methods can change the attributes of the class). However, in pandas **this
is not the case**. 

!!!note
    By default, in pandas, dataframe operations return a copy of the dataframe and _leave the original 
    dataframe_ data intact.

Let's see this with an example. Assume we have the following dataframe `df`:
```python
print(df)

# Output
  col1  col2  col3 col4
0    A     2     0    a
1    A     1     1    B
2    B     9     9    c
3  NaN     8     4    D
4    D     7     2    e
5    C     4     3    F
```
If we sort it by `col1` and then print `df` again, the result **will not change**:
```python
df.sort_values(by=['col1'])
print(df)

# Output
  col1  col2  col3 col4
0    A     2     0    a
1    A     1     1    B
2    B     9     9    c
3  NaN     8     4    D
4    D     7     2    e
5    C     4     3    F
```
This is because these methods do not change the internal state (i.e., its attributes) of the `df` instance.
Hence, if we want to use the output of this method, we will need to save the `return` output of this methods
to another variable (or chain it):

```python
new_df = df.sort_values(by=['col1'])
print(new_df)

# Output
  col1  col2  col3 col4
0    A     2     0    a
1    A     1     1    B
2    B     9     9    c
5    C     4     3    F
4    D     7     2    e
3  NaN     8     4    D
```

Hence, if we want to "update" the DataFrame after a change, we will need to assign the output of the method
to the original variable, like:

```python
new_df = df.sort_values(by=['col1'])
```

!!!note
    We can make this functions modify the original dataframe with the optional parameter `inplace=True` (`False`
    is the default). However, in general its use is discouraged.

#### Common tasks

##### Finding the size of a DataFrame

* To display the number of rows, columns, etc.: `df.info()`
* To get the number of rows and columns: `df.shape`
* To get the number of rows: `len(df)`
* To get the number of columns: `len(df.columns)`

We also have access to an attribute `df.empty`, of `bool` type, which returns `True` if the
DataFrame is empty and `False` otherwise.

##### Removing duplicates

Pandas `drop_duplicates()` method helps in removing duplicates from Pandas Dataframes.

```python
import pandas as pd 
  
data = { 
    "A": ["TeamA", "TeamB", "TeamB", "TeamC", "TeamA"], 
    "B": [50, 40, 40, 30, 50], 
    "C": [True, False, False, False, True] 
} 
  
df = pd.DataFrame(data) 
  
print(df.drop_duplicates())

# Output
       A       B      C
0    TeamA    50    True
1    TeamB    40    False
3    TeamC    30    False
```

##### Filling missing data

We can use the `.isna()` method to detect (and fill, if we need to) cells without values:

```python
df.loc[df["a"].isna(), :] = 0
```

!!!note
    The `.isnull()` method is an alias for the `.isna()` method, so both are exactly the same.

!!!note
    Pandas primarily uses the value `np.nan` to represent missing data. Missing data is by default 
    not included in computations.


##### Resetting the index

After dropping and filtering the rows of a DataFrame, the original index values for each row remain. 
If we want to re-create the index, dropping the original values, we can do it with
```python
DataFrame.reset_index(drop=True)
```

##### Renaming columns

To rename specific columns of a DataFrame, we may use the `df.rename()` method:
```python
df = df.rename(columns={'oldName1': 'newName1', 'oldName2': 'newName2'})
```
Only the columns in the dictionary passed as input will be renamed.

##### Sorting

We can sort a DataFrame by the values of one or multiple columns, like so:

```python
df.sort_values(by=['col1', 'col2'], ascending=True)
  col1  col2  col3 col4
1    A     1     1    B
0    A     2     0    a
2    B     9     9    c
5    C     4     3    F
4    D     7     2    e
3  NaN     8     4    D
```

##### Creating new columns

There are several ways to create a new column, but the most common one follow this syntax:
```python
df["new_column_name"] = ...
```
and on the `...` we can put whatever we want. For example, if we have a list `my_list` with the same
length as the DataFrame has rows, we could do:
```python
df["new_column_name"] = my_list
```
Alternatively, we could create a new column using the values from other columns, for example:
```python
df["new_column"] = df["A"] + df["B"]
```

##### Pivoting a dataframe

`pandas.melt()` unpivots a DataFrame from wide format to long format:

```python
>>> df = pd.DataFrame({'A': {0: 'a', 1: 'b', 2: 'c'},
...                    'B': {0: 1, 1: 3, 2: 5},
...                    'C': {0: 2, 1: 4, 2: 6}})
>>> df
   A  B  C
0  a  1  2
1  b  3  4
2  c  5  6


>>> pd.melt(df, id_vars=['A'], value_vars=['B', 'C'])
   A variable  value
0  a        B      1
1  b        B      3
2  c        B      5
3  a        C      2
4  b        C      4
5  c        C      6
```


#### Basic statistics

There are many statistical operations already implemented in pandas. We can use them by selecting
a subset of the DataFrame (e.g., a few columns) and then calling these methods. Some of the most 
important are:

* The mean: `some_df.mean()`
* The median: `some_df.median()`
* The standard deviation: `some_df.std()`
* The min and max: `.min()` and `.max()`
* The number of records for each category in a column:
    ```python
    In [12]: titanic["Pclass"].value_counts()
    Out[12]: 
    Pclass
    3    491
    1    216
    2    184
    Name: count, dtype: int64
    ```

There is also a helpful `.describe()` method that gives you several of these at the same time.

#### Applying custom functions

The pandas DataFrame `apply()` function is used to apply a function along an axis of the DataFrame.
This function that we apply can be an external or a custom defined function. It works like this:

```python
import pandas as pd

df = pd.DataFrame({'A': [1, 2], 'B': [10, 20]})
def square(x):
    return x * x

df1 = df.apply(square)

print(df)
print(df1)

# Output
   A   B
0  1  10
1  2  20

   A    B
0  1  100
1  4  400
```

!!!note
    We don't need to apply the function to the whole dataframe. We can slice it and only apply
    the function to a subset of the columns.

If you look at the above example, our `square()` function is very simple. We can easily convert it 
into a **lambda** function:
```python
df1 = df.apply(lambda x: x * x)
```
The output will remain the same as in the last example.

#### Group by

The `groupby()` method is used for grouping the data according to the categories and applying a 
function to aggregate them categories. This is easier seen with an example. Suppose that 
we have the following dataframe:

```python
In [87]: df = pd.DataFrame(
   ....:     {
   ....:         "A": ["foo", "bar", "foo", "bar", "foo", "bar", "foo", "foo"],
   ....:         "B": ["one", "one", "two", "three", "two", "two", "one", "three"],
   ....:         "C": np.random.randn(8),
   ....:         "D": np.random.randn(8),
   ....:     }
   ....: )
   ....: 

In [88]: df
Out[88]: 
     A      B         C         D
0  foo    one  1.346061 -1.577585
1  bar    one  1.511763  0.396823
2  foo    two  1.627081 -0.105381
3  bar  three -0.990582 -0.532532
4  foo    two -0.441652  1.453749
5  bar    two  1.211526  1.208843
6  foo    one  0.268520 -0.080952
7  foo  three  0.024580 -0.264610
```

We can ask ourselves what happens if we combine the data from all the columns that share
values in the columns `A` and `B` (e.g., all the columns that have `foo` in column `A` and `bar` in 
column `B`). 

For this we need what's known as an aggregating function. This function gives an answer to the following
question: assume that I found all the rows that share values in `A` and `B` and, somehow, I want to group
together the values in all the remaining columns. How do I do that? Do I calculate the mean of those values?
The sum? The minimum?

With Python's `groupby()` function we may use any aggregating function we want: the important thing to understand
is that this function will be applied to the set defined by all rows where `A` and `B` are shared.

Going back to the previous dataframe, and using the `sum()` function as an example, we could use it as:

```python
>>> df.groupby(["A", "B"], as_index=False).sum()
     A      B         C         D
0  bar    one  1.467065  1.366273
1  bar  three  1.994611  1.425282
2  bar    two -0.487412 -0.627660
3  foo    one  1.910821 -0.091346
4  foo  three -0.449329  0.213259
5  foo    two -2.369447 -0.389401
```

!!!note
    In the last example the `as_index=False` is important. If we don't use it, then the group by
    will return a dataframe with a "strange" new index, created by the combination of the values
    of `A` and `B`.

#### Joins and merges

Pandas provides a single function, `merge()`, as the equivalent of standard SQL database join operations, 
but in this case between DataFrames.

Before getting into the details of how to use `merge()`, you should first understand the various forms of joins.
The idea of a merge (or _join_, as is known in SQL) is that we have two tables, and we stitch them together
on the basis of two columns (one from the first table, another from the second table) having the same value.
The resulting merged table will be the first table _glued_ with the second table, with the shared column
acting as glue. This can be done in different ways:

* **inner merge**: the resulting merged table will contain all the rows from the original tables that found a 
    match in the other table, but all rows (from either table) that find no match are discarded.
* **outer merge**: the opposite of the previous case: every row from either table will appear in the merged table,
    and if for some of them there is no match, we will get null values on the columns coming from the other table.
* **left merge**: this is an intermediate case between the inner and the outer merge. It behaves like the inner 
  merge, but we also keep the rows from the first ("left") table that have no matches.    
* **right merge**: like the left merge, but the table on the second table is _acting_ as if it was the left table
  (usually we don't use the right merge for anything, since we can perform a left merge with the two tables in 
  the opposite order).

These are some examples on how to use the `merge()` function:

```python
import pandas as pd

#create DataFrame
df1 = pd.DataFrame({'team': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
                    'points': [18, 22, 19, 14, 14, 11, 20, 28]})

df2 = pd.DataFrame({'team': ['A', 'B', 'C', 'D', 'G', 'H'],
                    'assists': [4, 9, 14, 13, 10, 8]})

#view DataFrames
print(df1)

  team  points
0    A      18
1    B      22
2    C      19
3    D      14
4    E      14
5    F      11
6    G      20
7    H      28

print(df2)

  team  assists
0    A        4
1    B        9
2    C       14
3    D       13
4    G       10
5    H        8


df1.merge(df2, on='team', how='inner')

# Output
  team  points  assists
0    A      18        4
1    B      22        9
2    C      19       14
3    D      14       13
4    G      20       10
5    H      28        8

df1.merge(df2, on='team', how='outer')

# Output
  team  points  assists
0    A      18      4.0
1    B      22      9.0
2    C      19     14.0
3    D      14     13.0
4    E      14      NaN
5    F      11      NaN
6    G      20     10.0
7    H      28      8.0

df1.merge(df2, on='team', how='left')

# Output
  team  points  assists
0    A      18      4.0
1    B      22      9.0
2    C      19     14.0
3    D      14     13.0
4    E      14      NaN
5    F      11      NaN
6    G      20     10.0
7    H      28      8.0
```

#### Concatenating DataFrames

If we have 2 or more DataFrames which share _exactly_ the same columns, we can
concatenate them like this:

```python
In [1]: df1 = pd.DataFrame(
   ...:     {
   ...:         "A": ["A0", "A1", "A2", "A3"],
   ...:         "B": ["B0", "B1", "B2", "B3"],
   ...:         "C": ["C0", "C1", "C2", "C3"],
   ...:         "D": ["D0", "D1", "D2", "D3"],
   ...:     },
   ...:     index=[0, 1, 2, 3],
   ...: )
   ...: 

In [2]: df2 = pd.DataFrame(
   ...:     {
   ...:         "A": ["A4", "A5", "A6", "A7"],
   ...:         "B": ["B4", "B5", "B6", "B7"],
   ...:         "C": ["C4", "C5", "C6", "C7"],
   ...:         "D": ["D4", "D5", "D6", "D7"],
   ...:     },
   ...:     index=[4, 5, 6, 7],
   ...: )
   ...: 

In [3]: df3 = pd.DataFrame(
   ...:     {
   ...:         "A": ["A8", "A9", "A10", "A11"],
   ...:         "B": ["B8", "B9", "B10", "B11"],
   ...:         "C": ["C8", "C9", "C10", "C11"],
   ...:         "D": ["D8", "D9", "D10", "D11"],
   ...:     },
   ...:     index=[8, 9, 10, 11],
   ...: )
   ...: 

In [4]: frames = [df1, df2, df3]

In [5]: result = pd.concat(frames)
```

#### Variable types and memory usage

A pandas DataFrame can have columns of different types. To find out what these are, we may
use `df.dtypes`:
```python
print(df.dtypes)

# Output
float              float64
int                  int64
datetime    datetime64[ns]
string              object
dtype: object
```

!!!note
    For very large datasets, make sure that you are not using more bits than you need in
    the columns. You can transform the variable types, for example `int64` to `int16`, anytime.

To actually learn how much RAM memory (in Bytes) we are using for a particular DataFrame, we
may use the `df.memory_usage(deep=True)` function:

```python
>>> df.memory_usage(deep=True)
(lists each column's full memory usage)

>>> df.memory_usage(deep=True).sum()
462432
```