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
    Although Series are 1D, nothing prevents use from converting a Pandas Series into a Pandas DataFrame.
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

To select rows with the `.loc[]` operator we use the syntax `my_dataframe.loc[row_selector, :]`. 
The `:` indicates that we want to select all columns (but we could also put in here any list of columns
we want).

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

We can use this to fill cells without values:

```python
df.loc[df["a"].isna(), :] = 0
```

!!!note
    Pandas primarily uses the value `np.nan` to represent missing data. Missing data is by default 
    not included in computations

### Operations on DataFrames

#### Basic statistics, apply, etc.

#### Group by

#### Joins and merges
