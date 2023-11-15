# PySpark

## Introduction

**Spark** is a platform for cluster computing. Spark lets you spread data and computations over clusters 
with multiple nodes (think of each node as a separate computer). Splitting up your data makes it easier 
to work with very large datasets because each node only works with a small amount of data. Hence,
Spark is a common tool for what is informally known as _big data_ (for example, for cleaning it up or transforming
it into a more usable format).

As each node works on its own subset of the total data, it also carries out a part of the total 
calculations required, so that both data processing and computation are performed in parallel over 
the nodes in the cluster. 

**PySpark** is the Python API for Spark. Having a Python library that allows us to interact with Spark
using Python means that we can use Python to write code that will run on Spark clusters under the hood,
without having to learn Scala or Java.

## Setting up Spark

Setting up Spark is considerably more complicated than using Pandas, and it requires additional software. 
We should only use Spark when our data is too big to work with on a single machine. If that is not the case,
we should use Pandas instead (or a Pandas alternative, like Polars).

### Windows and Mac

Follow the course slides to set up PySpark on your computer

### Linux

The easiest way to start using PySpark is with a docker image. We have two options

#### Option 1: running a Jupyter notebook

Get the docker image from [here](https://hub.docker.com/r/jupyter/pyspark-notebook) by running:
```bash
docker pull jupyter/pyspark-notebook
```
Then run the image with
```bash
docker run -it --rm -p 8888:8888 jupyter/pyspark-notebook
```
After this two steps we can already write/run Python code with PySpark (from the notebook).

#### Option 2: running PySpark from an IDE (e.g. PyCharm)

1. Install docker following the instructions [here](https://docs.docker.com/engine/install/ubuntu/).
2. Pull the docker image following the instructions [here](https://hub.docker.com/r/apache/spark-py).

## Overview

At a high level, every Spark application consists of a driver program that runs the user’s main 
function and executes various parallel operations on a cluster. The main abstraction Spark provides 
is a resilient distributed dataset (**RDD**), which is a collection of elements partitioned across the 
nodes of the cluster that can be operated on in parallel. RDDs are created by starting with a file 
(Hadoop file system or any other Hadoop-supported file system), or an existing driver program (a Scala collection),
and transforming it. Users may persist an RDD in memory to be reused across parallel operations.

!!!note
    RDDs automatically recover from node failures.

### PySpark DataFrames

PySpark's main object is the PySpark DataFrame (which is not the same as a Pandas DataFrame). 
PySpark DataFrames are implemented on top of RDDs and are lazily evaluated. 
This means that whenever we create a PySpark DataFrame, nothing happens until we call an action on it:
Spark does not immediately compute the transformation but plans how to compute later. 

RDD actions are operations that actually return a value to the user program or write a value to storage,
and actually trigger the computation of a result. They can be:

* `collect()`: return all the elements of the dataset as an array at the driver program. 
This is usually useful after a filter or other operation that returns a sufficiently small subset of the data.
* `count()`: return the number of elements in the dataset.
* `first()`: return the first element of the dataset.
* `take(n)`: return an array with the first n elements of the dataset.
* `reduce(func)`: aggregate the elements of the dataset using a function `func`.

### Pandas vs Spark

PySpark DataFrames are conceptually equivalent to a Pandas DataFrame. The main difference is that
PySpark DataFrames are immutable, meaning that they cannot be changed after they are created. 

!!!note
    Every time we perform an operation on a PySpark Dataframe, we are actually creating a _new_
    dataframe.

This allows Spark to do more optimization under the hood.

## Working with PySpark DataFrames

### SparkSession

PySpark applications start with initializing SparkSession, which is the entry point of PySpark:

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
```

### DataFrame Creation

A PySpark DataFrame can be created via `pyspark.sql.SparkSession.createDataFrame` by several different ways.
This is similar to pandas, where we can create a DataFrame with data from different data structures.

For example, we can create a DataFrame by passing a list of lists, tuples, dictionaries or another
pandas DataFrame, and then an RDD consisting of such a list. 
`pyspark.sql.SparkSession.createDataFrame` takes the schema argument to specify the schema of the DataFrame.

!!!note
    A PySpark schema is a list that defines the name, type, and nullable/non-nullable information for each column
    in a DataFrame.

When the schema is omitted, PySpark infers the corresponding schema by taking a sample from the data.

Firstly, we can create a PySpark DataFrame from a list of rows

```python
from datetime import datetime, date
import pandas as pd
from pyspark.sql import Row

df = spark.createDataFrame([
    Row(a=1, b=2., c='string1', d=date(2000, 1, 1), e=datetime(2000, 1, 1, 12, 0)),
    Row(a=2, b=3., c='string2', d=date(2000, 2, 1), e=datetime(2000, 1, 2, 12, 0)),
    Row(a=4, b=5., c='string3', d=date(2000, 3, 1), e=datetime(2000, 1, 3, 12, 0))
])
print(df)

# Output
DataFrame[a: bigint, b: double, c: string, d: date, e: timestamp]
```

Other options to create a PySpark DataFrame:

* Create a PySpark DataFrame with an explicit schema.

```python
df = spark.createDataFrame([
    (1, 2., 'string1', date(2000, 1, 1), datetime(2000, 1, 1, 12, 0)),
    (2, 3., 'string2', date(2000, 2, 1), datetime(2000, 1, 2, 12, 0)),
    (3, 4., 'string3', date(2000, 3, 1), datetime(2000, 1, 3, 12, 0))
], schema='a long, b double, c string, d date, e timestamp')
print(df)

# Output
DataFrame[a: bigint, b: double, c: string, d: date, e: timestamp]
```

* Create a PySpark DataFrame from a pandas DataFrame

```python
pandas_df = pd.DataFrame({
    'a': [1, 2, 3],
    'b': [2., 3., 4.],
    'c': ['string1', 'string2', 'string3'],
    'd': [date(2000, 1, 1), date(2000, 2, 1), date(2000, 3, 1)],
    'e': [datetime(2000, 1, 1, 12, 0), datetime(2000, 1, 2, 12, 0), datetime(2000, 1, 3, 12, 0)]
})
df = spark.createDataFrame(pandas_df)
print(df)

# Output
DataFrame[a: bigint, b: double, c: string, d: date, e: timestamp]
```

The DataFrames created above all have the same results and schema.

```python
# All DataFrames above result same.
df.show()
df.printSchema()

# Output
+---+---+-------+----------+-------------------+
|  a|  b|      c|         d|                  e|
+---+---+-------+----------+-------------------+
|  1|2.0|string1|2000-01-01|2000-01-01 12:00:00|
|  2|3.0|string2|2000-02-01|2000-01-02 12:00:00|
|  3|4.0|string3|2000-03-01|2000-01-03 12:00:00|
+---+---+-------+----------+-------------------+

root
 |-- a: long (nullable = true)
 |-- b: double (nullable = true)
 |-- c: string (nullable = true)
 |-- d: date (nullable = true)
 |-- e: timestamp (nullable = true)
```

!!!note
    To print a dataframe in table format, you can use `df.show()`.

### Viewing Data

The top rows of a DataFrame can be displayed using `DataFrame.show(n)` (where `n` is the number of
printed rows).

```python
df.show(1)

# Output
+---+---+-------+----------+-------------------+
|  a|  b|      c|         d|                  e|
+---+---+-------+----------+-------------------+
|  1|2.0|string1|2000-01-01|2000-01-01 12:00:00|
+---+---+-------+----------+-------------------+
```

only showing top 1 row. You can see the DataFrame’s schema and column names as follows:

```python
df.columns

# Output
['a', 'b', 'c', 'd', 'e']


df.printSchema()

# Output
root
 |-- a: long (nullable = true)
 |-- b: double (nullable = true)
 |-- c: string (nullable = true)
 |-- d: date (nullable = true)
 |-- e: timestamp (nullable = true)
```

Show the summary of the DataFrame

```python
df.select("a", "b", "c").describe().show()

# Output
+-------+---+---+-------+
|summary|  a|  b|      c|
+-------+---+---+-------+
|  count|  3|  3|      3|
|   mean|2.0|3.0|   null|
| stddev|1.0|1.0|   null|
|    min|  1|2.0|string1|
|    max|  3|4.0|string3|
+-------+---+---+-------+
```

`DataFrame.collect()` collects the distributed data to the driver side as the local data in Python. 
Note that this can throw an out-of-memory error when the dataset is too large to fit in the driver side 
because it collects all the data from executors to the driver side.

```python
df.collect()

# Output
[Row(a=1, b=2.0, c='string1', d=datetime.date(2000, 1, 1), e=datetime.datetime(2000, 1, 1, 12, 0)),
 Row(a=2, b=3.0, c='string2', d=datetime.date(2000, 2, 1), e=datetime.datetime(2000, 1, 2, 12, 0)),
 Row(a=3, b=4.0, c='string3', d=datetime.date(2000, 3, 1), e=datetime.datetime(2000, 1, 3, 12, 0))]
```

In order to avoid throwing an out-of-memory exception, use `DataFrame.take(n)` or `DataFrame.tail()`.

```python
df.take(1)

# Output
[Row(a=1, b=2.0, c='string1', d=datetime.date(2000, 1, 1), e=datetime.datetime(2000, 1, 1, 12, 0))]
```

PySpark DataFrame also provides the conversion back to a pandas DataFrame to leverage pandas API. 
Note that `toPandas` also collects all data into the driver side that can easily cause an 
out-of-memory-error when the data is too large to fit into the driver side.

```python
df.toPandas()
print(df)

# Output
	a 	b 	    c 	        d 	        e
0 	1 	2.0 	string1 	2000-01-01 	2000-01-01 12:00:00
1 	2 	3.0 	string2 	2000-02-01 	2000-01-02 12:00:00
2 	3 	4.0 	string3 	2000-03-01 	2000-01-03 12:00:00
```

### Selecting and Accessing Data

PySpark DataFrame is lazily evaluated and simply selecting a column does not trigger the computation but 
it returns a Column instance.

```python
df.a

Column<b'a'>
```

In fact, most column-wise operations return Columns.

```python
from pyspark.sql import Column
from pyspark.sql.functions import upper

type(df.c) == type(upper(df.c)) == type(df.c.isNull())

True
```

These Columns can be used to select the columns from a DataFrame. For example, `DataFrame.select()` 
takes the Column instances that returns another DataFrame.

```python
df.select(df.c).show()

# Output
+-------+
|      c|
+-------+
|string1|
|string2|
|string3|
+-------+
```
To select seval columns, we would write either `df.select("col_1","col_2").show()` or
`df.select(df.col_1,df.col_2).show()` (they are equivalent).

* To assign a new column instance:

```python
df.withColumn('upper_c', upper(df.c)).show()

# Output
+---+---+-------+----------+-------------------+-------+
|  a|  b|      c|         d|                  e|upper_c|
+---+---+-------+----------+-------------------+-------+
|  1|2.0|string1|2000-01-01|2000-01-01 12:00:00|STRING1|
|  2|3.0|string2|2000-02-01|2000-01-02 12:00:00|STRING2|
|  3|4.0|string3|2000-03-01|2000-01-03 12:00:00|STRING3|
+---+---+-------+----------+-------------------+-------+
```

* To select a subset of rows, we use the filter method `DataFrame.filter()`.

```python
df.filter(df.a == 1).show()

# Output
+---+---+-------+----------+-------------------+
|  a|  b|      c|         d|                  e|
+---+---+-------+----------+-------------------+
|  1|2.0|string1|2000-01-01|2000-01-01 12:00:00|
+---+---+-------+----------+-------------------+
```

### Applying a Function

PySpark supports various user defined functions (UDFs) and APIs to allow users to execute Python native functions. 
See also the latest Pandas UDFs and Pandas Function APIs. For instance, the example below allows users to 
directly use the APIs in a pandas Series within Python native function.

```python
import pandas as pd
from pyspark.sql.functions import pandas_udf

@pandas_udf('long')
def pandas_plus_one(series: pd.Series) -> pd.Series:
    # Simply plus one by using pandas Series.
    return series + 1

df.select(pandas_plus_one(df.a)).show()

# Output
+------------------+
|pandas_plus_one(a)|
+------------------+
|                 2|
|                 3|
|                 4|
+------------------+
```

Another example is `DataFrame.mapInPandas` which allows users to directly use the APIs in a pandas DataFrame 
without any restrictions such as the result length.

```python
def pandas_filter_func(iterator):
    for pandas_df in iterator:
        yield pandas_df[pandas_df.a == 1]

df.mapInPandas(pandas_filter_func, schema=df.schema).show()

# Output
+---+---+-------+----------+-------------------+
|  a|  b|      c|         d|                  e|
+---+---+-------+----------+-------------------+
|  1|2.0|string1|2000-01-01|2000-01-01 12:00:00|
+---+---+-------+----------+-------------------+
```

### Grouping Data

PySpark DataFrame also provides a way of handling grouped data. It groups the data by a certain 
condition, applies a function to each group and then combines them back to the DataFrame.

```python
df = spark.createDataFrame([
    ['red', 'banana', 1, 10], ['blue', 'banana', 2, 20], ['red', 'carrot', 3, 30],
    ['blue', 'grape', 4, 40], ['red', 'carrot', 5, 50], ['black', 'carrot', 6, 60],
    ['red', 'banana', 7, 70], ['red', 'grape', 8, 80]], schema=['color', 'fruit', 'v1', 'v2'])
df.show()

# Output
+-----+------+---+---+
|color| fruit| v1| v2|
+-----+------+---+---+
|  red|banana|  1| 10|
| blue|banana|  2| 20|
|  red|carrot|  3| 30|
| blue| grape|  4| 40|
|  red|carrot|  5| 50|
|black|carrot|  6| 60|
|  red|banana|  7| 70|
|  red| grape|  8| 80|
+-----+------+---+---+
```

Grouping and then applying the `avg()` function to the resulting groups.

```python
df.groupby('color').avg().show()

# Output
+-----+-------+-------+
|color|avg(v1)|avg(v2)|
+-----+-------+-------+
|  red|    4.8|   48.0|
|black|    6.0|   60.0|
| blue|    3.0|   30.0|
+-----+-------+-------+
```

You can also apply a Python native function against each group by using pandas API.

```python
def plus_mean(pandas_df):
    return pandas_df.assign(v1=pandas_df.v1 - pandas_df.v1.mean())

df.groupby('color').applyInPandas(plus_mean, schema=df.schema).show()

# Output
+-----+------+---+---+
|color| fruit| v1| v2|
+-----+------+---+---+
|  red|banana| -3| 10|
|  red|carrot| -1| 30|
|  red|carrot|  0| 50|
|  red|banana|  2| 70|
|  red| grape|  3| 80|
|black|carrot|  0| 60|
| blue|banana| -1| 20|
| blue| grape|  1| 40|
+-----+------+---+---+
```

### Getting data In/Out

CSV files are straightforward and easy to use. Parquet, in contrast, is an efficient and compact file format
to read and write faster.

* CSV
```python
df.write.csv('foo.csv', header=True)
spark.read.csv('foo.csv', header=True).show()

# Output
+-----+------+---+---+
|color| fruit| v1| v2|
+-----+------+---+---+
|  red|banana|  1| 10|
| blue|banana|  2| 20|
|  red|carrot|  3| 30|
| blue| grape|  4| 40|
|  red|carrot|  5| 50|
|black|carrot|  6| 60|
|  red|banana|  7| 70|
|  red| grape|  8| 80|
+-----+------+---+---+
```

* Parquet

```python
df.write.parquet('bar.parquet')
spark.read.parquet('bar.parquet').show()

# Output
+-----+------+---+---+
|color| fruit| v1| v2|
+-----+------+---+---+
|  red|banana|  1| 10|
| blue|banana|  2| 20|
|  red|carrot|  3| 30|
| blue| grape|  4| 40|
|  red|carrot|  5| 50|
|black|carrot|  6| 60|
|  red|banana|  7| 70|
|  red| grape|  8| 80|
+-----+------+---+---+
```


### Working with SQL

An incredibly powerful option of PySpark DataFrames is that they can be queried with SQL.
For example, you can register a DataFrame as a table and run a SQL query on it easily as below:

```python
df.createOrReplaceTempView("tableA")
spark.sql("SELECT count(*) from tableA").show()

# Output
+--------+
|count(1)|
+--------+
|       8|
+--------+
```

In addition, functions can be registered and invoked in SQL out of the box:

```python
@pandas_udf("integer")
def add_one(s: pd.Series) -> pd.Series:
    return s + 1

spark.udf.register("add_one", add_one)
spark.sql("SELECT add_one(v1) FROM tableA").show()

# Output
+-----------+
|add_one(v1)|
+-----------+
|          2|
|          3|
|          4|
|          5|
|          6|
|          7|
|          8|
|          9|
+-----------+
```

These SQL expressions can directly be mixed and used as PySpark columns.

```python
from pyspark.sql.functions import expr

df.selectExpr('add_one(v1)').show()
df.select(expr('count(*)') > 0).show()

# Output
+-----------+
|add_one(v1)|
+-----------+
|          2|
|          3|
|          4|
|          5|
|          6|
|          7|
|          8|
|          9|
+-----------+

+--------------+
|(count(1) > 0)|
+--------------+
|          true|
+--------------+
```
