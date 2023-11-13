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

### Ubuntu

The easiest way to start using PySpark is with a docker image. We have two options

### Option 1: running a Jupyter notebook

Get the docker image from [here](https://hub.docker.com/r/jupyter/pyspark-notebook) by running:
```bash
docker pull jupyter/pyspark-notebook
```
Then run the image with
```bash
docker run -it --rm -p 8888:8888 jupyter/pyspark-notebook
```
After this two steps we can already write/run Python code with PySpark (from the notebook).

### Option 2: running PySpark from an IDE (e.g. PyCharm)

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
* `reduce(func)`: aggregate the elements of the dataset using a function func (which takes two arguments 
and returns one).

### Pandas vs Spark

Spark DataFrames are conceptually equivalent to a Pandas DataFrame. The main difference is that
Spark DataFrames are immutable, meaning that they cannot be changed after they are created. This
allows Spark to do more optimization under the hood.

## Working with PySpark DataFrames

### DataFrame Creation

A PySpark DataFrame can be created via `pyspark.sql.SparkSession.createDataFrame` typically by passing a list 
of lists, tuples, dictionaries and pyspark.sql.Rows, a pandas DataFrame and an RDD consisting of such a list. 
`pyspark.sql.SparkSession.createDataFrame` takes the schema argument to specify the schema of the DataFrame.

!!!note
    A PySpark schema is a list that defines the name, type, and nullable/non-nullable information for each column
    in a DataFrame.

When the schema is omitted, PySpark infers the corresponding schema by taking a sample from the data.

Firstly, you can create a PySpark DataFrame from a list of rows


```python
from datetime import datetime, date
import pandas as pd
from pyspark.sql import Row

df = spark.createDataFrame([
    Row(a=1, b=2., c='string1', d=date(2000, 1, 1), e=datetime(2000, 1, 1, 12, 0)),
    Row(a=2, b=3., c='string2', d=date(2000, 2, 1), e=datetime(2000, 1, 2, 12, 0)),
    Row(a=4, b=5., c='string3', d=date(2000, 3, 1), e=datetime(2000, 1, 3, 12, 0))
])
df

DataFrame[a: bigint, b: double, c: string, d: date, e: timestamp]
```


Create a PySpark DataFrame with an explicit schema.

df = spark.createDataFrame([
    (1, 2., 'string1', date(2000, 1, 1), datetime(2000, 1, 1, 12, 0)),
    (2, 3., 'string2', date(2000, 2, 1), datetime(2000, 1, 2, 12, 0)),
    (3, 4., 'string3', date(2000, 3, 1), datetime(2000, 1, 3, 12, 0))
], schema='a long, b double, c string, d date, e timestamp')
df

DataFrame[a: bigint, b: double, c: string, d: date, e: timestamp]

Create a PySpark DataFrame from a pandas DataFrame

pandas_df = pd.DataFrame({
    'a': [1, 2, 3],
    'b': [2., 3., 4.],
    'c': ['string1', 'string2', 'string3'],
    'd': [date(2000, 1, 1), date(2000, 2, 1), date(2000, 3, 1)],
    'e': [datetime(2000, 1, 1, 12, 0), datetime(2000, 1, 2, 12, 0), datetime(2000, 1, 3, 12, 0)]
})
df = spark.createDataFrame(pandas_df)
df

DataFrame[a: bigint, b: double, c: string, d: date, e: timestamp]

The DataFrames created above all have the same results and schema.

# All DataFrames above result same.
df.show()
df.printSchema()

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

Viewing Data

The top rows of a DataFrame can be displayed using DataFrame.show().

df.show(1)

+---+---+-------+----------+-------------------+
|  a|  b|      c|         d|                  e|
+---+---+-------+----------+-------------------+
|  1|2.0|string1|2000-01-01|2000-01-01 12:00:00|
+---+---+-------+----------+-------------------+
only showing top 1 row

Alternatively, you can enable spark.sql.repl.eagerEval.enabled configuration for the eager evaluation of PySpark DataFrame in notebooks such as Jupyter. The number of rows to show can be controlled via spark.sql.repl.eagerEval.maxNumRows configuration.

spark.conf.set('spark.sql.repl.eagerEval.enabled', True)
df

a	b	c	d	e
1	2.0	string1	2000-01-01	2000-01-01 12:00:00
2	3.0	string2	2000-02-01	2000-01-02 12:00:00
3	4.0	string3	2000-03-01	2000-01-03 12:00:00

The rows can also be shown vertically. This is useful when rows are too long to show horizontally.

df.show(1, vertical=True)

-RECORD 0------------------
 a   | 1
 b   | 2.0
 c   | string1
 d   | 2000-01-01
 e   | 2000-01-01 12:00:00
only showing top 1 row

You can see the DataFrame’s schema and column names as follows:

df.columns

['a', 'b', 'c', 'd', 'e']

df.printSchema()

root
 |-- a: long (nullable = true)
 |-- b: double (nullable = true)
 |-- c: string (nullable = true)
 |-- d: date (nullable = true)
 |-- e: timestamp (nullable = true)

Show the summary of the DataFrame

df.select("a", "b", "c").describe().show()

+-------+---+---+-------+
|summary|  a|  b|      c|
+-------+---+---+-------+
|  count|  3|  3|      3|
|   mean|2.0|3.0|   null|
| stddev|1.0|1.0|   null|
|    min|  1|2.0|string1|
|    max|  3|4.0|string3|
+-------+---+---+-------+

DataFrame.collect() collects the distributed data to the driver side as the local data in Python. Note that this can throw an out-of-memory error when the dataset is too large to fit in the driver side because it collects all the data from executors to the driver side.

df.collect()

[Row(a=1, b=2.0, c='string1', d=datetime.date(2000, 1, 1), e=datetime.datetime(2000, 1, 1, 12, 0)),
 Row(a=2, b=3.0, c='string2', d=datetime.date(2000, 2, 1), e=datetime.datetime(2000, 1, 2, 12, 0)),
 Row(a=3, b=4.0, c='string3', d=datetime.date(2000, 3, 1), e=datetime.datetime(2000, 1, 3, 12, 0))]

In order to avoid throwing an out-of-memory exception, use DataFrame.take() or DataFrame.tail().

df.take(1)

[Row(a=1, b=2.0, c='string1', d=datetime.date(2000, 1, 1), e=datetime.datetime(2000, 1, 1, 12, 0))]

PySpark DataFrame also provides the conversion back to a pandas DataFrame to leverage pandas API. Note that toPandas also collects all data into the driver side that can easily cause an out-of-memory-error when the data is too large to fit into the driver side.

df.toPandas()

	a 	b 	c 	d 	e
0 	1 	2.0 	string1 	2000-01-01 	2000-01-01 12:00:00
1 	2 	3.0 	string2 	2000-02-01 	2000-01-02 12:00:00
2 	3 	4.0 	string3 	2000-03-01 	2000-01-03 12:00:00
Selecting and Accessing Data

PySpark DataFrame is lazily evaluated and simply selecting a column does not trigger the computation but it returns a Column instance.

df.a

Column<b'a'>

In fact, most of column-wise operations return Columns.

from pyspark.sql import Column
from pyspark.sql.functions import upper

type(df.c) == type(upper(df.c)) == type(df.c.isNull())

True

These Columns can be used to select the columns from a DataFrame. For example, DataFrame.select() takes the Column instances that returns another DataFrame.

df.select(df.c).show()

+-------+
|      c|
+-------+
|string1|
|string2|
|string3|
+-------+

Assign new Column instance.

df.withColumn('upper_c', upper(df.c)).show()

+---+---+-------+----------+-------------------+-------+
|  a|  b|      c|         d|                  e|upper_c|
+---+---+-------+----------+-------------------+-------+
|  1|2.0|string1|2000-01-01|2000-01-01 12:00:00|STRING1|
|  2|3.0|string2|2000-02-01|2000-01-02 12:00:00|STRING2|
|  3|4.0|string3|2000-03-01|2000-01-03 12:00:00|STRING3|
+---+---+-------+----------+-------------------+-------+

To select a subset of rows, use DataFrame.filter().

df.filter(df.a == 1).show()

+---+---+-------+----------+-------------------+
|  a|  b|      c|         d|                  e|
+---+---+-------+----------+-------------------+
|  1|2.0|string1|2000-01-01|2000-01-01 12:00:00|
+---+---+-------+----------+-------------------+

Applying a Function

PySpark supports various UDFs and APIs to allow users to execute Python native functions. See also the latest Pandas UDFs and Pandas Function APIs. For instance, the example below allows users to directly use the APIs in a pandas Series within Python native function.

import pandas as pd
from pyspark.sql.functions import pandas_udf

@pandas_udf('long')
def pandas_plus_one(series: pd.Series) -> pd.Series:
    # Simply plus one by using pandas Series.
    return series + 1

df.select(pandas_plus_one(df.a)).show()

+------------------+
|pandas_plus_one(a)|
+------------------+
|                 2|
|                 3|
|                 4|
+------------------+

Another example is DataFrame.mapInPandas which allows users directly use the APIs in a pandas DataFrame without any restrictions such as the result length.

def pandas_filter_func(iterator):
    for pandas_df in iterator:
        yield pandas_df[pandas_df.a == 1]

df.mapInPandas(pandas_filter_func, schema=df.schema).show()

+---+---+-------+----------+-------------------+
|  a|  b|      c|         d|                  e|
+---+---+-------+----------+-------------------+
|  1|2.0|string1|2000-01-01|2000-01-01 12:00:00|
+---+---+-------+----------+-------------------+

Grouping Data

PySpark DataFrame also provides a way of handling grouped data by using the common approach, split-apply-combine strategy. It groups the data by a certain condition applies a function to each group and then combines them back to the DataFrame.

df = spark.createDataFrame([
    ['red', 'banana', 1, 10], ['blue', 'banana', 2, 20], ['red', 'carrot', 3, 30],
    ['blue', 'grape', 4, 40], ['red', 'carrot', 5, 50], ['black', 'carrot', 6, 60],
    ['red', 'banana', 7, 70], ['red', 'grape', 8, 80]], schema=['color', 'fruit', 'v1', 'v2'])
df.show()

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

Grouping and then applying the avg() function to the resulting groups.

df.groupby('color').avg().show()

+-----+-------+-------+
|color|avg(v1)|avg(v2)|
+-----+-------+-------+
|  red|    4.8|   48.0|
|black|    6.0|   60.0|
| blue|    3.0|   30.0|
+-----+-------+-------+

You can also apply a Python native function against each group by using pandas API.

def plus_mean(pandas_df):
    return pandas_df.assign(v1=pandas_df.v1 - pandas_df.v1.mean())

df.groupby('color').applyInPandas(plus_mean, schema=df.schema).show()

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

Co-grouping and applying a function.

df1 = spark.createDataFrame(
    [(20000101, 1, 1.0), (20000101, 2, 2.0), (20000102, 1, 3.0), (20000102, 2, 4.0)],
    ('time', 'id', 'v1'))

df2 = spark.createDataFrame(
    [(20000101, 1, 'x'), (20000101, 2, 'y')],
    ('time', 'id', 'v2'))

def merge_ordered(l, r):
    return pd.merge_ordered(l, r)

df1.groupby('id').cogroup(df2.groupby('id')).applyInPandas(
    merge_ordered, schema='time int, id int, v1 double, v2 string').show()

+--------+---+---+---+
|    time| id| v1| v2|
+--------+---+---+---+
|20000101|  1|1.0|  x|
|20000102|  1|3.0|  x|
|20000101|  2|2.0|  y|
|20000102|  2|4.0|  y|
+--------+---+---+---+

Getting Data In/Out

CSV is straightforward and easy to use. Parquet and ORC are efficient and compact file formats to read and write faster.

There are many other data sources available in PySpark such as JDBC, text, binaryFile, Avro, etc. See also the latest Spark SQL, DataFrames and Datasets Guide in Apache Spark documentation.
CSV

df.write.csv('foo.csv', header=True)
spark.read.csv('foo.csv', header=True).show()

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

Parquet

df.write.parquet('bar.parquet')
spark.read.parquet('bar.parquet').show()

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

ORC

df.write.orc('zoo.orc')
spark.read.orc('zoo.orc').show()

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

Working with SQL

DataFrame and Spark SQL share the same execution engine so they can be interchangeably used seamlessly. For example, you can register the DataFrame as a table and run a SQL easily as below:

df.createOrReplaceTempView("tableA")
spark.sql("SELECT count(*) from tableA").show()

+--------+
|count(1)|
+--------+
|       8|
+--------+

In addition, UDFs can be registered and invoked in SQL out of the box:

@pandas_udf("integer")
def add_one(s: pd.Series) -> pd.Series:
    return s + 1

spark.udf.register("add_one", add_one)
spark.sql("SELECT add_one(v1) FROM tableA").show()

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

These SQL expressions can directly be mixed and used as PySpark columns.

from pyspark.sql.functions import expr

df.selectExpr('add_one(v1)').show()
df.select(expr('count(*)') > 0).show()

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

