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

#### Option 2: using Google Colab

Open a new google colab notebook and run
```bash
!pip install pyspark py4j
```
on the first cell.

!!!note
    This is the most straightforward way to work with PySpark, but executions are rather slow.

#### Option 3: running PySpark from an IDE (e.g. PyCharm)

1. Install docker following the instructions [here](https://docs.docker.com/engine/install/ubuntu/).
2. Pull the docker image following the instructions [here](https://hub.docker.com/r/apache/spark-py).
3. Configure Pycharm to work with a remote [docker interpreter](https://www.jetbrains.com/help/pycharm/using-docker-as-a-remote-interpreter.html#config-docker)
4. 

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

We can create a PySpark DataFrame from a list of rows:

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

But we have other options to create a PySpark DataFrame, and sometimes they are more convenient:

* Create a PySpark DataFrame with an explicit schema

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

All the DataFrames created above have the same content and schema.

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


### Getting data In/Out

Other common options to put data in/out of a DataFrame are to get data from a file.

* CSV files:
```python
df.write.csv('fruits.csv', header=True)
spark.read.csv('fruits.csv', header=True).show()

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

* Parquet is an efficient and compact file format to read and write faster.

```python
df.write.parquet('fruits.parquet')
spark.read.parquet('fruits.parquet').show()

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

only showing the first row. You can see the DataFrame’s schema and column names as follows:

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

PySpark DataFrame also provides the conversion back to a pandas DataFrame. 


```python
df.toPandas()
print(df)

# Output
	a 	b 	    c 	        d 	        e
0 	1 	2.0 	string1 	2000-01-01 	2000-01-01 12:00:00
1 	2 	3.0 	string2 	2000-02-01 	2000-01-02 12:00:00
2 	3 	4.0 	string3 	2000-03-01 	2000-01-03 12:00:00
```

!!!note 
    `toPandas` also collects all data into the driver side, and that can easily cause an 
    out-of-memory-error if the data is too large to fit into the driver side.

### Selecting and accessing data

PySpark DataFrame is lazily evaluated and simply selecting a column does not trigger a computation.
Instead, it returns a Column instance.

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

These column objects can be used to get new DataFrames that are subsets of the original one.

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
To select several columns, we would write either `df.select("col_1","col_2").show()` or
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

# Filter multiple condition
df.filter((df.state  == "OH") & (df.gender  == "M")).show()  

...
```

### Applying a Function

PySpark supports various user defined functions (UDFs) and APIs to allow users to execute Python native functions. 
For instance, the example below allows users to directly use the APIs in a pandas Series within a Python native function.

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


### Grouping Data

The DataFrame also provides a way of handling grouped data. It groups the data by a certain 
condition, applies a function to each group and then combines them back to the DataFrame.

```python
simpleData = [("James","Sales","NY",90000,34,10000),
    ("Michael","Sales","NY",86000,56,20000),
    ("Robert","Sales","CA",81000,30,23000),
    ("Maria","Finance","CA",90000,24,23000),
    ("Raman","Finance","CA",99000,40,24000),
    ("Scott","Finance","NY",83000,36,19000),
    ("Jen","Finance","NY",79000,53,15000),
    ("Jeff","Marketing","CA",80000,25,18000),
    ("Kumar","Marketing","NY",91000,50,21000)
  ]

schema = ["employee_name","department","state","salary","age","bonus"]
df = spark.createDataFrame(data=simpleData, schema = schema)

df.show(truncate=False)


# Output
+-------------+----------+-----+------+---+-----+
|employee_name|department|state|salary|age|bonus|
+-------------+----------+-----+------+---+-----+
|        James|     Sales|   NY| 90000| 34|10000|
|      Michael|     Sales|   NY| 86000| 56|20000|
|       Robert|     Sales|   CA| 81000| 30|23000|
|        Maria|   Finance|   CA| 90000| 24|23000|
|        Raman|   Finance|   CA| 99000| 40|24000|
|        Scott|   Finance|   NY| 83000| 36|19000|
|          Jen|   Finance|   NY| 79000| 53|15000|
|         Jeff| Marketing|   CA| 80000| 25|18000|
|        Kumar| Marketing|   NY| 91000| 50|21000|
+-------------+----------+-----+------+---+-----+
```

Grouping and then applying the `avg()` function to the resulting groups.

```python
df.groupBy("department","state").sum("salary","bonus").show()

# Output
+----------+-----+-----------+----------+
|department|state|sum(salary)|sum(bonus)|
+----------+-----+-----------+----------+
|Finance   |NY   |162000     |34000     |
|Marketing |NY   |91000      |21000     |
|Sales     |CA   |81000      |23000     |
|Marketing |CA   |80000      |18000     |
|Finance   |CA   |189000     |47000     |
|Sales     |NY   |176000     |30000     |
+----------+-----+-----------+----------+
```

### Joins

The PySpark Join is used to combine two DataFrames from the values of two of their columns. 
It supports all basic join type operations available in traditional SQL:

```python
empDF.show()

# Output
+------+--------+---------------+-----------+-----------+------+------+
|emp_id|name    |superior_emp_id|year_joined|emp_dept_id|gender|salary|
+------+--------+---------------+-----------+-----------+------+------+
|1     |Smith   |-1             |2018       |10         |M     |3000  |
|2     |Rose    |1              |2010       |20         |M     |4000  |
|3     |Williams|1              |2010       |10         |M     |1000  |
|4     |Jones   |2              |2005       |10         |F     |2000  |
|5     |Brown   |2              |2010       |40         |      |-1    |
|6     |Brown   |2              |2010       |50         |      |-1    |
+------+--------+---------------+-----------+-----------+------+------+

deptDF.show()

# Output
+---------+-------+
|dept_name|dept_id|
+---------+-------+
|Finance  |10     |
|Marketing|20     |
|Sales    |30     |
|IT       |40     |
+---------+-------+


# Inner join
empDF.join(deptDF,empDF.emp_dept_id ==  deptDF.dept_id,"inner").show()

# Output
# Output
+------+--------+---------------+-----------+-----------+------+------+---------+-------+
|emp_id|name    |superior_emp_id|year_joined|emp_dept_id|gender|salary|dept_name|dept_id|
+------+--------+---------------+-----------+-----------+------+------+---------+-------+
|1     |Smith   |-1             |2018       |10         |M     |3000  |Finance  |10     |
|2     |Rose    |1              |2010       |20         |M     |4000  |Marketing|20     |
|3     |Williams|1              |2010       |10         |M     |1000  |Finance  |10     |
|4     |Jones   |2              |2005       |10         |F     |2000  |Finance  |10     |
|5     |Brown   |2              |2010       |40         |      |-1    |IT       |40     |
+------+--------+---------------+-----------+-----------+------+------+---------+-------+
```

In this last example we used `inner`, but we can also use:

* `inner`
* `left`
* `outer`


## Working with SQL

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
