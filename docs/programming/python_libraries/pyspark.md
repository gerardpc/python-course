# PySpark

## Introduction

**Spark** is a platform for cluster computing. Spark lets you spread data and computations over clusters 
with multiple nodes (think of each node as a separate computer). Splitting up your data makes it easier 
to work with very large datasets because each node only works with a small amount of data. This means
that Spark is a common tool for working with big data (for example, for cleaning it up or transforming
it into a more usable format).

As each node works on its own subset of the total data, it also carries out a part of the total 
calculations required, so that both data processing and computation are performed in parallel over 
the nodes in the cluster. 

**PySpark** is the Python API for Spark. It is a Python library that allows you to interact with Spark
using Python. 

## Pandas vs Spark

Spark DataFrames are conceptually equivalent to a Pandas DataFrame. The main difference is that
Spark DataFrames are immutable, meaning that they cannot be changed after they are created. This
allows Spark to do more optimization under the hood. 

However, setting up Spark is considerably more complicated than using Pandas, and it requires additional software. 
Hence, we should only use Spark when our data is too big to work with on a single machine. If that is not the case,
we should use Pandas instead.

## Setting up accounts/computers 

## Resilient Distributed Dataset (RDDs)

The fundamentals of using  and mapping
Filtering and sorting

## RDD Actions and key/value datastores.