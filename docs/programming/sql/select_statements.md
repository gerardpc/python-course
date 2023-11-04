# SELECT statement

!!!note
    Although it is common practice to put SQL reserved words in capital letters,
    in general it is not necessary. For example, `SELECT` and `select` are equivalent.

The `SELECT` statement is used to read data from a database. The basic syntax is:
```sql
SELECT 
    c1, c2 
FROM some_table;
```
This query will return columns `c1`, `c2` from the table `some_table`. 
We can also use the `*` to query all rows from a table
```sql
SELECT * FROM some_table;
```

!!!note
    The select statement from SQL is similar to panda's `.loc` function. 
    For example, to write the previous query in pandas we would do:
    ```python
    df.loc[:, ['c1', 'c2']]
    ```

## Aliases

Sometimes we want to rename the columns that we are querying. We can do this using aliases for tables and columns:

```sql
SELECT 
    CustomerID AS ID, 
    CustomerName AS Customer
FROM some_table AS t;
```

This is particularly useful when we are accessing data from multiple tables, and we want to avoid
ambiguities. For example, if we have two tables with a column called `id`, we can use aliases to
distinguish between them:

```sql
SELECT 
    t1.id AS id1, 
    t2.id AS id2
FROM table1 AS t1
INNER JOIN table2 AS t2 ON t1.id = t2.id;
```

We will see more about `JOIN` statements later. We can also use aliases to define new columns:
```sql
SELECT 
    CustomerName, 
    Address + ', ' + PostalCode + ' ' + City + ', ' + Country AS Address 
FROM Customers;
```

!!!note
    The table we query from does not need to be an existing table: it can be any form of 
    derived table, created by another `SELECT ...` statement. Typical ways to create this
    _new_ tables to query from are Common Table Expressions (CTEs) and subqueries (just another
    `SELECT` statement between parentheses). We will see more about this later.


## WHERE clauses

So far we have seen how to query all rows from a table. However, in most cases we want to query
only a subset of the rows. We can do this using the `WHERE` clause:

```sql
SELECT c1, c2 FROM table_1 as t WHERE some_condition;
```

In the last example, `some_condition` is a boolean expression that evaluates to `True` or `False`.

!!!note
    `some_condition` can be of type `=` (e.g. `A = 3`), different `!=`, `>`, `<`, etc. 
    Conditions can be chained by `AND` and `OR` operators. They can also be of the type `in a set`, e.g.:
    `IN (value1, value2, ...);`, or equivalently in a derived table, `IN (SELECT ...)`.

## Other operators 

### DISTINCT

Distinct rows from a table can be queried using the `DISTINCT` keyword:

```sql
SELECT DISTINCT c1, c2 FROM t WHERE some_condition;
```

### ORDER BY

We can also sort the result set in ascending or descending order using the `ORDER BY` clause:

```sql
SELECT c1, c2 FROM t ORDER BY c1 ASC [or DESC];
```

### LIMIT and OFFSET

To limit the number of rows returned by a query, we can use the `LIMIT` clause:

```sql
SELECT c1, c2 FROM t LIMIT n;
```
To skip offset of rows and return the next n rows, we can use the `OFFSET` clause:

```sql
SELECT c1, c2 FROM t ORDER BY c1 LIMIT n OFFSET offset;
```

## GROUP BY and AGGREGATES

Group rows using an aggregate function
```sql
SELECT c1, aggregate(c2) FROM t GROUP BY c1;
```

Aggregate functions are, essentially, COUNT. You can’t use COUNT without telling by which 
GROUP BY the rows should be aggregated (basically every other column that is not counted). E.g.
```sql
SELECT etl_origin_id, creator, count(*) from etl_fragrances group by etl_origin_id, creator;
```

Inside a COUNT parentheses you can put a DISTINCT/ALL to count different appearances or all of them.

!!!note 
    The COUNT function returns the number of rows for which the expression evaluates to a non-null value. 
    (* is a special expression that is not evaluated, it simply returns the number of rows.)

There are two additional modifiers for the expression: ALL and DISTINCT. 
These determine whether duplicates are discarded. Since ALL is the default, your example is the same 
as count(ALL 1), which means that duplicates are retained. Since the expression "1" evaluates to non-null 
for every row, and since you are not removing duplicates, COUNT(1) should always return the same number as COUNT(*).

!!!note
    Difference between HAVING and WHERE: 
    * HAVING is used to check conditions after the aggregation takes place. 
    * WHERE: is used to check conditions before the aggregation takes place.

This code:
```sql
select City, COUNT(*)
From Address
Where State = 'MA'
Group By City
```
Gives you a table of all cities in MA and the number of addresses in each city. This code:
```sql
select City, COUNT(*)
From Address
Where State = 'MA'
Group By City
Having COUNT(*)>5
```

Gives you a table of cities in MA with more than 5 addresses and the number of addresses in each city.

The CASE statement goes through conditions as and `if/elif` statement, and returns a column. E.g.
```sql
SELECT OrderID, Quantity,
CASE 
	WHEN Quantity > 30 THEN 'The quantity is greater than 30'
	WHEN Quantity = 30 THEN 'The quantity is 30'
	ELSE 'The quantity is under 30'
END AS QuantityText
FROM OrderDetails;
```

TABLE JOINS


## Table joins

A join “sews” together 2 tables, based on some condition (e.g. equality of a row):
```sql
SELECT c1, c2 FROM t1 INNER JOIN t2 ON condition;
```
 
Joins can be of type:

* INNER (intersection of values)
* LEFT (Intersection + unpaired from left table)
* RIGHT (same with right table) or 
* FULL OUTER (both).

Combine rows from two queries
```sql
SELECT c1, c2 FROM t1 UNION [ALL] SELECT c1, c2 FROM t2;
```
Return the intersection of two queries
```sql
SELECT c1, c2 FROM t1 INTERSECT SELECT c1, c2 FROM t2;
```
Subtract a result set from another result set
```sql
SELECT c1, c2 FROM t1 MINUS SELECT c1, c2 FROM t2;
```

MANIPULATING TABLES

Create a new table:
```sql
CREATE TABLE t (
     id INT PRIMARY KEY,
     name VARCHAR NOT NULL,
     price INT DEFAULT 0
);
```
Delete the table from the database
```sql
DROP TABLE t;
```
Add a new column to the table
```sql
ALTER TABLE t ADD column;
```
Drop column c from the table
```sql
ALTER TABLE t DROP COLUMN c;
```