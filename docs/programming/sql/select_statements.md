# SELECT statement

The `SELECT` statement is used to read data from a database. The basic syntax is:
```sql
SELECT 
    c1, c2 
FROM some_table;
```

!!!note
    Although it is common practice to put SQL reserved words in capital letters,
    in general it is not necessary. For example, `SELECT` and `select` are equivalent.

The last query will return columns `c1`, `c2` from the table `some_table`. 
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

We will see more about `JOIN` statements later. 

!!!note
    We can use break lines in SQL queries to make them more readable, but this is not necessary.
    With breaklines or not, the query will be executed in the same way.

We can also use aliases to define new columns:
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
    `IN (value1, value2, ...);`, or equivalently in a derived table with one column, `IN (SELECT ...)`.

!!!note
    To match a string, we can use the `LIKE` operator, which allows us to use wildcards such as `%` and `_`.
    For example, to match all strings that start with `a` and end with `b`, we can use the following condition:
    ```sql
    WHERE c1 LIKE 'a%b';
    ```

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

## GROUP BY and aggregate functions

Like in pandas, we can group unique values of one or more columns and apply an aggregate function to them

```sql
SELECT c1, aggregate(c2) FROM t GROUP BY c1;
```

Aggregate functions are, essentially, functions that take a set of values and return a single value.
We apply the aggregate function to each column that is not grouped by. For example, we can count the
number of rows in each group:
```sql
SELECT creator, count(*) from fragrances group by creator;
```

Inside the `COUNT` parentheses you can put a DISTINCT to count different appearances or all of them.

!!!note 
    The COUNT function returns the number of rows for which the expression evaluates to a non-null value. 
    (* is a special expression that is not evaluated, it simply returns the number of rows.)

Usual aggregate functions are:

* `COUNT` (count the number of rows)
* `SUM` (sum of all values in a column)
* `AVG` (average of all values in a column)
* `MIN` (minimum value in a column)
* `MAX` (maximum value in a column)
* `STDDEV` (standard deviation of all values in a column)

### HAVING

HAVING is used to filter records that work on summarized GROUP BY results. HAVING is typically used with a 
GROUP BY clause. When GROUP BY is not used, HAVING behaves like a WHERE clause.

!!!note
    Difference between HAVING and WHERE: 
    * HAVING is used to check conditions after the aggregation takes place. 
    * WHERE is used to check conditions before the aggregation takes place.
    Hence, usually WHERE is faster than HAVING.

For example, this code:
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

## CASE statements

The CASE statement goes through conditions as and `if/elif` statement, and returns a column:

```sql
SELECT OrderID, Quantity,
CASE 
	WHEN Quantity > 30 THEN 'The quantity is greater than 30'
	WHEN Quantity = 30 THEN 'The quantity is 30'
	ELSE 'The quantity is under 30'
END AS QuantityText
FROM OrderDetails;
```

## Table JOINs

A join _stitches_ together 2 tables, based on some condition (e.g. equality of a row).
We use the `JOIN` when we want to query data from multiple tables at once. The basic syntax is:

```sql
SELECT 
    c1, c2 
FROM 
    t1 
INNER JOIN 
    t2
ON 
    condition;
```

For example, we can join the `Customers` and `Orders` tables to get the name of the customer
who made each order:

```sql
SELECT 
    Orders.OrderID, Customers.CustomerName, Orders.OrderDate
FROM 
    Orders
INNER JOIN
    Customers
ON
    Orders.CustomerID = Customers.CustomerID;
```

!!!note
    The SQL Join is the equivalent of the `merge` function in pandas.

Joins can be of several different types:

* INNER (intersection of values)
* LEFT (Intersection + unpaired from left table)
* RIGHT (same but with right table) 
* FULL OUTER (both).

## Combine rows from two queries

UNION ALL is used to combine the result from multiple SELECT statements into a single result set.

```sql
SELECT c1, c2 FROM t1 
UNION ALL
SELECT c1, c2 FROM t2;
```

We can also find the intersection of two queries, or subtract one from the other. Intersection:

```sql
SELECT c1, c2 FROM t1 
INTERSECT 
SELECT c1, c2 FROM t2;
```
Subtract a result set from another result set:
```sql
SELECT c1, c2 FROM t1 
MINUS 
SELECT c1, c2 FROM t2;
```

## Subqueries

A subquery is a SELECT query nested inside another query. We can use subqueries to query data from
multiple tables, or to query data from the same table using different conditions. For example,
we can use a subquery to find the name of the customer who made the most recent order:

```sql
SELECT 
    CustomerName
FROM
    Customers
WHERE
    CustomerID = (
        SELECT 
            CustomerID 
        FROM 
            Orders 
        ORDER BY 
            OrderDate DESC 
        LIMIT 1
    );
```

## Common Table Expressions (CTEs)

A Common Table Expression (CTE) is a temporary result set that we can reference within another query.
It is similar to a subquery, but it is more readable (because it has a name), and hence it is easier to 
maintain. 

CTEs always start with the `WITH` keyword, followed by the name of the CTE, and the query that defines it.
For example, we can use a CTE to find the name of the customer who made the most
recent order:

```sql
WITH last_order AS (
    SELECT 
        CustomerID 
    FROM 
        Orders 
    ORDER BY 
        OrderDate DESC 
    LIMIT 1
)

SELECT 
    CustomerName
FROM
    Customers
WHERE
    CustomerID = last_order.CustomerID;
```

We can define multiple CTEs in the same query, and we can also use CTEs to query data from multiple tables:
    
```sql
WITH 
    last_order AS (
        SELECT 
            CustomerID 
        FROM 
            Orders 
        ORDER BY 
            OrderDate DESC 
        LIMIT 1
    ),
    customer AS (
        SELECT 
            CustomerName
        FROM
            Customers
        WHERE
            CustomerID = last_order.CustomerID
    )

SELECT
    customer.CustomerName,
    Orders.OrderDate
FROM
    Orders
INNER JOIN
    customer
ON
    Orders.CustomerID = customer.CustomerID;
```
