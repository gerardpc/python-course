# Table operations

Beyond the basic CRUD (create, read, update, delete) operations, there are a number of other operations 
that can be performed on tables.

## Creating a table

To create a new table in a database, we use the `CREATE TABLE` statement. The syntax is as follows:

```sql
CREATE TABLE t (
     id INT PRIMARY KEY,
     name VARCHAR NOT NULL,
     price INT DEFAULT 0
);
```
As seen in the example, we need to specify the name of the table, and a list of columns, each with a 
name and a data type. We can also specify constraints on the columns, the most important being the
primary key. The primary key is a column (or a combination of columns) that uniquely identifies each row.
Hence, it cannot contain NULL values, and it cannot contain duplicate values.

In the example, we have specified that the `id` column is the primary key. We have also specified that
the `name` column cannot contain NULL values, and that the `price` column has a default value of 0.

## Deleting a table

Similarly, to delete a table from a database, we use the `DROP TABLE` statement. The syntax is as follows:

```sql
DROP TABLE t;
```

## Altering a table

There are a number of operations that can be performed on a table after it has been created. These
operations are performed using the `ALTER TABLE` statement. 

* Add a new column to the table:
    ```sql
    ALTER TABLE t ADD column;
    ```
* Drop column `c` from the table
    ```sql
    ALTER TABLE t DROP COLUMN c;
    ```

* Rename a table from `t1` to `t2`
    ```sql
    ALTER TABLE t1 RENAME TO t2;
    ``` 

* Rename column c1 to c2
    ```sql
    ALTER TABLE t1 RENAME c1 TO c2 ;
    ```
  
* Remove all data from a table (but keep the table structure):
    ```sql
    TRUNCATE TABLE t;
    ```

### Table constraints

Table constraints are rules that are enforced on columns of a table. These are used to limit the
type of data that can go into the table. The following constraints are commonly used in SQL:

* `NOT NULL` - Ensures that a column cannot have a `NULL` value
* `UNIQUE` - Ensures that all values in a column are different
* `PRIMARY KEY` - A combination of `NOT NULL` plus `UNIQUE`. Uniquely identifies each row in a table.
    !!!note
        Each table can have only **one** primary key, and although it is not mandatory, it is a very good
        practice to have a primary key in every table. Columns that can potentially be primary keys are known 
        as **candidate keys**; we can choose any candidate key to be the primary key. 
        
    !!!note
        Primary keys can also be "composite", i.e., a combination of columns that uniquely identify each row.

* `FOREIGN KEY` - Foreign keys are used to link two tables together. A foreign key in one table points 
    to a primary key in another table. 
    !!!note
        For example, if we have a table called `Orders` that contains information about orders made by 
        customers, we can create a foreign key in the `Orders` table that points to the `CustomerID` 
        column in the `Customers` table. This way, we can easily find the customer who made each order. 

* `DEFAULT` - Sets a default value for a column if no value is specified
* `CREATE INDEX` - Used to create and retrieve data from the database very quickly

!!!note
    Table indexes are used to speed up the retrieval of data from a table. Usually, the data in a 
    table is stored in an unordered manner. When we create an index, the database stores the data in
    a sorted manner, which makes it much faster to retrieve data from the table. However, this comes
    at the cost of slower insertions, updates and deletions, since the database has to maintain the
    sorted order of the data. Hence, we should only create indexes on columns that we frequently use
    to retrieve data from the table.

To add a constraint on a table, we also use the `ALTER TABLE` statement. The syntax is as follows:
```sql
ALTER TABLE t ADD constraint;
```

### Table constraints example queries

* To add a primary key constraint on the `id` column of the `t` table, we can use the following
statement:
    ```sql
    ALTER TABLE t ADD PRIMARY KEY (id);
    ```
* To add a foreign key constraint on the `customer_id` column of the `orders` table, we can use the
following statement:
    ```sql
    ALTER TABLE orders ADD FOREIGN KEY (customer_id) REFERENCES customers(id);
    ```
* To add a default value of 0 to the `price` column of the `t` table, we can use the following statement:
    ```sql
    ALTER TABLE t ALTER COLUMN price SET DEFAULT 0;
    ```
* To add a `NOT NULL` constraint on the `name` column of the `t` table, we can use the following statement:
    ```sql
    ALTER TABLE t ALTER COLUMN name SET NOT NULL;
    ```
* To add a `UNIQUE` constraint on the `name` column of the `t` table, we can use the following statement:
    ```sql
    ALTER TABLE t ADD UNIQUE (name);
    ```
* To add an index on the `name` column of the `t` table, we can use the following statement:
    ```sql
    CREATE INDEX name_index ON t (name);
    ```

To drop a constraint we use the following syntax:
```sql
ALTER TABLE t DROP constraint;
```
For example, to drop the `NOT NULL` constraint on the `name` column of the `t` table, we can use the following statement:
```sql
ALTER TABLE t ALTER COLUMN name DROP NOT NULL;
```
