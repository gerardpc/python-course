#### Table constraints

Constraints: rules for the data in a table. The following constraints are commonly used in SQL:
* NOT NULL - Ensures that a column cannot have a NULL value
* UNIQUE - Ensures that all values in a column are different
* PRIMARY KEY - A combination of a NOT NULL and UNIQUE. Uniquely identifies each row in a table
* FOREIGN KEY - Prevents actions that would destroy links between tables
* CHECK - Ensures that the values in a column satisfies a specific condition
* DEFAULT - Sets a default value for a column if no value is specified
* CREATE INDEX - Used to create and retrieve data from the database very quickly

Add a constraint
```sql
ALTER TABLE t ADD constraint;
```
Drop a constraint
```sql
ALTER TABLE t DROP constraint;
```
Rename a table from t1 to t2
```sql
ALTER TABLE t1 RENAME TO t2;
```
Rename column c1 to c2
```sql
ALTER TABLE t1 RENAME c1 TO c2 ;
```
Remove all data in a table
```sql
TRUNCATE TABLE t;
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


### Table management