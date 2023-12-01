# Data modification statements

Data modification statements are used to modify data in a database, in contrast to data query statements,
which are used to read data from a database and do not modify the data. 

## INSERT

INSERT statements are used to add new rows to a table. The syntax is as follows, to insert one row into a table:

```sql
INSERT INTO table_name (column_list)
VALUES (value_list);
```
To insert multiple rows into a table, we can use the following syntax:

```sql
INSERT INTO table_name (column_list)
VALUES (value_list_1),
        (value_list_2),
        (value_list_3),
        ...
        (value_list_n);
```

We can also insert rows into a table from another table, using the following syntax:

```sql
INSERT INTO table_name (column_list)
SELECT column_list
FROM table_name;
```

!!!note
    INSERT operations on a table without any indices are fast because the new row can simply be 
    appended to the end of the table. It is an O(1) operation. Conversely, INSERT/UPDATE/DELETE 
    statements with indices are no longer simple. These operations render all indexes out-of-date, 
    and as such need to be reconstructed.

    The situation is reversed with SELECT statements: SELECT operations containing only non-key 
    fields in the WHERE clause on the same table will require a full table scan, an O(n) operation. 
    With indices, however, this operation becomes O(log(n)). 


## UPDATE

UPDATE statements are used to modify existing rows in a table. The syntax is as follows:

```sql
UPDATE table_name
SET column_1 = value_1,
    column_2 = value_2,
    ...
    column_n = value_n
WHERE condition;
```

## DELETE

DELETE statements are used to delete rows from a table. The syntax is as follows:

```sql
DELETE FROM table_name
WHERE condition;
```
