# Data modification statements

Insert one row into a table
```sql
INSERT INTO t(column_list)
VALUES(value_list);
```
Insert multiple rows into a table
```sql
INSERT INTO t(column_list)
VALUES (value_list), 
       (value_list), …;
```
Insert rows from t2 into t1
```sql
INSERT INTO t1(column_list)
SELECT column_list
FROM t2;
```

!!!note
    INSERT operations on a table without any indices are fast because the new row can simply be 
    appended to the end of the table. It is an O(1) operation. Conversely, INSERT/UPDATE/DELETE 
    statements with indices are no longer simple. These operations render all indexes out-of-date, 
    and as such need to be reconstructed.

    The situation is reversed with SELECT statements: SELECT operations containing only non-key 
    fields in the WHERE clause on the same table will require a full table scan, an O(n) operation. 
    With indices, however, this operation becomes O(log(n)). 


Update new value in the column c1 for all rows
```sql
UPDATE t
SET c1 = new_value;
```
Update values in the column c1, c2 that match the condition
```sql
UPDATE t
SET c1 = new_value, 
        c2 = new_value
WHERE condition;
```
Delete all data in a table
```sql
DELETE FROM t;
Delete subset of rows in a table
DELETE FROM t
WHERE condition;
```

MISSING
WITH, TIES, USING, partition by, TOP, computed, window function, coalesce a outer join, LIKE, LOCATE, With, exemples https://ploomber.io/blog/sql/ , union, returning