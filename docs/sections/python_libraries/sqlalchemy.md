# Connecting SQL to Python

To connect to a SQL database from Python, we need to install a package that allows us to do so. 
One of the most popular packages for this is `sqlalchemy`. We can install it with `pip`:

```bash
pip install sqlalchemy
```

## Connecting to SQL Databases with SQLAlchemy

To create a connection to a SQL database, we need to create an engine, an object that 
manages connections to the database. To create an engine, we need to provide a database URI, which
is a string that tells SQLAlchemy how to connect to the database. The format of the URI depends on
the type of database we are connecting to. For example, to connect to a SQLite database from a file
called `database.db` that is on the same folder as the Python script, we can use the following URI:

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

DB_URI = "sqlite:///database.db"

engine = create_engine(DB_URI, pool_pre_ping=True)
```

Once we have an engine, we can create a session, which is
an object that manages transactions (i.e. reading and writing) to the database.

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

DB_URI = "sqlite:///database.db"

engine = create_engine(DB_URI, pool_pre_ping=True)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)
```

!!!note
    The `pool_pre_ping` argument is used to check if the connection to the database is still alive
    before using it. This is useful when using a database in a remote server, as the connection
    might be lost if the server is not used for a while.

`SessionLocal` is a class that we can use to create session objects. We can create a session object
by calling the class thus defined:

```python
with SessionLocal() as session:
    # Do something with the session
    ...
```

!!!note
    The session creation code is a bit obscure, but it is always the same, so it can be copied
    into a separate file. For example, we can copy the code above into a file 
    called `database_session.py` (for example). Then, we can import the `SessionLocal` class 
    from that file as follows:

    ```python
    from database_session import SessionLocal
    ```    

## Reading/inserting into a SQL database with Pandas

Now that we have a session object, we can use it to read and write to the database.

### Reading from a SQL database

To read data from a SQL database and load it into a pandas DataFrame, we need to provide a SQL query and 
a session object. For example, to read all the rows from a table called `users`, we can use the following code:

```python
from sqlalchemy import text
import pandas as pd
from database_session import SessionLocal


query = text("SELECT * FROM users")

with SessionLocal() as session:
    df = pd.DataFrame(session.execute(query))
```

This code will return a pandas dataframe with all the rows from the `users` table. 

!!!note
    The `text` function is used to convert a string into a SQLAlchemy `TextClause` object, which
    is the type of object that `session.execute` expects as the first argument.


### Inserting into a SQL database

Pandas' `to_sql` is a function that allows us to insert data from a pandas dataframe
into a SQL database with a single line of code, without having to write any SQL INSERT queries.
To use `to_sql`, we need to provide a pandas dataframe and a table name. For example, to insert
the rows from a dataframe called `df` into a table called `users`, we can use the following code:

```python
from sqlalchemy import text
import pandas as pd
from database_session import SessionLocal


with SessionLocal() as session:
    df.to_sql("users", session.get_bind(), if_exists="append", index=False)
```

!!!note
    The `if_exists` argument is used to specify what to do if the table already exists. In this
    case, we are telling pandas to append the rows from the dataframe to the table. If we wanted
    to replace the table, we could use `if_exists="replace"` instead.

    The `index` argument is used to specify whether to include the index of the dataframe as a 
    column or not (usually we will not want to do that).

### Working snippet

After installing sqlalchemy and saving a SQLite database called `hr` in the same folder as the
Python script, we can use the following code to read all the rows from the `employees` table:

```python
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import pandas as pd

DB_URI = "sqlite:///hr"

engine = create_engine(DB_URI, pool_pre_ping=True)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

query = text("SELECT * FROM employees")

with SessionLocal() as session:
    df = pd.DataFrame(session.execute(query))

print(df.head())
```

!!!note
    The previous code should run "as is" if the `hr` database is in the same folder as the Python.

```python
# Output:
   employee_id first_name last_name  ... manager_id department_id Avg_Salary
0          100     Steven      King  ...          0            90       None
1          101      Neena   Kochhar  ...        100            90       None
2          102        Lex   De Haan  ...        100            90       None
3          103  Alexander    Hunold  ...        102            60       None
4          104      Bruce     Ernst  ...        103            60       None
```

### Executing raw SQL queries

Not all SQL queries come from or should be inserted into a pandas DataFrame. With SQLAlchemy we can also execute 
raw SQL queries (i.e., any valid SQL code we want) with the line `session.execute(query)`. For
example, if we wanted to delete a table from the DB we would do:

```python
from sqlalchemy import text
from database_session import SessionLocal


query = text("drop table some_table")

with SessionLocal() as session:
    session.execute(query)
```