# SQL

SQL (Structured Query Language) is a programming language used to communicate with data stored in a 
relational database management system. SQL syntax is similar to the English language, which makes it 
relatively easy to write, read, and interpret.

SQL is used to perform tasks such as:

* Create new databases
* Create new tables in a database
* Insert records in a database
* Update records in a database
* Delete records from a database
* Retrieve data from a database

In this section we will cover the basics of SQL, and how to connect to a SQL database from Python.

## SQL DataBase engines 

There are many different SQL database engines, but the most common are:

* SQLite
* MySQL
* PostgreSQL
* Microsoft SQL Server
* Oracle SQL

They all share a common SQL syntax, but each has its own unique set of features and capabilities. Once
we get familiar with SQL, it is relatively easy to switch between different database engines.

## Connecting to a SQL database

In order to connect to an already existing SQL database, we have several options:

* Use a GUI (Graphical User Interface) such as [DBeaver](https://dbeaver.io/).
* Connect to the DB from PyCharm (see [here](https://www.jetbrains.com/help/pycharm/connecting-to-a-database.html)). 
* Use a general Python package such as [SQLAlchemy](https://www.sqlalchemy.org/), that works
  with many different SQL engines.
* Use a specific Python package for the SQL engine that we are using. For example, 
  [mysql.connector](https://dev.mysql.com/doc/connector-python/en/connector-python-example-connecting.html) 
  for MySQL, or [psycopg2](https://www.psycopg.org/docs/usage.html) for PostgreSQL.

## SQL language

SQL is a declarative language, which means that we tell the database what we want to do, and the database
engine figures out how to do it. This is different from imperative languages such as Python, where we
tell the computer exactly what to do, step by step.

SQL is built over queries, which are statements that we send to the database engine. The most common
query is the `SELECT` statement, which is used to read data from a database. Other common queries are
`INSERT`, `UPDATE` and `DELETE`, which are used to modify data in a database.