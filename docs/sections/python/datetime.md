# Dates and times

Python provides a number of modules for working with dates and times, but the most
commonly used is the `datetime` module, which is part of the standard library.

The `datetime` module provides three classes for working with dates and times:

* `datetime.date` - for working with dates in isolation (which look like `YYYY-MM-DD`)
* `datetime.time` - for working with times in isolation (which look like `HH:MM:SS`)
* `datetime.datetime` - for working with dates and times together (which look like `YYYY-MM-DD HH:MM:SS`)

The `datetime` module also provides a `datetime.timedelta` class for representing
durations of time.

## Importing the datetime module

We can import the `datetime` module with the following statement:

```python
import datetime
```

The datetime module name and the class names are the same (that's unfortunate), so we can't import the
classes directly. We can import the classes we need into our namespace using the `from ... import ...` syntax:

```python
from datetime import date, time, datetime, timedelta
```

## Creating date and time objects

We can create `date`, `time` and `datetime` objects using the `date()`, `time()` and `datetime()` constructors
respectively. Each of these constructors takes a number of arguments, which are used to initialise the object.

### Creating date objects

The `date()` constructor takes three arguments:

* `year` - the year as an integer
* `month` - the month as an integer (1-12)
* `day` - the day as an integer (1-31)

```python
d = date(2018, 1, 1)
print(d)

# Output: 2018-01-01
```

### Creating time objects

The `time()` constructor takes four arguments:

* `hour` - the hour as an integer (0-23)
* `minute` - the minute as an integer (0-59)
* `second` - the second as an integer (0-59)
* `microsecond` - the microsecond as an integer (0-999999)

```python
t = time(12, 30, 0, 0)
print(t)

# Output: 12:30:00
```

### Creating datetime objects

The `datetime()` constructor takes seven arguments:

* `year` - the year as an integer
* `month` - the month as an integer (1-12)
* `day` - the day as an integer (1-31)
* `hour` - the hour as an integer (0-23)
* `minute` - the minute as an integer (0-59)
* `second` - the second as an integer (0-59)
* `microsecond` - the microsecond as an integer (0-999999)

!!!note
    Only the `year`, `month` and `day` arguments are required. The other arguments default to `0`.

```python
dt = datetime(2018, 1, 1, 12, 30, 0, 0)
print(dt)

# Output: 2018-01-01 12:30:00
```

The `datetime()` constructor also takes a `tzinfo` argument, which is used to specify the time zone
of the `datetime` object. This parameter is also optional, but if you don't specify a `tzinfo`
argument (known as naive time), then the `datetime` object will be created in the local time zone. This
is usually a bad idea that can lead to many problems

We will discuss time zones in more detail in the time zones section.

## Working with date and time objects

Once we have created a `date`, `time` or `datetime` object, we can access the individual components
of the object using the following attributes:

* `year` - the year as an integer
* `month` - the month as an integer (1-12)
* `day` - the day as an integer (1-31)
* `hour` - the hour as an integer (0-23)
* `minute` - the minute as an integer (0-59)
* `second` - the second as an integer (0-59)
* `microsecond` - the microsecond as an integer (0-999999)
* `tzinfo` - the time zone as a `tzinfo` object

```python
from datetime import date

d = date(2018, 1, 1)
print(d.year)
print(d.month)
print(d.day)

# Output:
# 2018
# 1
# 1
```

We can also use the `strftime()` method to format a `date`, `time` or `datetime` object as a string:

```python
from datetime import date

d = date(2018, 1, 1)
print(d.strftime('%Y-%m-%d'))

# Output: 2018-01-01
```

## Parsing date and time strings

To do the opposite, and parse a string into a `date`, `time` or `datetime` object, we can use the `strptime()` function.
The `strptime()` function takes two arguments:

* `date_string` - the string to parse
* `format` - the format of the string to parse

The `format` argument is a string that specifies the format of the string to parse. The format string
uses the same directives as the `strftime()` method.

```python
from datetime import datetime

str_date = '2018-01-01 12:30:00'
dt = datetime.strptime(str_date, '%Y-%m-%d %H:%M:%S')

print(dt)

# Output: 2018-01-01 12:30:00
```

## Difference between two dates or times

To calculate the difference between two `date`, `time` or `datetime` objects, we can use the `-` operator.
The result of this operation is a `timedelta` object, which represents the time difference between the two objects.

```python
from datetime import date

d1 = date(2018, 1, 1)
d2 = date(2018, 1, 2)

dt = d2 - d1
print(dt)

# Output: 1 day, 0:00:00
```

Timedeltas can be added to or subtracted from `date`, `time` or `datetime` objects using the `+` and `-` operators.

```python
from datetime import date, timedelta

d = date(2018, 1, 1)
dt = d + timedelta(days=1)

print(dt)

# Output: 2018-01-02
```

!!!note
    However, we can't sum two different `date`, `time` or `datetime` objects together. The sum only works
    when adding a `timedelta` to a `date`, `time` or `datetime` object.


## Working with time zones

The `datetime` module provides a `tzinfo` class for representing time zones. However, this class
is an abstract base class (i.e., its methods are empty), and to use it we would need to implement
it. The `pytz` module (a module that needs to be installed, since it doesn't come with Python) 
provides a concrete implementation of the `tzinfo` class, which we can use to represent time zones. 
For example, to define a `datetime` object in the UTC time zone and convert it to european summer time, 
we can do the following:

```python
from datetime import datetime
import pytz


dt_1 = datetime(2018, 1, 1, 12, 30, 0, 0, tzinfo=pytz.utc)
dt_2 = dt_1.astimezone(pytz.timezone('Europe/Paris'))

print(dt_1)
print(dt_2)

# Output:
# 2018-01-01 12:30:00+00:00
# 2018-01-01 13:30:00+01:00
```

In this example, the `astimezone()` method converts a `datetime` object from one time zone to another.


## Working with dates and times in pandas

The pandas library provides a number of data structures for working with dates and times. The most
commonly used are the `Timestamp` and `DatetimeIndex` classes.

### The Timestamp class

The `Timestamp` class represents a single date and time, and is very similar to the `datetime` class.
To create a `Timestamp` object, we can use the `to_datetime()` function, which takes a string or
a number of arguments, which are used to initialise the object.

```python
import pandas as pd

df = pd.DataFrame({'date': ['2018-01-01 12:30:00']})
df['date'] = pd.to_datetime(df['date'])

print(df['date'].dtype)

# Output: datetime64[ns]
```

This class provides a number of methods for working with dates and times. For example, we can use the
`strftime()` method to format a `Timestamp` object as a string (like the `datetime` class), or we can
use the `year`, `month`, `day`, `hour`, `minute`, `second` and `microsecond` attributes to access the
individual components of the object.

```python
import pandas as pd

df = pd.DataFrame({'date': ['2018-01-01 12:30:00']})
df['date'] = pd.to_datetime(df['date'])

print(df['date'].dt.weekday_name)

# Output: 0    Monday
# Name: date, dtype: object
```

### The DatetimeIndex class

When we have a date or datetime column in a pandas DataFrame, we can set is as the index of the DataFrame
using the `set_index()` method. This will create a `DatetimeIndex` object, which is used to index the
DataFrame.

!!!note
    Using the dates as index is only possible if the dates are unique, and is only useful if we want to
    select rows by date. Otherwise, we can just leave the dates as a normal column in the DataFrame.

```python
import pandas as pd


df = pd.DataFrame({'date': ['2018-01-01 12:30:00', '2018-01-02 12:30:00', '2018-01-03 12:30:00']})
df.set_index('date', inplace=True)

print(df.index)

# Output
# DatetimeIndex(['2018-01-01 12:30:00', '2018-01-02 12:30:00',
#                '2018-01-03 12:30:00'],
#               dtype='datetime64[ns]', name='date', freq=None)
```

!!!note
    Setting the index of a DataFrame to a `DatetimeIndex` object is a very common operation, so pandas
    provides a `parse_dates` argument for the `read_csv()` function, which can be used to automatically
    parse date and time columns into a `DatetimeIndex` object.

We can also create a `DatetimeIndex` with the `pd.date_range()` function. This function takes a number
of arguments, which are used to create a range of dates. For example, to create a `DatetimeIndex` with
the dates from 2018-01-01 to 2018-01-10, we can do the following:

```python
import pandas as pd


index = pd.date_range('2018-01-01', '2018-01-20', freq='W')
print(index)

# Output:
# DatetimeIndex(['2018-01-07', '2018-01-14'], dtype='datetime64[ns]', freq='W-SUN')
```

!!!note
    The `freq` argument specifies the frequency of the dates. In this example, we have used `W` to
    specify weekly frequency, and `W-SUN` to specify weekly frequency with the week ending on Sunday.




