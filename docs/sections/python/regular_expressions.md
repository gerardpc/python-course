# Regular expressions

## Introduction

Regular expressions are a powerful tool for matching patterns in text. They are used in many contexts, including 
search engines, text editors, and programming languages. For example, if we wanted to find all the lines in a
file that began with the word "From:", we could use the following regular expression:

```python
import re


pattern = '^From:'
with open('file.txt') as f:
    for line in f:
        if re.search(pattern, line):
            print(line)
```

In this example, the `^` character matches the beginning of the line. This means that the pattern will only match
lines that begin with the word "From:". 

Usage of such special characters is the main advantage of regular expressions. They allow you to create strings
that match specific patterns of characters, such as digits, words, or whitespace. There are several different
types of special characters, but they can be learned relatively quickly. For example, the following code
matches any line that contains a word that begins with "F" and ends with "m":

```python
import re

string_var = 'Farmland'
pattern = 'F.*m'
if re.search(pattern, string_var):
    print("Found a match!")

# Output:
"Found a match!"
```

The use of these characters can be confusing at first, but once you've learned the basics, you'll be able to
to use them to perform powerful text manipulation with just a few lines of code. This section will teach you 
the basics of regular expressions in Python, and how to use them to search text.

!!!note
    To check if a string matches a regular expression without using Python, 
    you can use [this website](https://regex101.com/)

## Python re module

Python has a built-in module called `re`, which can be used to work with regular expressions. The `re` module
provides several functions that make it a powerful tool for working with regular expressions.

!!!note
    The `re` module is part of the Python standard library, which means that it is installed by default when
    you install Python. This means that you don't need to install any additional packages to use regular
    expressions in Python. Doing `import re` is enough to use the `re` module.

The most important functions in the `re` module are:

* `re.search()`: try to find a match with the pattern anywhere in the string.
* `re.findall()`: Return a list of all matches in the text.
* `re.sub()`: Replaces one or more matches with a string.
* `re.split()`: Splits the text into a list, splitting it wherever the pattern matches.

!!!note
    The `re` module also contains the `re.match()` function, which is similar to `re.search()`, but it only
    matches the pattern if it occurs at the beginning of the string. However, using `re.search()` is usually
    preferred, because it is more flexible and can be used to match patterns anywhere in the string.

!!!note
    Perhaps you come across the `re.compile()` function, which compiles a string into a regular expression object.
    However, this function is rarely used in practice, because the `re` module automatically compiles the string
    into a regular expression object when you use it (so typically the gains in performance are minimal).

### Regular expression syntax

Regular expressions are used to match patterns in text. They are made up of a combination of regular characters
and special characters. Regular characters are characters that match themselves, such as the letter "a" or the
digit "1". Special characters are characters that have a special meaning in regular expressions. For example,
the special character `.` matches _any character_, while the special character `*` matches zero or more occurrences
of the previous character (or group of characters, if they are grouped for example with parentheses).

!!!note
    Regular expressions are case-sensitive, which means that uppercase and lowercase letters are treated as
    different characters. For example, the regular expression `a` will match the letter "a", but not the letter "A".

You can find a list of all the special characters in the [Python documentation](https://docs.python.org/3/library/re.html#regular-expression-syntax).

The following tables gives a summary of the most important special characters and groups:

| Character  | Description                                                                                                                                                       |
|------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `.`        | Matches any character except a newline.                                                                                                                           |
| `^`        | Matches the start of the string.                                                                                                                                  |
| `$`        | Matches the end of the string.                                                                                                                                    |
| `*`        | Matches zero or more occurrences of the previous character.                                                                                                       |
| `+`        | Matches one or more occurrences of the previous character.                                                                                                        |
| `?`        | Matches zero or one occurrences of the previous character.                                                                                                        |
| `{n}`      | Matches exactly `n` occurrences of the previous character.                                                                                                        |
| `{n,}`     | Matches `n` or more occurrences of the previous character.                                                                                                        |
| `{n,m}`    | Matches between `n` and `m` occurrences of the previous character.                                                                                                |
| `[...]`    | Matches any character (replace the `...`) inside the brackets.                                                                                                    |
| `[^...]`   | Matches any character (replace the `...`) not inside the brackets.                                                                                                |
| `|`                                                                                                                                                                  | Matches either the expression before or after the `|`.                                                         |
| `\`        | Escapes a special character (used to match, for example, the `.`).                                                                                                |
| `\d`       | Matches any digit character.                                                                                                                                      |
| `\D`       | Matches any non-digit character.                                                                                                                                  |
| `\w`       | Matches any alphanumeric character.                                                                                                                               |
| `\W`       | Matches any non-alphanumeric character.                                                                                                                           |
| `\s`       | Matches any whitespace character.                                                                                                                                 |
| `\S`       | Matches any non-whitespace character.                                                                                                                             |
| `\b`       | Matches the empty string, but only at the beginning or end of a word. It is used to match whole words.                                                             |
| `(?:...)`  | Matches the expression inside the parentheses, but does not add it to the match.                                                                                  |
| `(?<=...)` | Matches if the current position in the string is preceded by a match for ... that ends at the current position. This is called a _lookbehind assertion_.          |
| `(?<!...)` | Matches if the current position in the string is not preceded by a match for ... that ends at the current position. This is a _negative lookbehind assertion_.    |
| `(?=...)`  | Matches if ... matches next, but doesn’t consume any of the string. This is called a _lookahead assertion_                                                        |
| `(?!...)`  | Matches if ... doesn’t match next. This is a _negative lookahead assertion_.                                                                                      |

The following table shows some examples of regular expressions and the strings that they match:

| Regular expression | Matches                                            |
|--------------------|----------------------------------------------------|
| `a`                | The letter "a"                                     |
| `abc`              | The string "abc"                                   |
| `a.c`              | The string "abc", "axc", "a1c", etc.               |
| `a.*c`             | The string "ac", "abc", "abbc", "abbbc", etc.      |
| `a+c`              | The string "ac", "aac", "aaac", etc.               |
| `\d{2}-\d{3}`        | A string of the form "12-345"                      |
| `(.*)(\d{8}-[A-Z]{3})(.*)` | e.g. "John Doe 12345678-ABC"                       |
| `a| b`                                                 | The string "a" or the string "b" |
| `\W(a| b)\W`                                              | The string " a ", " b ", etc. |
| `^a`               | The string "a" at the beginning of the string      |
| `a$`               | The string "a" at the end of the string            |
| `^a.*c$`           | The string "ac", "abc", "abbc", "abbbc", etc.      |
| `(?<=a)b`          | The string "b" if it is preceded by the string "a" |
| `(?<!a)b`          | The string "b" if it is not preceded by the string "a" |
| `b(?=a)`           | The string "b" if it is followed by the string "a" |
| `b(?!a)`           | The string "b" if it is not followed by the string "a" |


!!!note
    When defining a regular expression in Python, it is recommended to always use raw strings, which are
    strings that are prefixed with an `r`. This is because regular expressions often contain backslashes,
    which are special characters in Python strings. For example, the string literal `r"\n"` consists of 
    two characters: a backslash and a lowercase `n`, in contrast to the string `"\n"`, which Python would
    interpret as a single newline character.    

### `re` functions
#### The `search` function

The `re.search()` function takes two arguments: a regular expression pattern and a string. It searches the string
for the pattern and returns a match object if it finds a match. If it doesn't find a match, it returns `None`.

For example, the following code searches for the pattern `abc` in the string `abcdef`:

```python
import re


pattern = 'abc'
string = 'abcdef'
match = re.search(pattern, string)
print(match)

# Output:
<re.Match object; span=(0, 3), match='abc'>
```

The `match` object contains information about the match, including the start and end position of the match,
and the string that was matched:

* `match.group()`: returns the string that was matched (in the last example, this would be `'abc'`).
* `match.start()`: returns the start position of the match.
* `match.end()`: returns the end position of the match.
* `match.span()`: returns a tuple containing the start and end positions of the match.
* `match.string`: returns the whole string that was searched (in the last example, this would be `'abcdef'`).
* `match.re`: returns the regular expression object that was used to create the match object.

!!!note
    The search function only returns the first match. If you want to find all matches, you can use the
    `re.findall()` function, which returns a list of all matches.

#### The `findall` function

The `re.findall()` function returns a list of all matches in the string. For example, the following code
searches for all occurrences of the pattern `abc` in the string `abcdefgeabc`:

```python
import re


pattern = 'abc'
string = 'abcdefgeabc'
matches = re.findall(pattern, string)
print(matches)

# Output:
['abc', 'abc']
```

If there are no matches, `findall` will return an empty list.

#### The `sub` function

The `re.sub()` function replaces one or more matches with a string. It takes three arguments: a regular expression
pattern, a replacement string, and a string to search. It returns a new string with the matches replaced.

For example, the following code replaces all occurrences of the pattern `abc` with the string `xyz` in the string
`abcdefgeabc`:

```python
import re


pattern = 'abc'
replacement = 'xyz'
string = 'abcdefgeabc'
new_string = re.sub(pattern, replacement, string)
print(new_string)

# Output:
'xyzdefgexyz'
```

#### The `split` function

The `re.split()` function splits the string into a list, splitting it wherever the pattern matches. It takes
two arguments: a regular expression pattern and a string to split. It returns a list of strings.

For example, the following code splits the string `abc,def,ghi` into a list of strings:

```python
import re


pattern = ','
string = 'abc,def,ghi'
new_string = re.split(pattern, string)
print(new_string)

# Output:
['abc', 'def', 'ghi']
```

This method is similar to the `str.split()` method, but it allows you to split the string using a regular
expression instead of a fixed string.


## String manipulation in Pandas

### Standard string methods

Pandas offers several methods and attributes that allow you to work with strings. These methods
are similar to the string methods and attributes in Python, but they are designed to work with Pandas Series.

These methods can be accessed using the `str` attribute of a Series. For example, the following code
creates a Series containing the strings "John Doe" and "Jane Doe", and then uses the `str.upper()` method
to convert the strings to uppercase:

```python
import pandas as pd


df = pd.DataFrame({'name': ['John Doe', 'Jane Doe']})
df['name'] = df['name'].str.upper()
print(df)

# Output:
        name
0   JOHN DOE
1   JANE DOE
```

The following table gives an overview of the most important string methods in Pandas:

| Method | Description |
|--------|-------------|
| `str.lower()` | Converts all characters to lowercase. |
| `str.upper()` | Converts all characters to uppercase. |
| `str.title()` | Converts the first character of each word to uppercase and the rest to lowercase. |
| `str.capitalize()` | Converts the first character to uppercase and the rest to lowercase. |
| `str.strip()` | Removes leading and trailing whitespace. |
| `str.lstrip()` | Removes leading whitespace. |
| `str.rstrip()` | Removes trailing whitespace. |
| `str.replace()` | Replaces all occurrences of a string with another string. |
| `str.split()` | Splits the string into a list of strings. |
| `str.join()` | Joins the elements of a list into a string. |
| `str.cat()` | Concatenates strings in a Series. |


### Regular expressions in Pandas

Pandas also offers several functions that allow you to use regular expressions to search and replace text in 
a DataFrame. These functions are:

* `str.contains()`: Returns a boolean Series indicating whether each string contains a match of a regular expression.
* `str.findall()`: Returns a Series containing lists of all matches of a regular expression.
* `str.replace()`: Replaces all matches of a regular expression with some other string.

These functions are similar to the described functions in the `re` module, but they are designed to work with Pandas
Series and DataFrames. For example, the following code searches for all rows in the DataFrame `df` where the
column `name` contains the string "John":

```python
import pandas as pd


df = pd.DataFrame({'name': ['John Doe', 'Jane Doe', 'John Smith', 'Jane Smith']})
pattern = 'John'
matches = df['name'].str.contains(pattern)
print(matches)

# Output:
0     True
1    False
2     True
3    False
Name: name, dtype: bool
```

The `contains` and `findall` functions are typically used to create boolean masks, which can be used to filter
the DataFrame. For example, the following code creates a boolean mask that is `True` for all rows where the
column `name` contains the string "John":

```python
import pandas as pd


df = pd.DataFrame({'name': ['John Doe', 'Jane Doe', 'John Smith', 'Jane Smith']})
pattern = 'John'
df_john = df.loc[df['name'].str.contains(pattern), :]
print(df_john)

# Output:
         name
0    John Doe
2  John Smith
```

The `replace` function is typically used to replace text in a DataFrame. For example, the following code
replaces all occurrences of the string "John" with the string "Jane" in the column `name`:

```python
import pandas as pd


df = pd.DataFrame({'name': ['John Doe', 'Jane Doe', 'John Smith', 'Jane Smith']})
pattern = 'John'
replacement = 'Jane'
df['name'] = df['name'].str.replace(pattern, replacement)
print(df)

# Output:
         name
0    Jane Doe
1    Jane Doe
2  Jane Smith
3  Jane Smith
```
    
