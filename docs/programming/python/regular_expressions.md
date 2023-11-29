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
    you can use this website: https://regex101.com/

## Python re module

Python has a built-in module called `re`, which can be used to work with regular expressions. The `re` module
provides several functions that make it a powerful tool for working with regular expressions.

The most important functions in the `re` module are:

* `re.search()`: find a match with the pattern anywhere in the string and return a match object.
If nothing is found, it returns `None`.
* `re.findall()`: Returns a list of all matches in the text.
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
of the previous character.

You can find a list of all the special characters in the [Python documentation]
(https://docs.python.org/3/library/re.html#regular-expression-syntax).

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
| `|`       | Matches either the expression before or after the `|`.                                                         |
| `(...)`    | Matches the expression inside the parentheses and groups it. Parentheses can be used to group expressions and to define the precedence of operators, like in math. |
| `\`        | Escapes a special character (used to match, for example, the `.`).                                                                                                |
| `\d`       | Matches any digit character.                                                                                                                                      |
| `\D`       | Matches any non-digit character.                                                                                                                                  |
| `\w`       | Matches any alphanumeric character.                                                                                                                               |
| `\W`       | Matches any non-alphanumeric character.                                                                                                                           |
| `\s`       | Matches any whitespace character.                                                                                                                                 |
| `\S`       | Matches any non-whitespace character.                                                                                                                             |
| `(?:...)`  | Matches the expression inside the parentheses, but does not add it to the match.                                                                                  |
| `(?=...)`  | Matches if ... matches next, but doesn’t consume any of the string. This is called a _lookahead assertion_                                                        |
| `(?!...)`  | Matches if ... doesn’t match next. This is a _negative lookahead assertion_.                                                                                      |
| `(?<=...)` | Matches if the current position in the string is preceded by a match for ... that ends at the current position. This is called a _lookbehind assertion_.          |
| `(?<!...)` | Matches if the current position in the string is not preceded by a match for ... that ends at the current position. This is a _negative lookbehind assertion_.    |
| `(?...)`   | Matches the expression inside the parentheses, but does not add it to the match. This is called a _non-capturing group_.                                          |


The following table shows some examples of regular expressions and the strings that they match:

| Regular expression | Matches                                            |
|--------------------|----------------------------------------------------|
| `a`                | The letter "a"                                     |
| `abc`              | The string "abc"                                   |
| `a.c`              | The string "abc", "axc", "a1c", etc.               |
| `a.*c`             | The string "ac", "abc", "abbc", "abbbc", etc.      |
| `a+c`              | The string "ac", "aac", "aaac", etc.               |
| \d{2}-\d{3}        | A string of the form "12-345"                      |
| (.*)(\d{8}-[A-Z]{3})(.*) | e.g. "John Doe 12345678-ABC"                       |
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

## Basic matching with `search`

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




