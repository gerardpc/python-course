# IO, files and exceptions

## IO and files

### Introduction 

One of the most common tasks that you can do with Python is reading and writing files. 
Whether it’s writing to a simple text file, reading a complicated server log, or even analyzing raw 
byte data, all of these situations require reading or writing a file.

Before we can go into how to work with files in Python, it’s important to understand what exactly a 
file is and how modern operating systems handle some of their aspects.

At its core, a file is a contiguous set of bytes used to store data. This data is organized 
in a specific format and can be anything as simple as a text file or as complicated as a 
program executable. In the end, these byte files are then translated into binary 1 and 0 for 
easier processing by the computer.

Files on most modern file systems are composed of three main parts:

- **Header**: metadata about the contents of the file (file name, size, type, and so on)
- **Data**: contents of the file as written by the creator or editor
- **End of file (EOF)**: special character that indicates the end of the file

What this data represents depends on the format specification used, which is typically represented by an extension.

### File paths

When you access a file on an operating system, a file path is required. The file path is a string 
that represents the location of a file. It’s broken up into three major parts:

- **Folder Path**: the file folder location on the file system where subsequent folders are separated 
  by a forward slash / (Unix) or backslash \ (Windows)
- **File Name**: the actual name of the file
- **Extension**: the end of the file path pre-pended with a period (.) used to indicate the file type

Here’s a quick example. Let’s say you have a file located within a file structure like this:

    /
    │
    ├── path/
    |   │
    │   ├── to/
    │   │   └── cats.gif
    │   │
    │   └── dog_breeds.txt
    |
    └── animals.csv

Let’s say you wanted to access the cats.gif file, and your current location was in the same folder as path. 
In order to access the file, you need to go through the path folder and then the to folder, finally 
arriving at the `cats.gif` file. The Folder Path is `path/to/`. The File Name is cats. The File 
Extension is `.gif`. So the full path is `path/to/cats.gif`.

### Opening and Closing a File in Python

When you want to work with a file, the first thing to do is to open it. This is done by invoking 
the `open()` built-in function. `open()` has a single required argument that is the path to the file. 
`open()` has a single return, the file object. Once we are done with the file, we need to close it:

```python
file = open('dog_breeds.txt')
# do something with file
file.close()
```

It’s important to remember that it’s your responsibility to close the file! This is why it's always
recommended to use the `with` statement when dealing with files:

```python
with open('dog_breeds.txt') as reader:
    # Further file processing goes here
    ...
```

When the `with` statement is finished, everything goes back to normal (and we don't need to remember
to close anything). 

When opening a file, we are (directly or indirectly) using one of the different modes provided by Python.
The most commonly used modes are the following:

- `r`: Open for reading (default mode if nothing is specified)
- `w`: Open for writing, truncating (overwriting) the file first
- `a`: Open for writing, appending to the end of the file
- `rb` or `wb`: Open in binary mode (read/write using byte data)

Reading example:

```python
with open('dog_breeds.txt', 'r') as file:
    # Read & print the entire file
    print(file.read())

# Output
Pug
Jack Russell Terrier
English Springer Spaniel
German Shepherd
Staffordshire Bull Terrier
Cavalier King Charles Spaniel
Golden Retriever
West Highland White Terrier
Boxer
Border Terrier
```

Writing example:
```python
with open('dog_breeds_reversed.txt', 'w') as file:
    # Write the dog breeds to the file in reversed order
    for line in reversed(file):
        file.write(line)
```

Appending example:
```python
with open("test.txt", "a") as file:
    file.write("appended text")
```

!!!note
    To read a binary file, in contrast to a text file, we would use `open("filename", "b")`

### Reading and writing JSON files

To load the data from a `.json` file, we use the following code structure:

```python
import json

with open("strings.json", "r") as file:
    d = json.load(file)
    print(d)
```
In this example, the `strings.json` file is loaded into a variable `d` of type `dict`. If, on the contrary,
we want to _write_ a new JSON file from a dictionary variable we already have, we would use:
```python
import json

my_dict = ...

with open("out_file.json") as file:
    json.dump(my_dict, file)
```
Bear in mind, however, that the structure of a JSON file is included on its use of curly brackets, in contrast
to YAML files or Python code, that use indentation for that. Hence, the `out_file.json` (from the last example)
will not be pretty to look at. If we want to force the use of indentation, we can add the optional parameter `indent`
like so:
```python
import json

my_dict = ...

with open("out_file.json") as file:
    json.dump(my_dict, file, indent=4)
```

!!!note
    The `json` package is included with the standard Python installation, we don't need to install it.

The `json` package also has functions to serialize a Python object into a JSON string, and also to perform
the inverse operation and deserialize a JSON string into a Python dictionary:

* Dictionary to JSON string:
    ```python   
    import json 
        
    # Data to be written 
    dictionary ={ 
      "id": "04", 
      "name": "sunil", 
      "department": "HR"
    } 
        
    # Serializing json  
    json_object = json.dumps(dictionary, indent=4) 
    print(json_object)
  
    # Output
    {
        "department": "HR",
        "id": "04",
        "name": "sunil"
    }    
    ```
* JSON string to dictionary:
    ```python   
    import json
  
    data = """
        {  
        "Name": "Jennifer Smith",  
        "Contact Number": 7867567898,  
        "Email": "jen123@gmail.com",  
        "Hobbies":["Reading", "Sketching", "Horse Riding"]  
        }
    """
        
    # parse data:  
    res = json.loads(data)  
    ```