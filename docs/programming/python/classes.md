# Classes and objects

## Classes and Object-Oriented-Programming

Objects and classes are a way to group up a bunch of function and variables into a single "thing". When you get 
all the way down to it, this is simply a way of organizing everything into groups that make sense. 
There are benefits down the road for making things easier to understand, debug, extend, or maintain, but 
basically it is just a way to make code easy to understand and develop.

This might sound very abstract, but in practice it is not: everything is an object in Python! 
```int`, `dict`, `list`... etc. are all different types of objects.

To be more precise, `int`, `dict`, `list`... are known as **classes**, and objects are particular instances
of such classes. In the next example: 
```python
a = 2
```
`int` would be the class, and `a` would be the object (or instance, they are used interchangeably). So, 
in other words, classes are the blueprint from which objects are created.

!!!note 
    It is often helpful to make analogies of classes with animals: for instance, if `dog` is a class,
    then `Max = dog()` (one dog in particular) is the instance.

Making use of classes in programming is what is known as **Object-Oriented-Programming** (OOP). OOP is highly 
concerned with code organization, reusability, and encapsulation. OOP is partially in contrast to **Functional 
Programming**, which is a different paradigm used a lot in Python. Not everyone who programs in Python 
uses OOP. 

!!!note
    So, should we use functions or classes when programming? As a general rule of thumb, we can think of classes/objects 
    as _nouns_ and functions/methods as _verbs_. In other words, functions **do** specific things, classes **are** 
    specific things. With a language like Python we are not forced to choose: we can use just a little bit of each,
    when it is most convenient.

### Defining classes

Python makes it very easy to define our own classes. A very basic class would look something like this:

```python
class MyClass:
    variable = "blah"

    def function(self):
        print("This is a message inside the class.")
```

!!!note
    Variables in classes are also known as _attributes_, and functions in classes are also known as _methods_.
    These words are complete synonyms.

We'll explain why you have to include that `self` as a parameter a little bit later. 
First, to assign the above class to an object you would do the following:

```python
class MyClass:
    variable = "blah"

    def function(self):
        print("This is a message inside the class.")

myobjectx = MyClass()
```

Now the variable `myobjectx` holds an object of the class `MyClass` that contains the variable 
and the function defined within the class called `MyClass`.
Accessing Object Variables

### Accessing object variables

To access the variable inside the newly created object `myobjectx` you would do the following:
```python
class MyClass:
    variable = "blah"

    def function(self):
        print("This is a message inside the class.")

myobjectx = MyClass()

myobjectx.variable
```

So for instance the below would output the string `blah`:
```python
class MyClass:
    variable = "blah"

    def function(self):
        print("This is a message inside the class.")

myobjectx = MyClass()

print(myobjectx.variable)
```

You can create multiple different objects that are of the same class (i.e., have the same variables and functions 
defined). However, each object contains independent copies of the variables defined in the class. For instance, 
if we were to define another object with the `MyClass` class and then change the string in the variable above:

```python
class MyClass:
    variable = "blah"

    def function(self):
        print("This is a message inside the class.")

myobjectx = MyClass()
myobjecty = MyClass()

myobjecty.variable = "yackity"

# Then print out both values
print(myobjectx.variable)
print(myobjecty.variable)
```
nothing would happen to the first object.

### Accessing object functions

To access a function inside of an object you use notation similar to accessing a variable:

```python
class MyClass:
    variable = "blah"

    def function(self):
        print("This is a message inside the class.")

myobjectx = MyClass()

myobjectx.function()
```

The above would print out the message `This is a message inside the class.`

#### The meaning of `self`

The built-in `self` is used by a class to refer to a specific instantiation of the class.
This is easier seen with an example. Consider this class:

```python
class Dog:
    def set_name(name):
        dog_name = name
```

and now compare it to:

```python
class Dog:
    def set_name(self, name):
        self.dog_name = name
```

The idea of the function `set_name` is clear: we want to give a name to our object of the class `Dog`.
So, what's the difference between both cases? In the first case, we are defining `dog_name` inside of the 
function `set_name`, but when we leave the function this "inside" variable disappears. Hence, the function
has no effect.

!!!note
    Actually, the first function would not work for reasons a bit more obscure, but that doesn't matter
    for now. See [this](https://stackoverflow.com/questions/23944657/typeerror-method-takes-1-positional-argument-but-2-were-given-but-i-only-pa) 
    if you're interested in reading more aaa.

What we want to do is _access the particular dog instance_, and set a name to that instance. This is the 
purpose of `self`: it means "our current instance".

#### `__init__()` function

The `__init__()` function is a special function that is called when the class is being instantiated. 
`__init__` doesn't _initialize_ a class, it initializes an instance of a class or an object. 

The difference between variables assigned inside the `__init__()` method versus variables assigned
in the class definition is that in the first case we are only defining values for one particular
instance, whereas if we define a value in the class definition, all objects from that class will have
the same value. 

!!!note
    In other words, each dog has colour, but dogs as a class don't: hence the color for one particular dog
    should be defined inside the `__init__()` method. 

    The class is a concept of an object. When you see Fido and Spot, you recognise their similarity, 
    their doghood. That's the class. However, when you say
    ```python
    class Dog:
        def __init__(self, legs, colour):
            self.legs = legs
            self.colour = colour
    
    fido = Dog(4, "brown")
    spot = Dog(3, "mostly yellow")
    ```
    
    You're saying, Fido is a brown dog with 4 legs while Spot is a bit of a cripple and is mostly yellow. 

### Class inheritance

**Inheritance** is a mechanism that allows you to create a hierarchy of classes that share a set of 
properties and methods by deriving a class from another class. This is extremely useful because it allows
us to write _less code_ thanks to code reuse.

For instance, imagine that we need to define a class for a dogs and cats (and potentially many more). Instead of 
writing everything every time for each other animal that we define, we could make them inherit from another parent
class `Animal`:

```python
class Animal:
    def __init__(self, age, height):
        self.age = age
        self.height = height
    def print_height(self):
        print(self.height)
                
class Dog(Animal):
    def __init__(self, age, height, race):
        super().__init__(age, height)
        self.race = race        
        
class Cat(Animal):
    def __init__(self, age, height, meowness):
        super().__init__(age, height)
        self.meowness = meowness
        
my_cat = Cat(age=2, height=35, meowness=0.5)
some_dog = Dog(age=1, height=50, race="beagle")

my_cat.print_height()
print(some_dog.race)

# Output
35
beagle
```

In this example, `Dog` and `Cat` _inherit_ from the class `Animal`. This means that all
`Animal` methods and attributes are also available to `Dog` and `Cat` as if they were defined
inside their definitions. 

!!!note
    In the previous code snippet you might notice the use of the `super()` function.
    `super()` is a built-in function returns a proxy that allows us to access methods of the base class.
    We use it to refer to the base class: in the previous example, we are using it to call the `Animal.__init__`
    function from the `Dog` and `Cat` `__init__` functions.