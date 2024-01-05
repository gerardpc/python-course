# Python projects

## Introduction

In this section we will learn how to create a Python project from scratch. This project will be hosted in a
remote repository, and we will learn how to create a Python package from it.

We will use [Poetry](https://python-poetry.org/) to manage our project dependencies (i.e., the libraries that
our project uses), and the following extra tools to help us with the development process:

* **Git**: the most popular version control system. This will allow us to keep track of the changes in our code
between different versions of our project.
* **GitHub**: a website to host our remote repository (although we could use other alternatives, such as 
[GitLab](https://about.gitlab.com/) or [Bitbucket](https://bitbucket.org/)). This will allow us to share our code
with other people and to collaborate with them.
* **Typer**: a library to create command-line interfaces (CLIs) in Python. This will allow us to create a 
program that uses our Python code and can be executed from the terminal.
* **Ruff**: an (extremely fast) Python linter and code formatter. This will help us to keep our code clean and
consistent, following a "standard" style of writing Python code.

If we have more time, we can also add to our project:

* **Pytest**: a library to create unit tests in Python.
* **[NiceGUI](https://nicegui.io/)**: a library to create quick graphical user interfaces (GUIs) in Python in web browsers (other alternatives
are [Streamlit](https://streamlit.io/) and [Dash](https://plotly.com/dash/)).

!!!note
    Some basic nomeclature:
    * **Python module**: a file containing Python code. It can be imported by other Python modules.
    * **Python package**: a directory containing several Python modules or other Python packages. 
    It can be imported by other Python modules.

## Project structure



## Converting your repository into a Python package

### Python packages introduction

A `.whl` (**wheel**) file is a distribution package file saved in Python’s wheel format. It is a standard format 
installation of Python distributions and contains all the files and metadata required for installation. 
The WHL file also contains information about the Python versions and platforms supported by this wheel file. 
WHL file format is a ready-to-install format that allows running the installation package without building the 
source distribution.

!!!note
    * All else being equal, wheels are typically smaller in size than source distributions.
    * Installing from wheels directly avoids the intermediate step of building packages off of 
    the source distribution.

A `.whl` file is essentially a zip archive with a specially crafted filename that tells installers what 
Python versions and platforms the wheel will support.

### Creating a wheel with Poetry

We will use [Poetry](https://python-poetry.org/) to create a wheel file for our project.

