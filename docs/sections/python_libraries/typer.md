# Typer

Typer is a library for building command-line interface (CLI) applications in Python. It is built on top of 
[Click](https://click.palletsprojects.com/en/7.x/) (another popular CLI library) and makes it very easy 
to build complex CLI applications by providing a clean interface for defining commands and arguments.

<figure markdown>
  ![Typer](../../images/typer.png){ width="500" }
  <figcaption>A screenshot of Typer from a terminal.</figcaption>
</figure>

## Installation

Typer can be installed with `pip`:

```bash
pip install typer
```

Or, in a poetry project:

```bash
poetry add typer
```

## Usage

It is recommended to use Typer in a Python project with a clean directory structure. This makes it easier to
organize the code and to add new commands and arguments. In what follows, we will assume that the project
directory structure looks like this:

```bash
.
├── cli # The directory containing the CLI code
│   ├── __init__.py
│   ├── command1.py
│   ├── command2.py
│   └── command3.py
├── __main__.py # T
```

where `cli` is a folder inside the `my_project` directory. In this last example, our CLI has three commands
(`command1`, `command2`, and `command3`) and a `__main__.py` file that will be used to run the CLI.