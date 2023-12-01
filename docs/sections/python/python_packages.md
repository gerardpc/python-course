# Python packages

## Package distribution

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
