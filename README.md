# InvertSy ![GitHub top language](https://img.shields.io/github/languages/top/InsectRobotics/InvertSy) [![GitHub license](https://img.shields.io/github/license/InsectRobotics/InvertSy)](https://github.com/InsectRobotics/InvertSy/blob/main/LICENSE) ![GitHub last-commit](https://img.shields.io/github/last-commit/InsectRobotics/InvertSy) [![Build Status](https://travis-ci.com/InsectRobotics/InvertSy.svg?token=tyo7V4GZ2Vq6iYPrXVLD&branch=main)](https://travis-ci.com/InsectRobotics/InvertSy)

This Python package implements environments such as the *sky* and an *AntWorld of
vegetation*, using  simple-to-install python packages, e.g. NumPy and SciPy. These
environments contain information that humans can or cannot detect but invertebrates
definitely can (e.g. polarised light in the sky). This package also contains some
examples of how to use the [InvertPy](https://github.com/InsectRobotics/InvertPy) package.


### invertsy.agent

Python package that allows the development of agents that observe their environment with
their sensors, process this information using their brain components and act-back in their
environment. This is useful for closed-loop experiments as the actions of the agents can
also change their environment and therefore affect their observations. 

### invertsy.env

Python package that implements the environments where the agents can get their observations
from. These are models of the sky, 3D object (e.g. vegetation), odour gradients, etc, that
affect what the agents observe using their sensors. These environments are in abstract form
and are rendered using the sensors of the agents, designed for invertebrate observations,
which allows for more invertebrate-like stimulation.

### inversy.sim

Python package implementing a variety of simulations and animations related to tasks that
invertebrates are put through. These simulations collect logs of the data produced during
the task and also allow for visualisation of the resulting behaviour by matplotlib
animations.

## Environment

In order to be able to use this code, the required packages are listed below:
* [Python 3.8](https://www.python.org/downloads/release/python-380/)
* [NumPy](https://numpy.org/)  >= 1.20.1
* [SciPy](https://www.scipy.org/) >= 1.6.1
* [Matplotlib]() >= 3.3.4
* [InvertPy](https://github.com/InsectRobotics/InvertPy)

## Installation

In order to install the package and reproduce the results of the manuscript you need to clone
the code, navigate to the main directory of the project, install the dependencies and finally
the package itself. Here is an example code that installs the package:

1. Clone this repo.
```commandline
mkdir ~/src
cd ~/src
git clone https://github.com/InsectRobotics/InvertSy.git
cd InvertSy
```
2. Install the required libraries. 
   1. using pip :
      ```commandline
      pip install -r requirements.txt
      ```

   2. using conda :
      ```commandline
      conda env create -f environment.yml
      ```
3. Install the package.
   1. using pip :
      ```commandline
      pip install .
      ```
   2. using conda :
      ```commandline
      conda install .
      ```

Note that the [pip](https://pypi.org/project/pip/) project might be needed for the above installation.

## Report an issue

If you have any issues installing or using the package, you can report it
[here](https://github.com/InsectRobotics/InvertSy/issues).

## Author

The code is written by [Evripidis Gkanias](https://evgkanias.github.io/).

## Copyright

Copyright &copy; 2021, Insect robotics Group, Institute of Perception,
Action and Behaviour, School of Informatics, the University of Edinburgh.
