# **<project>**

This is the code for our course project in CMPUT 624 Fall 2023. 

<p align="center">
  <img src="assets/myproject.png" alt="General illustration of the project" width="60%"/>
</p>

# Why SparsEEG?

What is the initial motivation of this project ?

## Objective of the project

How this project will be considered achieved ?

# Table of Contents

-   [**Installation**](#installation)
-   [**Usage**](#usage)

# Installation

Make sure you have Python3 installed:

**1. Install package dependencies:**

```bash
pip install --upgrade "jax[cpu]"
```
```bash
pip3 install torch torchvision torchaudio
```
```bash
pip install scikit-learn
```
```bash
!pip install -q git+https://github.com/google/CommonLoopUtils
```

**2. Install the local package with pip IN EDIT MODE in your terminal from the main package folder:**

```bash
python -m pip install -e .
```

# Usage

**Run the project with the following command:**

```bash
python -m sparseeg -s "SAVE_DIR" -i "INT" "EXPERIMENT_FILE" "CONFIG_FILE"
```


