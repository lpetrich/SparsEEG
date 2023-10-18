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

pip install --upgrade pip
pip install --upgrade "jax[cpu]"
pip3 install torch torchvision torchaudio
pip install scikit-learn

Install the latest version of clu:
```bash
!pip install -q git+https://github.com/google/CommonLoopUtils
```

**2. Install the package with pip in your terminal:**

```bash
pip3 install git+https://github.com/IRLL/<project>.git
```

# Usage

**Run the project with the following command:**

```bash
python -m <project>
```

**For testing you can run the main file directly from src/project:**
```bash
python3 __main__.py -s "SAVE_DIR" -i "INT" experiment/default.py config/dense.yaml
```
