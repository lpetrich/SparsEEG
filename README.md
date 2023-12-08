## Objective of the project

The goal of this project is to evaluate how sparse neural networks perform on a
classification task using EEG data. The goal is to better understand how we can
use sparse neural networks in a brain-computer interface system.

# Table of Contents

-   [**Installation**](#installation)
-   [**Usage**](#usage)

# Installation

Make sure you have Python3 installed:

**1. Clone the repository and `cd` into it
```bash
git clone git@github.com:lpetrich/SparsEEG.git
cd SparsEEG
```

**2. Install package dependencies:**

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

**3. Download the dataset files and unzip into `src/sparseeg`:**
https://drive.google.com/file/d/1f49lJ2fuja27wC-ze8o6CC3O1ENMzbMJ/view?usp=share_link


# Usage

**Run the project with the following command:**

```bash
python -m sparseeg -s "SAVE_DIR" -i "INT" "EXPERIMENT_FILE" "CONFIG_FILE"
```

This will run an experiment outlined by `CONFIG_FILE`. The final configuration
files used in our experiments can be found at `./src/sparseeg/config/`. The
`CONFIG_FILE` command-line argument should be one of the config files in this
directory, but without the `./src/sparseeg/config/` prefix.


The `EXPERIMENT_FILE` argument outlines which experiment to run. This can be
one of `default.py` or `nested_cv.py`. Our final experiments for the paper used
the `default.py` experiment, and we do not provide any configuration files for
the `nested_cv.py` experiment. All configuration files should work with the
`default.py` experiment.

The `-i` flag indicates the hyperparameter setting to use. This is a 0-based
index into the hyperparameters settings in the configuration file. See
[Configuration Files](#configuration-files) for more information.

The `-s` flag tells where to save the data generated from the program. By
default, this is `./results`.

# Configuration Files

A very simplified example configuration file is:

```
{
train_percent: [0.8]
valid_percent: 0.1
weighted_loss: true
epochs: 500
seed: [0, 1, 2]
save_dir: "single_subject_weight_pruning_500epochs_weighted"
model:
  type: "DenseMLP"
  hidden_layers: [[1024, 1024]]
  activations: [["relu", "relu"]]
  optim:
    type: "adam"
    args: [[]]
    kwargs:
        learning_rate: [0.01, 0.001, 0.0001]
}
```

This configuration file outlines 9 different hyperparameter configurations for
algorithm `DenseMLP`, one hyperparameter setting for each combination of
"learning_rate" and "seed". When determining hyperparameter settings in a
configuration file, we iterate over each top-level `Iterable` python type.
Here, the top-level iterables are:

```
seed: [0, 1, 2]					# HERE
model:
  hidden_layers: [[1024, 1024]]			# HERE
  activations: [["relu", "relu"]]		# HERE
  optim:
    args: [[]]					# HERE
    kwargs:
        learning_rate: [0.01, 0.001, 0.0001]	# HERE
```

Since only two of these (`seed` and `learning_rate`) are top-level iterables of
more than one element, they are the only ones swept over. Of course, the `seed`
here is swept over, but in our experiments we **did not** tune over seeds. This
functionality simply is a mechanism of sweeping specific values for random
seeds, and random seeds should **not** be tuned over when analyzing the
resulting data.

We refer to these hyperparameter combinations with 0-based indexing:

- index 0 has `seed = 0` and `learning_rate = 0.01`
- index 1 has `seed = 1` and `learning_rate = 0.01`
- index 2 has `seed = 2` and `learning_rate = 0.01`
- index 3 has `seed = 0` and `learning_rate = 0.001`
- index 4 has `seed = 1` and `learning_rate = 0.001`
- index 5 has `seed = 2` and `learning_rate = 0.001`
- index 3 has `seed = 0` and `learning_rate = 0.0001`
- index 4 has `seed = 1` and `learning_rate = 0.0001`
- index 5 has `seed = 2` and `learning_rate = 0.0001`

When running experiments from the command line, you specify this index using
the `-i/--index` option. A single run of the experiment will then be executed
using the associated hyperparameter setting. To run multiple hyperparameter
settings in parallel, you can use [GNU
Parallel](https://www.gnu.org/software/parallel/). This is useful for HPC
clusters. Otherwise, if you are using a slurm-based HPC cluster, then you can
use array jobs, which is how we ran our experiments. The slurm job scripts can
be found in the `parallel` directory.

One data file will be saved each time you run an experiment. To combine these
individual data files into a single data file, you can use the `combine.py`
script:

```bash
./combined.py -s "FILE_NAME" DIR CONFIG
```

where `FILE_NAME` is the name of the resulting combined file, `DIR` is the
directory holding the data files, and `CONFIG` is the configuration file used
to run the experiments which generated the data.

# Plotting

To generate plots similar to what were used in our experiments you can use the
`plot/accuracy_vs_epochs_combined.py` file, simply use:

```bash
python3 ./plot/accuracy_vs_epochs_combined.py PLOT_TITLE DATA_FILES
```

where `PLOT_TITLE` is the title of the plot and `DATA_FILES` are the data files
to plot. If multiple data files have been combined into one using the
`combine.py` script, then this file will plot the performance of the tuned
hyperparameters.
