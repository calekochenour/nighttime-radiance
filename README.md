# Nighttimme Radiance
Python scripts and Jupyter Notebooks for working with nighttime radiance data.

## Contents

The project contains folders for all stages of the project workflow as well as other files necessary to run the code.

### `01-code-scripts/`

Contains all Python and Jupyter Notebook files.

### `02-raw-data/`

Contains all original/unprocessed data.

### `03-processed-data/`

Contains all created/processed data.

### `04-graphics-outputs/`

Contains all figures and plots.

### `05-papers-writings/`

Contains all written content.

### `Makefile`

Contains instructions to run the Python scripts in the `01-code-scripts/` folder.

### `environment.yml`

Contains all information to create the Conda environment required to run the Python and Jupyter Notebook files in the `01-code-scripts/` folder.  

## Prerequisites

To run the Python and Jupyter Notebook files in the `01-code-scripts/` folder, you will need:

 * Conda ([Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://docs.anaconda.com/anaconda/install/))

## Local Setup Instructions

The instructions expect you to have a local copy of this GitHub repository.

### Create and Activate Conda Environment

From the terminal, you can create and activate the Conda environment.

Create environment:

```bash
$ conda env create -f environment.yml
```

Activate environment:

```bash
$ conda activate nighttime-radiance
```

### Open Jupyter Notebook

Once you activate the Conda environment, you can work with the Jupyter Notebook files.

Open Jupyter Notebook:

```bash
$ jupyter notebook
```
