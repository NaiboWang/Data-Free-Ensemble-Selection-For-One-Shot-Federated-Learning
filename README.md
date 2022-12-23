# Ensemble Selection for One-Shot Federated Learning in Machine Learning Model Market

## Initialization

1. Install the required packages

```bash
pip install -r requirements.txt
```

2. Initialize the environment and files

```bash
python init.py
```

This step will help you to:
* Make the directory for the algorithm.
* Rename `dbconfig_init.py` to `dbconfig.py` and then you need to artificially fill in the database information (default database name is `exps`, there are two tables in the database, which are `ensemble_selection_exp_results`  `ensemble_selection_results`).

1. Configure the database


## Partition the dataset

### Four types of partition strategies

### The concept of dataset batch ID

## Generate/Train models based on the partitioned dataset

## Evaluate the models and get the output results of the models

## Run baseline algorithms

### The concept of ensemble batch ID

For each batch_ensemble, every different configuration will have different batch_ensemble id.

## Run out DeDES algorithm

## Intergrity Checking

### Check the partitioned dataset

### Check the trained models

### Check the test results of the models

## Experimental data processing and visualization

## Other Notes

* You can use ./upload.cmd to upload the project to Github.



<!-- For shell scripts, the path of the shell scripts should be the absolute path, i.e., .sh files will be affected by the current directory.

E.g., if you are in the directory of `A/B`, then the following two commands will have **different** results:

```bash
sh B/C.sh # run the script C.sh in the directory of A 
sh C.sh # run the script C.sh in the directory of A/B
```

[//]: # (For python scripts, the path of the python scripts will not be affected by the current directory. E.g., if you are in the directory of `A/B`, then the following two commands will have **same** results:)

[//]: # ()
[//]: # (```bash)

[//]: # (python B/C.py # run the script C.py in the directory of A)

[//]: # (python C.py # run the script C.py in the directory of A/B)

[//]: # (```) -->



# The remaining parts are still updating because the author has been affected by COVID-19 and is now having a high fever, but we will finish this readme file by Dec 25, 2022, thank you for your patience!