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
* Rename `dbconfig_init.py` to `dbconfig.py` and then you need to artificially fill in the database information.

3. Configure the database

You can use ./upload.cmd to upload the project to Github.

**Document wait to be updated.**


For each batch_ensemble, every different configuration will have different batch_ensemble id.

For shell scripts, the path of the shell scripts should be the absolute path, i.e., .sh files will be affected by the current directory.

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

[//]: # (```)

# 解释： 有时候即使设置了K，也不一定可以聚成K类，所有有些方法的结果相同就如此解释