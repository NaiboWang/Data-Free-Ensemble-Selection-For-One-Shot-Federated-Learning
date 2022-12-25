# Ensemble Selection for One-Shot Federated Learning in Machine Learning Model Market

## Initialization

1. Install the required packages

```bash
pip install -r requirements.txt
```

We also recommend to install pytorch with `conda`.

2. Initialize the environment and files

```bash
python init.py
```

This step will help you to:
* Make the directory for the algorithm.
* Rename `dbconfig_init.py` to `dbconfig.py` and then you need to artificially fill in the database information (username, password, ip, and port name).


3. Configure your mongodb database

The default database name is `exps`, you will need to create three tables in your database, which are `train_config`, `ensemble_selection_exp_results` and `ensemble_selection_results`.

## Partition the dataset

Use the following command to partition the dataset:

```bash
python generate_dataset.py --dataset <dataset_name> --split <split_name> --ID <batch_ID> --num_clients [<num_clients>,<num_clients>,...,<num_clients>]
```

Every dataset will have a `batch ID` for us to identify and distinguish different batch of datasets. When training models, our code will seek the specified batch ID of the dataset. If the batch ID is not specified, the code will use the default batch ID, which is `0`.

The code will help you to automatically partition the dataset according to the four types of partition strategies (homo, iid-dq, noniid-lds, noniid-lk).

E.g., you can partition the `EMNIST Digits` dataset with batch ID `1` into 100 clients with all 4 partition strategies by using the following command:

```bash
python generate_dataset.py --dataset emnist --split digits --ID 1 --num_clients [100]
```

After partitioning the dataset, you will get 4 files in the `files` directory, which represent the 4 partition strategies. The file name is in the format of `dataset_split_partition_(num_clients)_b(ID).pkl`:

* `files/emnist_digits_homo_100_b1.pkl`
* `files/emnist_digits_iid-diff-quantity_100_b1.pkl` // iid-dq
* `files/emnist_digits_noniid-labeldir_100_b1.pkl` // noniid-lds
* `files/emnist_digits_noniid-#label3_100_b1.pkl` // noniid-lk


## Generate/Train models based on the partitioned dataset

To train the model on the partitioned dataset, you can use the following command:

```bash
python train_model.py --index $i --partition $partition --party_num $party_num --split $split --device $device --batch $batch --dataset $dataset --model $model --input_channels $input_channels --num_classes $num_classes
```

* `$i` is the index of the client.
* `$partition` is the partition strategy.
* `$party_num` is the number of clients.
* `$split` is the split name of the dataset.
* `$device` is the GPU device name.
* `$batch` is the batch size.
* `$dataset` is the dataset name.
* `$model` is the model name.
* `$input_channels` is the number of input channels.
* `$num_classes` is the number of classes.

E.g., you can train the model `resnet50` on the `10th` (start from 0) data of the partitioned `EMNIST Digits` dataset with batch ID `1` with `100` clients using the `homo` partition strategy on your local device `cuda:0` by using the following command:

```bash
python train_model.py --index 9 --partition homo --party_num 100 --split digits --device cuda:0 --batch 1 --dataset emnist --model resnet50 --input_channels 1 --num_classes 10
```

After training the model, you will get a model file in the `models` directory.

## Evaluate the models and get the output results of the models

To evaluate the model, you can use the following command:

```bash
python test_model.py --index $i --partition $element --party_num $party_num --split $split --batch $batch --device $device --dataset $dataset --input_channels $input_channels --num_classes $num_classes --model $model
```

The configuration of the parameters is the same as the training model section.

Example command:

```bash
python test_model.py --index 9 --partition homo --party_num 100 --split digits --device cuda:0 --batch 1 --dataset emnist --model resnet50 --input_channels 1 --num_classes 10
```

The test results and statistics will be saved to the dataset and also local files in the `models` directory.

Before you run any ensemble selection method, ensure that you have trained all models you need to select from and already tested by the `train_model.py` and `test_model.py` scripts respectively.

## Run baseline algorithms

Similar as the `batch ID`, we will have a `ensemble batch ID` for each run of the ensemble algorithms.

To run the baseline algorithms, you can use the following command:

```bash
python ensemble_selection_baselines.py --parameter <parameter_value> --parameter <parameter_value> ... --parameter <parameter_value>
```

Please refer to the `config.py` script to see all configurable parameters, e.g., to set the dataset name, pass the parameter `--dataset <dataset_name>`; to specify `K` as `20`, pass `--K 20`.

## Run our DeDES algorithm

Similarly, to run our DeDES algorithm, you can use the following command:

```bash
python ensemble_selection_clustering.py --parameter <parameter_value> --parameter <parameter_value> ... --parameter <parameter_value>
```

One example of the command is:

```bash
python ensemble_selection_clustering.py --split balanced --party_num 400 --K 10 --batch 2 --batch_ensemble 8 --selection_method mixed --normalization 1 --last_layer 0 
```

The `--batch` is the `batch ID` and the `--batch_ensemble` is the `ensemble batch ID`.

Then the results of the baselines and DeDES will be saved to the database in the `ensemble_selection_exp_results` table.

<!-- 
## Intergrity Checking

### Check the partitioned dataset

### Check the trained models

### Check the test results of the models

## Experimental data processing and visualization -->

## Other Notes

* `batch_train.sh`, `batch_test.sh` can help you to train and test models in batch.

* The scripts end with `_script.sh` inside the `batches` directory are the scripts used to run the experiments in the paper in batch.

    E.g., to run all baselines methods and DeDES on the `EMNIST Digits` dataset you can use the following command:

    ```bash
    cd batches
    bash batch_ensemble_script_emnist_digits.sh
    ```

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
