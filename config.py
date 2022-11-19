import os
import sys
from prettytable import PrettyTable

exp_config = {
    "batch": 221,  # which batch of experiment
    "batch_ensemble":221, # which batch of the final written results
    "selection_method":"mixed", # how to select the model inside a cluster, data, CV or mixed
    "indexes": [],  # 方法会自动修改
    "partition": "",
    "partitions" : ["iid-diff-quantity", "homo", "noniid-labeldir", "-1"],
    "split": "cifar100",
    "dataset": "cifar100",
    "device":"cuda:0",
    "party_num": 5,
    "model":"resnet50",
    "K": 2,  # 选几个cilents
    "cluster_method": "",  # 聚类方法
    "last_layer":True, # whether only read the last layer's parameter to do the clustering
    "layer": 0, # which layer to do the clustering, 0 means all layers
    "dr_method":["noDimensionReduction", 0.3], # Dimension Reduction Method, [noDimensionReduction/PCA/Kernel_PCA, 0.3 (What percentage of dimensions do we want to keep) ]
    "normalization":1, # if maxmin normalization for the model parameters
    "avg":"none", # what kind of avergaing method will we use: mean_avg or fed_avg or none (don't use model averging)
    "label_distribution":False, # whether calculate label distribution oracle, True or False
    "input_channels":3,
    "num_classes":100,
    "additional_parameters":[] # additional parameters for the code
}

def get_noniid_label_number_split_name(split="digits"):
    if split == "digits":
        return "noniid-#label3"
    elif split == "letters":
        return "noniid-#label8"
    elif split == "balanced":
        return "noniid-#label18"
    elif split == "cifar10":
        return "noniid-#label4"
    elif split == "cifar100":
        return "noniid-#label45"
    elif split == "tinyimagenet":
        return "noniid-#label40"

def get_label_numbers(split="digits"):
    if split == "digits" or split == "cifar10":
        return 10
    elif split == "letters":
        return 26
    elif split == "balanced":
        return 47
    elif split == "cifar100":
        return 100
    elif split == "tinyimagenet":
        return 200

def get_dataset_shape(dataset="cifar10"):
    if dataset == "emnist":
        return (1,28,28)
    elif dataset == "cifar10" or dataset == "cifar100":
        return (3,32,32)
    elif dataset == "tinyimagenet":
        return (3,64,64)