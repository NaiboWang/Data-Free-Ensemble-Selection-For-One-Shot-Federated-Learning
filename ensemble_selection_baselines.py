import copy
import json
import pickle
import random
import sys
import time

from common_tools import repeat
from dbconfig import ensemble_selection_exp_results, ensemble_selection_results
from ensemble import main
# from .test import generate_timestamp
from commandline_config import Config
from config import get_noniid_label_number_split_name, exp_config
from dbconfig import get_path

DATASET_DIR, MODEL_DIR = get_path()

# Cross-Validation Selection


def CV_selection(config):
    K = config["K"]
    party_num = config["party_num"]
    total = 0
    party_local_validation_accuracies = []
    meta_data = "%s_%s_%s_%s_b%d" % (
        config.dataset, config.split, config.model, config.partition, config.batch)
    test_results = list(ensemble_selection_results.find(
        {"meta_data": meta_data, "model": config.model, "batch": config.batch, "party_num": party_num, "ensemble": None,
         "parties": {"$size": 1  # Find the test results in the combination contains only one model
                     }}))
    assert len(test_results) == party_num + 1
    for index in range(len(test_results)):
        local_validation_accuracy = test_results[index]["local_validation_accuracy"]
        party = int(test_results[index]["parties"][0])
        if party != -1:  # skip -1 which is the oracle
            party_local_validation_accuracies.append(
                (party, local_validation_accuracy))

    party_local_validation_accuracies = sorted(party_local_validation_accuracies,
                                               key=lambda party_acc: party_acc[1], reverse=True)
    # print(party_local_validation_accuracies)
    indexes = []
    for i in range(K):
        indexes.append(party_local_validation_accuracies[i][0])
    # print(indexes)
    indexes.sort()
    # print(indexes)
    # funcName = sys._getframe().f_back.f_code.co_name # function name of callee
    funcName = sys._getframe().f_code.co_name  # current function name
    return indexes, funcName, party_local_validation_accuracies


def data_selection(config):
    K = config["K"]
    party_num = config["party_num"]
    total = 0
    party_dataset_amount = []
    pkl_file = open(
        DATASET_DIR + "files/" + config["dataset"] + "_" + config["split"] + "_" + str(partition) + "_" + str(party_num) + "_b" + str(
            config.batch) + ".pkl",
        'rb')
    data = pickle.load(pkl_file)
    for index in range(party_num):
        targets = data[index]["train_y"]
        party_dataset_amount.append((index, len(targets)))
        # print(index, len(targets))
        total += len(targets)

    party_dataset_amount = sorted(party_dataset_amount, key=lambda party_dataset_amount: party_dataset_amount[1],
                                  reverse=True)
    # print(total, total / 280000, party_dataset_amount)

    indexes = []
    for i in range(K):
        indexes.append(party_dataset_amount[i][0])
    # print(indexes)
    indexes.sort()
    # print(indexes)
    # funcName = sys._getframe().f_back.f_code.co_name
    funcName = sys._getframe().f_code.co_name
    return indexes, funcName, party_dataset_amount


def random_selection(config):
    K = config["K"]
    party_num = config["party_num"]

    random_indexes = list(random.sample(range(party_num - 1), K))
    random_indexes.sort()
    # print(random_indexes)
    # funcName = sys._getframe().f_back.f_code.co_name
    funcName = sys._getframe().f_code.co_name
    return random_indexes, funcName, None


def traverse_selection(config):
    selections = config.additional_parameters[0]
    i = config.additional_parameters[1]
    indexes = selections[i]
    funcName = sys._getframe().f_code.co_name
    print(indexes)
    # raise OSError("Traverse Selection")
    return indexes, funcName, None


def oracle(config):
    funcName = sys._getframe().f_code.co_name
    return [-1], funcName, None


def all_selection(config):
    party_num = config["party_num"]
    indexes = [i for i in range(party_num)]
    funcName = sys._getframe().f_code.co_name
    return indexes, funcName, None


def get_all_situations(length):
    if length == 1:
        return ["0", "1"]
    combination = get_all_situations(length - 1)
    new_combination = []
    for item in combination:
        new_combination.append(item + "0")
        new_combination.append(item + "1")
    return new_combination


if __name__ == '__main__':
    c = Config(exp_config)
    partitions = ["noniid-labeldir", "homo", "iid-diff-quantity",
                  get_noniid_label_number_split_name(c.split)]
    if c.partitions[-1] == "-1":
        c.partitions = partitions

    for partition in c.partitions:
        c.partition = partition
        print(c)
        # Traverse all situations
        if c.ALL:
            if c.K > 10:
                raise OSError("K is too large")
            all_situations = get_all_situations(c.party_num)
            print(all_situations)
            selections = []
            for i in range(len(all_situations)):
                situation = all_situations[i]
                selection = []
                k = 0
                for j in situation:
                    # print(j, selection)
                    if j == "1":
                        selection.append(k)
                    k += 1
                if len(selection) > 0:
                    selections.append(selection)
            print(len(selections), selections)
            for i in range(len(selections)):
                try:
                    c.additional_parameters = [selections, i]
                    repeat(traverse_selection, 1, config=c)
                    print("Traverse Selection", i, selections)
                except OSError as e:
                    print(e)
        else:
            # All selection
            repeat(all_selection, 1, config=c)

            # CV Selection
            repeat(CV_selection, 1, config=c)

            # Data Selection
            repeat(data_selection, 1, config=c)

            # Random Selection
            repeat(random_selection, repeat_times=10, config=c)

            # Oracle
            repeat(oracle, 1, config=c)
