import argparse
import calendar
import csv
import json
import sys
import time
from copy import deepcopy

import numpy as np
import pymongo
from sklearn.metrics import classification_report

from dbconfig import get_path
from dbconfig import ensemble_selection_results
from test_model import convert_report_to_json, generate_timestamp
from commandline_config import Config

DATASET_DIR, MODEL_DIR = get_path()
def max_voting(result_list, weights=None):
    if weights is None:
        weights = [1 for s in result_list]
    result_dict = dict()
    for i in range(len(weights)):
        weight = weights[i]
        outcome = result_list[i]
        if outcome in result_dict:
            result_dict[outcome] += weight
        else:
            result_dict[outcome] = weight
    # print(result_dict)
    max_value = max(result_dict.values())
    for key, value in result_dict.items():
        if value == max_value:
            return key

def main(config=None):
    if config == None:
        config = {
            "indexes": [0, 4, 6, 9, 15, 24, 25, 26, 30, 36, 38, 41, 47, 48, 49, 50, 54, 55, 56, 58, 60, 61, 71, 74, 75, 76, 82, 84, 85, 98],
            "partition": "noniid-#label3",
            "split": "digits",
            "dataset": "emnist",
            "party_num": 100,
        }
    c = Config(config)
    # print(c)
    from common_tools import get_dataset_amount
    dataset_amount = get_dataset_amount(c)
    # print(dataset_amount)
    meta_data = "%s_%s_%s_%s_b%d" % (c.dataset, c.split, c.model, c.partition, c.batch)
    save_dir = MODEL_DIR + "models/" + meta_data + "/predictions/"
    combine = []
    weights = []
    for index in c.indexes:
        weights.append(dataset_amount[index][1])
        if index == c.indexes[0]:
            flag_first = True
        else:
            flag_first = False
        i = 0
        tmp_file = save_dir + str(index) + "_" + str(c.party_num) + ".csv"
        file_exist = True
        try:
            t = open(tmp_file, "r")
        except:
            file_exist = False
        if not file_exist:
            meta_data = "%s_%s_%s_%s_b%d" % (c.dataset, c.split, c.model, c.partition, c.batch)
            save_dir = MODEL_DIR + "models/" + meta_data + "/predictions/"
            tmp_file = save_dir + str(index) + "_" + str(c.party_num) + ".csv"
        with open(tmp_file) as tmp_f:
            # print(tmp_file)
            tmp_f_csv = csv.reader(tmp_f)
            for row in tmp_f_csv:
                if flag_first:
                    combine.append([float(row[1]), float(row[0])])
                else:
                    combine[i].append(float(row[0]))
                    i += 1
    labels = []
    voting_results = []
    for i in range(len(combine)):
        dt = list(map(float, combine[i]))
        if c.weights == "equal":
            result = max_voting(dt[1:])
        elif c.weights == "data_size":
            result = max_voting(dt[1:], weights)
        if i == 0:
            # print("weigits", weights)
            pass
        voting_results.append(float(result))
        labels.append(combine[i][0])
    a = convert_report_to_json(classification_report(labels, voting_results, output_dict=True))
    accuracy = a["accuracy"]
    f1_score = a['macro avg']["f1-score"]
    # output_info = c.get_config()
    print("Accuracy:", accuracy, "F1-Score:", f1_score)

    ts = generate_timestamp()
    output_test_accuracy_file = MODEL_DIR + "models/" + meta_data + "/test_accuracy/%s.json" % ts
    with open(output_test_accuracy_file, "w") as f:
        output_info = {
            "timestamp":ts,
            "model": c.model,
            "parties": c.indexes,
            "ensemble": 1,
            "ID": str(c.indexes),
            "meta_data": meta_data,
            "party_num": c.party_num,
            "report": a,
            "batch":config.batch,
        }
        json.dump(output_info, f)
        ensemble_selection_results.insert_one(output_info)
    return a

if __name__ == '__main__':
    config = {
        "indexes": [0, 1, 2, 3, 4],
        "partition": "noniid-labeldir",
        "split": "digits",
        "dataset": "emnist",
        "party_num": 100,
    }
    indexes_list = [[0, 1, 2], [0], [2, 3, 11]]
    for indexes in indexes_list:
        config["indexes"] = indexes
        a = main(config)
