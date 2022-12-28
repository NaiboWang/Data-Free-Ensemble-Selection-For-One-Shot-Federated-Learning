# -*- coding: utf-8 -*-
import os
import pickle
import calendar
import re
from copy import deepcopy

import pymongo
import torch
import torchvision
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
import json
import argparse
import time
from datetime import datetime, timedelta

from model_config import get_model
from sklearn.metrics import classification_report
from config import get_dataset_shape
from dbconfig import get_path
from dbconfig import ensemble_selection_results
import csv
from train_model import get_validation_data_from_pkl, get_test_data_from_pkl, get_whole_test_set, convert_dataset, \
    test_on_dataset, get_train_data_from_pkl


class TimeUtil(object):
    @classmethod
    def parse_timezone(cls, timezone):
        """
        :param timezone: str eg: +8
        :return: dict{symbol, offset}
        """
        result = re.match(r'(?P<symbol>[+-])(?P<offset>\d+)', timezone)
        symbol = result.groupdict()['symbol']
        offset = int(result.groupdict()['offset'])

        return {
            'symbol': symbol,
            'offset': offset
        }

    @classmethod
    def convert_timezone(cls, dt, timezone="+0"):
        result = cls.parse_timezone(timezone)
        symbol = result['symbol']

        offset = result['offset']

        if symbol == '+':
            return dt + timedelta(hours=offset)
        elif symbol == '-':
            return dt - timedelta(hours=offset)
        else:
            raise Exception('dont parse timezone format')


def generate_timestamp():
    current_GMT = time.gmtime()
    # ts stores timestamp
    ts = calendar.timegm(current_GMT)

    current_time = datetime.utcnow()
    convert_now = TimeUtil.convert_timezone(current_time, '+8')
    print("current_time:    " + str(convert_now))
    return str(convert_now)

def convert_report_to_json(a):
    b = deepcopy(a)
    for key in b:
        if key.find(".0") >= 0:
            a[key.replace(".0", "")] = a.pop(key)
    return a

DATASET_DIR, MODEL_DIR = get_path()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='index HHH')
    parser.add_argument('--index', default=3, type=int, help='index')
    parser.add_argument('--partition', default="noniid-#label45", type=str, help='partition')
    parser.add_argument('--split', default="cifar100", type=str)
    parser.add_argument('--device', default="cuda:4", type=str, help='partition methods')
    parser.add_argument('--dataset', default="cifar100", type=str, help='partition methods')
    parser.add_argument('--party_num', default=5, type=int, help='partition methods')
    parser.add_argument('--batch', default=221, type=int, help='')
    parser.add_argument('--input_channels', default=3, type=int, help='')
    parser.add_argument('--num_classes', default=100, type=int, help='')
    parser.add_argument('--save', default=1, type=int, help='if save the test result')
    parser.add_argument('--model', default="resnet50", type=str, help='model name')
    args = parser.parse_args()
    print("Args", args)
    device = args.device

    meta_data = "%s_%s_%s_%s_b%d" % (args.dataset, args.split,args.model, args.partition, args.batch)
    save_dir = MODEL_DIR + "models/" + meta_data
    model_path = save_dir + "/party_%d_%d.pkl" % (args.index, args.party_num)

    try:
        model = get_model(args)
        model.load_state_dict(torch.load(model_path, map_location=device)) # need to specific GPU device otherwise will cause error if one of the GPUs is full
    except:
        print("Change model class number to 62")
        model = get_model(args, 62)
        model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    if not os.path.exists(save_dir + "/predictions"):
        os.mkdir(save_dir + "/predictions")
    if not os.path.exists(save_dir + "/test_accuracy"):
        os.mkdir(save_dir + "/test_accuracy")
    output_prediction_file = save_dir + "/predictions/%d_%d.csv" % (args.index, args.party_num)
    ts = generate_timestamp()
    output_test_accuracy_file = save_dir + "/test_accuracy/%d_%s.json" % (args.party_num, ts)

    batch_size_test = 10

    feature, target = get_train_data_from_pkl(args.index, args.partition, args.party_num, args.dataset, args.split,
                                              args.batch)
    feature, target = convert_dataset(args, feature, target)

    validation_feature, validation_target = get_validation_data_from_pkl(args.index, args.partition, args.party_num,                                                                        args.dataset, args.split, args.batch)
    validation_feature, validation_target = convert_dataset(args, validation_feature, validation_target)

    test_feature, test_target = get_test_data_from_pkl(args.index, args.partition, args.party_num, args.dataset,
                                                       args.split, args.batch)
    test_feature, test_target = convert_dataset(args, test_feature, test_target)

    test_features, test_targets = get_whole_test_set(-1, args.partition, args.party_num, args.dataset, args.split, args.batch)
    test_features, test_targets = convert_dataset(args, test_features, test_targets)
    if args.index == -1:
        test_feature, test_target = test_features, test_targets

    # Test the model
    print("Test begin")
    with torch.no_grad():
        training_accuracy = test_on_dataset(args, feature, target, model, device, name="training")
        validation_accuracy = test_on_dataset(args, validation_feature, validation_target, model, device, name="validation")
        test_accuracy = test_on_dataset(args, test_feature, test_target, model, device, name="test")

        # Test who;e test accuracy
        correct = 0
        total = 0
        results = [[], []]

        for i in range(math.ceil(len(test_features) / batch_size_test)):
            begin = i * batch_size_test
            end = (i + 1) * batch_size_test
            reshape_size = min(batch_size_test, len(test_features[begin:end]))
            shape = get_dataset_shape(args.dataset)
            images = test_features[begin:end].reshape(reshape_size, shape[0], shape[1], shape[2]).type(
                torch.FloatTensor).to(device)
            labels = test_targets[begin:end].to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            results[0].extend(predicted.cpu().numpy().tolist())
            results[1].extend(labels.cpu().numpy().tolist())
        # print(correct,total, len(results[0]),len(results[1]))
        whole_test_accuracy = correct / total * 100
        print(f"Whole test accuracy: {round(whole_test_accuracy,4)}%")
        a = convert_report_to_json(
            classification_report(np.asarray(results)[1, :], np.asarray(results)[0, :], output_dict=True))

        # Save predictions
        with open(output_prediction_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for i in range(len(results[0])):
                writer.writerow([float(results[0][i]), float(results[1][i])])
        if args.save == 1:
            with open(output_test_accuracy_file, "w") as f:
                output_info = {
                    "timestamp":ts,
                    "batch": args.batch,
                    "model": args.model,
                    "parties": [args.index],
                    "ID": str([args.index]),
                    "meta_data": meta_data,
                    "party_num": args.party_num,
                    "local_training_accuracy": training_accuracy,
                    "local_validation_accuracy": validation_accuracy,
                    "local_test_accuracy": test_accuracy,
                    "whole_test_accuracy": whole_test_accuracy,
                    "report": a,
                }
                json.dump(output_info, f)
                ensemble_selection_results.insert_one(output_info)
