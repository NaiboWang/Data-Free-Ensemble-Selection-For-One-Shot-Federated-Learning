# -*- coding: utf-8 -*-
"""
This Script contains the default and Spinal VGG code for EMNIST(Digits).

This code trains both NNs as two different models.

This code randomly changes the learning rate to get a good result.

@author: Dipu
"""
import os

import torch
import math
import numpy as np
import json
import argparse

from model_config import get_model
from sklearn.metrics import classification_report

from config import get_dataset_shape
from dbconfig import ensemble_selection_results
import csv

from test_model import get_whole_test_set, convert_report_to_json, generate_timestamp
from old_code.EMNIST_VGG_and_SpinalVGG_Oracle import get_whole_validation_set


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='index HHH')
    parser.add_argument('--index', default=-1, type=int, help='index')
    parser.add_argument('--partition', default="homo", type=str, help='partition')
    parser.add_argument('--split', default="cifar100", type=str)
    parser.add_argument('--device', default="cuda:4", type=str, help='partition methods')
    parser.add_argument('--dataset', default="cifar100", type=str, help='partition methods')
    parser.add_argument('--party_num', default=10, type=int, help='partition methods')
    parser.add_argument('--batch', default=20, type=int, help='')
    parser.add_argument('--input_channels', default=3, type=int, help='')
    parser.add_argument('--num_classes', default=100, type=int, help='')
    parser.add_argument('--save', default=1, type=int, help='')
    parser.add_argument('--model', default="efficientnet-b7", type=str, help='model name')
    args = parser.parse_args()
    print("Args", args)
    device = args.device

    meta_data = "%s_%s_%s_%s_b%d" % (args.dataset, args.split, args.model, args.partition, args.batch)
    save_dir = "./models/" + meta_data
    model_path = save_dir + "/party_%d_%d.pkl" % (args.index, args.party_num)
    model = get_model(args)
    model.load_state_dict(torch.load(model_path))
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
    transpose = (0, 3, 2, 1)
    validation_feature, validation_target = get_whole_validation_set(args.partition, args.party_num,
                                              args.dataset, args.split, args.batch)
    if args.input_channels == 3:
        validation_feature = np.transpose(validation_feature,  transpose)
    validation_feature = torch.Tensor(validation_feature)
    validation_target = torch.Tensor(validation_target)

    test_feature, test_target = get_whole_test_set(-1, args.partition, args.party_num, args.dataset, args.split, args.batch)
    if args.input_channels == 3:
        test_feature = np.transpose(test_feature,  transpose)
    test_feature = torch.Tensor(test_feature)
    test_target = torch.Tensor(test_target)

    test_features, test_targets = get_whole_test_set(-1, args.partition, args.party_num, args.dataset, args.split, args.batch)
    if args.input_channels == 3:
        test_features = np.transpose(test_features,  transpose)
    test_features = torch.Tensor(test_features)
    test_targets = torch.Tensor(test_targets)

    # Test the model
    print("Test begin")
    with torch.no_grad():
        # Test validation accuracy
        correct2 = 0
        total2 = 0

        for i in range(math.ceil(len(validation_feature) / batch_size_test)):
            begin = i * batch_size_test
            end = (i + 1) * batch_size_test
            reshape_size = min(batch_size_test, len(validation_feature[begin:end]))
            shape = get_dataset_shape(args.dataset)
            images = validation_feature[begin:end].reshape(reshape_size, shape[0], shape[1], shape[2]).type(
                torch.FloatTensor).to(device)
            labels = validation_target[begin:end].to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total2 += labels.size(0)
            correct2 += (predicted == labels).sum().item()
        validation_accuracy = correct2 / total2
        print("Validation accuracy:", validation_accuracy)

        # Test local test accuracy
        correct2 = 0
        total2 = 0

        for i in range(math.ceil(len(test_feature) / batch_size_test)):
            begin = i * batch_size_test
            end = (i + 1) * batch_size_test
            reshape_size = min(batch_size_test, len(test_feature[begin:end]))
            shape = get_dataset_shape(args.dataset)
            images = test_feature[begin:end].reshape(reshape_size, shape[0], shape[1], shape[2]).type(
                torch.FloatTensor).to(device)
            labels = test_target[begin:end].to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total2 += labels.size(0)
            correct2 += (predicted == labels).sum().item()
        local_test_accuracy = correct2 / total2
        print("Local test accuracy:", local_test_accuracy)

        # Test who;e test accuracy
        correct2 = 0
        total2 = 0
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
            total2 += labels.size(0)
            correct2 += (predicted == labels).sum().item()
            results[0].extend(predicted.cpu().numpy().tolist())
            results[1].extend(labels.cpu().numpy().tolist())
        # print(correct2,total2, len(results[0]),len(results[1]))
        whole_test_accuracy = correct2 / total2
        print("Whole test accuracy:", whole_test_accuracy)
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
                    "batch":args.batch,
                    "model":args.model,
                    "parties": [args.index],
                    "ID": str([args.index]),
                    "meta_data": meta_data,
                    "party_num": args.party_num,
                    "local_validation_accuracy": validation_accuracy,
                    "local_test_accuracy": local_test_accuracy,
                    "whole_test_accuracy": whole_test_accuracy,
                    "report": a,
                }
                json.dump(output_info, f)
                ensemble_selection_results.insert_one(output_info)
