# -*- coding: utf-8 -*-
import os
import pickle

import torch
import torch as torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np
import argparse

from model_config import get_model
from config import get_dataset_shape
from dbconfig import *

DATASET_DIR, MODEL_DIR = get_path()


def get_train_data_from_pkl(index, partition, party_num=20, dataset="emnist", split="default", batch=2):
    pkl_file = open(DATASET_DIR + "files/" + dataset + "_" + split + "_" +
                    str(partition) + "_" + str(party_num) + "_b" + str(batch) + ".pkl", 'rb')
    data = pickle.load(pkl_file)
    feature = data[index]["train_X"]
    target = data[index]["train_y"]
    return feature, target


def get_validation_data_from_pkl(index, partition, party_num=20, dataset="emnist", split="default", batch=2):
    pkl_file = open(DATASET_DIR + "files/" + dataset + "_" + split + "_" +
                    str(partition) + "_" + str(party_num) + "_b" + str(batch) + ".pkl", 'rb')
    data = pickle.load(pkl_file)
    feature = data[index]["validation_X"]
    target = data[index]["validation_y"]
    return feature, target


def get_test_data_from_pkl(index, partition, party_num=20, dataset="emnist", split="default", batch=2):
    pkl_file = open(DATASET_DIR + "files/" + dataset + "_" + split + "_" +
                    str(partition) + "_" + str(party_num) + "_b" + str(batch) + ".pkl", 'rb')
    data = pickle.load(pkl_file)
    feature = data[index]["test_X"]
    target = data[index]["test_y"]
    return feature, target


def get_whole_training_set(index, partition, party_num=100, dataset="emnist", split="default", batch=10):
    pkl_file = open(DATASET_DIR + "files/" + dataset + "_" + split + "_" +
                    str(partition) + "_" + str(party_num) + "_b" + str(batch) + ".pkl", 'rb')
    data = pickle.load(pkl_file)
    features, targets = data[0]["train_X"], data[0]["train_y"]
    for i in range(1, party_num):
        feature = data[i]["train_X"]
        target = data[i]["train_y"]
        features = np.concatenate((features, feature), axis=0)
        targets = np.concatenate((targets, target))

    return features, targets


def get_whole_validation_set(index, partition, party_num=100, dataset="emnist", split="default", batch=10):
    pkl_file = open(DATASET_DIR + "files/" + dataset + "_" + split + "_" +
                    str(partition) + "_" + str(party_num) + "_b" + str(batch) + ".pkl", 'rb')
    data = pickle.load(pkl_file)
    features, targets = data[0]["validation_X"], data[0]["validation_y"]
    for i in range(1, party_num):
        feature = data[i]["validation_X"]
        target = data[i]["validation_y"]
        features = np.concatenate((features, feature), axis=0)
        targets = np.concatenate((targets, target))

    return features, targets


def get_whole_test_set(index, partition, party_num=100, dataset="emnist", split="default", batch=2):
    pkl_file = open(DATASET_DIR + "files/" + dataset + "_" + split + "_" +
                    str(partition) + "_" + str(party_num) + "_b" + str(batch) + ".pkl", 'rb')
    data = pickle.load(pkl_file)
    features, targets = data[0]["test_X"], data[0]["test_y"]
    for i in range(1, party_num):
        feature = data[i]["test_X"]
        target = data[i]["test_y"]
        features = np.concatenate((features, feature), axis=0)
        targets = np.concatenate((targets, target))

    return features, targets


def convert_dataset(args, feature, target):
    transpose = (0, 3, 2, 1)
    if args.input_channels == 3:
        feature = np.transpose(feature, transpose)
    feature = torch.Tensor(feature)
    target = torch.Tensor(target)
    return feature, target


def test_on_dataset(args, feature, target, model, device, name="test"):
    batch_size_test = 10
    correct = 0
    total = 0
    for i in range(math.ceil(len(feature) / batch_size_test)):
        begin = i * batch_size_test
        end = (i + 1) * batch_size_test
        reshape_size = min(batch_size_test, len(feature[begin:end]))
        shape = get_dataset_shape(args.dataset)
        images = feature[begin:end].reshape(reshape_size, shape[0], shape[1], shape[2]).type(
            torch.FloatTensor).to(device)
        labels = target[begin:end].to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print(
        'Test Accuracy of {} on {} set: {}%'.format(args.model, name, round(100 * correct / total, 4)))
    return 100 * correct / total


if __name__ == '__main__':

    num_epochs = 200
    batch_size_train = 128  # 100 for emnist, 128 for cifar10
    batch_size_test = 10
    log_interval = 500

    parser = argparse.ArgumentParser(description='index')
    parser.add_argument('--index', default=9, type=int, help='party index')
    parser.add_argument('--partition', default="iid-diff-quantity",
                        type=str, help='partition methods')
    parser.add_argument('--device', default="cuda:1",
                        type=str, help='partition methods')
    parser.add_argument('--dataset', default="emnist",
                        type=str, help='dataset name')
    parser.add_argument('--split', default="letters", type=str)
    parser.add_argument('--party_num', default=10, type=int)
    parser.add_argument('--batch', default=201, type=int, help='')
    parser.add_argument('--model', default="SpinalNet",
                        type=str, help='model name')
    parser.add_argument('--input_channels', default=1, type=int, help='')
    parser.add_argument('--num_classes', default=26, type=int, help='')

    args = parser.parse_args()
    device = args.device
    print("Args", args)

    # Train the model
    learning_rate = 0.1
    print("Learning Rate", learning_rate)
    model = get_model(args).to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
                                momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=200)

    if args.index != -1:
        print("Regular party")
        feature, target = get_train_data_from_pkl(
            args.index, args.partition, args.party_num, args.dataset, args.split, args.batch)
        feature, target = convert_dataset(args, feature, target)

        validation_feature, validation_target = get_validation_data_from_pkl(
            args.index, args.partition, args.party_num, args.dataset, args.split, args.batch)
        validation_feature, validation_target = convert_dataset(
            args, validation_feature, validation_target)

        test_feature, test_target = get_test_data_from_pkl(
            args.index, args.partition, args.party_num, args.dataset, args.split, args.batch)
        test_feature, test_target = convert_dataset(
            args, test_feature, test_target)
    else:
        print("Centralized party")
        feature, target = get_whole_training_set(
            args.index, args.partition, args.party_num, args.dataset, args.split, args.batch)
        feature, target = convert_dataset(args, feature, target)

        validation_feature, validation_target = get_whole_validation_set(
            args.index, args.partition, args.party_num, args.dataset, args.split, args.batch)
        validation_feature, validation_target = convert_dataset(
            args, validation_feature, validation_target)

        test_feature, test_target = get_whole_test_set(
            args.index, args.partition, args.party_num, args.dataset, args.split, args.batch)
        test_feature, test_target = convert_dataset(
            args, test_feature, test_target)
    test_whole_feature, test_whole_target = get_whole_test_set(
        args.index, args.partition, args.party_num, args.dataset, args.split, args.batch)
    test_whole_feature, test_whole_target = convert_dataset(
        args, test_whole_feature, test_whole_target)

    print("Training set", feature.shape, target.shape)
    print("Validation set", validation_feature.shape, validation_target.shape)
    print("Test set", test_feature.shape, test_target.shape)
    print("Whole test set", test_whole_feature.shape, test_whole_target.shape)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()

    # Train the model
    total_step = math.ceil(len(feature) / batch_size_train)
    best_accuracy = 0

    train_info = train_config.find({"index": args.index, "partition": args.partition, "party_num": args.party_num,
                                   "dataset": args.dataset, "split": args.split, "batch": args.batch, "model": args.model})
    if len(list(train_info)) == 0:
        train_config.insert_one({"index": args.index, "partition": args.partition, "party_num": args.party_num, "dataset": args.dataset, "split": args.split, "batch": args.batch, "model": args.model,
                                "train_accuracy": -1,  "validation_accuracy": -1, "test_accuracy": -1, "test_whole_accuracy": -1, "best_validation_accuracy": -1, "epoch": -1, "best_epoch": -1, "test_whole_accuracy_last_epoch": -1})
    else:
        train_config.update_one({"index": args.index, "partition": args.partition, "party_num": args.party_num, "dataset": args.dataset, "split": args.split, "batch": args.batch, "model": args.model}, {"$set": {
                                "train_accuracy": -1,  "validation_accuracy": -1, "test_accuracy": -1, "test_whole_accuracy": -1, "best_validation_accuracy": -1, "epoch": -1, "best_epoch": -1, "test_whole_accuracy_last_epoch": -1}})

    for epoch in range(num_epochs):
        model.train()
        idxs = np.random.permutation(len(target))
        feature_train = feature[idxs]
        target_train = target[idxs]
        for i in range(total_step):
            begin = i * batch_size_train
            end = (i + 1) * batch_size_train
            reshape_size = min(batch_size_train, len(feature_train[begin:end]))
            shape = get_dataset_shape(args.dataset)
            images = feature_train[begin:end].reshape(
                reshape_size, shape[0],  shape[1],  shape[2]).type(torch.FloatTensor).to(device)
            labels = target_train[begin:end].to(device)
            if labels.shape[0] == 1:
                continue

            outputs = model(images)
            loss = criterion(outputs, labels.long())

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

        train_config.update_one({"index": args.index, "partition": args.partition, "party_num": args.party_num,
                                "dataset": args.dataset, "split": args.split, "batch": args.batch, "model": args.model}, {"$set": {"epoch": epoch}})

        # Test the model on validation set
        model.eval()
        with torch.no_grad():
            train_accuracy = test_on_dataset(
                args, feature, target, model, device, name="training")
            validation_accuracy = test_on_dataset(
                args, validation_feature, validation_target, model, device, name="validation")
            if best_accuracy < validation_accuracy:
                best_accuracy = validation_accuracy
                print('Validation Accuracy of {}: {} % (improvement)'.format(
                    args.model, validation_accuracy))
                save_dir = MODEL_DIR + "models/%s_%s_%s_%s_b%d" % (
                    args.dataset, args.split, args.model, args.partition, args.batch)
                best_epoch = epoch
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                model_path = save_dir + "/party_" + str(args.index) + "_%d" % (
                    args.party_num) + ".pkl"
                torch.save(model.state_dict(), model_path)
                train_config.update_one({"index": args.index, "partition": args.partition, "party_num": args.party_num, "dataset": args.dataset, "split": args.split,
                                        "batch": args.batch, "model": args.model}, {"$set": {"best_epoch": best_epoch, "best_validation_accuracy": best_accuracy}})

            test_accuracy = test_on_dataset(
                args, test_feature, test_target, model, device, name="test")
            train_config.update_one({"index": args.index, "partition": args.partition, "party_num": args.party_num, "dataset": args.dataset, "split": args.split, "batch": args.batch, "model": args.model}, {
                                    "$set": {"train_accuracy": train_accuracy, "validation_accuracy": validation_accuracy, "test_accuracy": test_accuracy}})
            print(args)
        scheduler.step()

    # Test the model on whole test set
    model.eval()
    with torch.no_grad():
        test_whole_accuracy = test_on_dataset(
            args, test_whole_feature, test_whole_target, model, device, name="whole test")
        train_config.update_one({"index": args.index, "partition": args.partition, "party_num": args.party_num, "dataset": args.dataset,
                                "split": args.split, "batch": args.batch, "model": args.model}, {"$set": {"test_whole_accuracy_last_epoch": test_whole_accuracy}})
        print(args)
    print("Test on local test set begin")

    model_final = get_model(args)
    model_final.load_state_dict(torch.load(model_path))
    model_final = model_final.to(device)
    model_final.eval()

    with torch.no_grad():
        print("Best epoch:", best_epoch)
        test_on_dataset(args, validation_feature, validation_target,
                        model_final, device, name="validation")
        test_on_dataset(args, test_feature, test_target,
                        model_final, device, name="test")
        whole_test_accuracy = test_on_dataset(
            args, test_whole_feature, test_whole_target, model_final, device, name="test_whole")
        train_config.update_one({"index": args.index, "partition": args.partition, "party_num": args.party_num, "dataset": args.dataset,
                                "split": args.split, "batch": args.batch, "model": args.model}, {"$set": {"test_whole_accuracy": whole_test_accuracy}})
        print(args)
