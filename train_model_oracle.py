# -*- coding: utf-8 -*-

import os
import pickle
import sys

import torch
import torchvision
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
import json
import argparse


from model_config import get_model

from config import get_dataset_shape
from test_model import get_whole_test_set
from prepare_data import get_train_data
from prepare_data import get_test_data
import matplotlib.pyplot as plt
SCRIPT_DIR = "/home/naibo/data/exps/SpinalNet/MNIST_VGG/"
print(SCRIPT_DIR)
sys.path.append(os.path.dirname(SCRIPT_DIR))
DATASET_DIR = "/home/naibo/data/datasets/"
def get_whole_training_set(index, partition, party_num=100, dataset="emnist", split="default", batch=10):
    train_file_name = DATASET_DIR + "files/" + dataset + "_" + split + "_" + str(partition) + "_" + str(party_num) + "_b" + str(batch) + "_whole_train.pkl"
    if os.path.exists(train_file_name):
        pkl_file = open(train_file_name, 'rb')
        data = pickle.load(pkl_file)
        features, targets = data["train_X"], data["train_y"]
    else:
        pkl_file = open(DATASET_DIR + "files/" + dataset + "_" + split + "_" + str(partition) + "_" + str(party_num) + "_b" + str(batch) + ".pkl", 'rb')
        data = pickle.load(pkl_file)
        features, targets = data[0]["train_X"], data[0]["train_y"]
        for i in range(1, party_num):
            feature = data[i]["train_X"]
            target = data[i]["train_y"]
            features = np.concatenate((features, feature), axis=0)
            targets = np.concatenate((targets, target))
        pickle.dump([{"train_X": features, "train_y":targets}], open(train_file_name, "wb"))

    return features, targets

def get_whole_validation_set(index, partition, party_num=100, dataset="emnist", split="default", batch=10):
    validation_file_name = DATASET_DIR + "files/" + dataset + "_" + split + "_" + str(partition) + "_" + str(
        party_num) + "_b" + str(batch) + "_whole_validation.pkl"
    if os.path.exists(validation_file_name):
        pkl_file = open(validation_file_name, 'rb')
        data = pickle.load(pkl_file)
        features, targets = data["train_X"], data["train_y"]
    else:
        pkl_file = open(DATASET_DIR + "files/" + dataset + "_" + split + "_" + str(partition) + "_" + str(party_num) + "_b" + str(batch) + ".pkl", 'rb')
        data = pickle.load(pkl_file)
        features, targets = data[0]["validation_X"], data[0]["validation_y"]
        for i in range(1, party_num):
            feature = data[i]["validation_X"]
            target = data[i]["validation_y"]
            features = np.concatenate((features, feature), axis=0)
            targets = np.concatenate((targets, target))
        pickle.dump([{"train_X": features, "train_y":targets}], open(validation_file_name, "wb"))

    return features, targets

def get_whole_test_set(index, partition, party_num=100, dataset="emnist", split="default", batch = 2):
    test_file_name = DATASET_DIR + "files/" + dataset + "_" + split + "_" + str(partition) + "_" + str(
        party_num) + "_b" + str(batch) + "_whole_test.pkl"
    if os.path.exists(test_file_name):
        pkl_file = open(test_file_name, 'rb')
        data = pickle.load(pkl_file)
        features, targets = data["train_X"], data["train_y"]
    else:
        pkl_file = open(DATASET_DIR + "files/" + dataset + "_" + split + "_" + str(partition) + "_" + str(party_num) + "_b" + str(batch) + ".pkl", 'rb')
        data = pickle.load(pkl_file)
        features, targets = data[0]["test_X"], data[0]["test_y"]
        for i in range(1, party_num):
            feature = data[i]["test_X"]
            target = data[i]["test_y"]
            features = np.concatenate((features, feature), axis=0)
            targets = np.concatenate((targets, target))
        pickle.dump([{"train_X": features, "train_y":targets}], open(test_file_name, "wb"))

    return features, targets

if __name__ == '__main__':

    num_epochs = 200
    batch_size_train = 128 # 100 for emnist, 128 for cifar10
    batch_size_test = 10
    log_interval = 500

    parser = argparse.ArgumentParser(description='index')
    parser.add_argument('--index', default=-1, type=int, help='party index')
    parser.add_argument('--partition', default="noniid-#label45", type=str, help='partition methods')
    parser.add_argument('--device', default="cuda:3", type=str, help='partition methods')
    parser.add_argument('--dataset', default="cifar100", type=str, help='dataset name')
    parser.add_argument('--split', default="cifar100", type=str)
    parser.add_argument('--party_num', default=5, type=int)
    parser.add_argument('--batch', default=121, type=int, help='')
    parser.add_argument('--input_channels', default=3, type=int, help='')
    parser.add_argument('--num_classes', default=100, type=int, help='')
    parser.add_argument('--model', default="resnet50", type=str, help='model name')
    args = parser.parse_args()
    device = args.device
    print("Args", args)

    # Train the model
    learning_rate = 0.1
    print("Learning Rate", learning_rate)
    model = get_model(args).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
                      momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    feature, target = get_whole_training_set(args.partition, args.party_num, args.dataset, args.split, args.batch)
    transpose = (0,3,2,1)
    print(transpose)
    if args.input_channels == 3:
        feature = np.transpose(feature, transpose)  # 0 1 2 3/0 1 3 2/0 2 1 3/0 2 3 1/0 3 1 2/ 0 3 2 1
    feature = torch.Tensor(feature)
    target = torch.Tensor(target)

    test_feature, test_target = get_whole_test_set(args.partition, args.party_num, args.dataset, args.split, args.batch)
    if args.input_channels == 3:
        test_feature = np.transpose(test_feature, transpose)
    test_feature = torch.Tensor(test_feature)
    test_target = torch.Tensor(test_target)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()

    # Train the model
    total_step = math.ceil(len(feature) / batch_size_train)

    best_accuracy = 0

    for epoch in range(num_epochs):
        model.train()
        idxs = np.random.permutation(len(target))  # 随机打乱
        feature_train = feature[idxs]
        target_train = target[idxs]
        for i in range(total_step):
            begin = i * batch_size_train
            end = (i + 1) * batch_size_train
            reshape_size = min(batch_size_train, len(feature_train[begin:end]))
            shape = get_dataset_shape(args.dataset)
            images = feature_train[begin:end].reshape(reshape_size, shape[0], shape[1], shape[2]).type(
                torch.FloatTensor).to(device)
            labels = target_train[begin:end].to(device)
            if labels.shape[0] == 1:
                continue

            outputs = model(images)
            loss = criterion(outputs, labels.long())

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

        model.eval()
        print("Test on local training set begin")
        with torch.no_grad():
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
                'Train Accuracy of {}: {} % Best: {} %'.format(args.model, 100 * correct / total,
                                                                           100 * best_accuracy))
            best_accuracy = correct / total
            save_dir = "./models/%s_%s_%s_%s_b%d" %(args.dataset,args.split,args.model,args.partition, args.batch)
            best_epoch = epoch
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)

            model_path = save_dir + "/party_"+ str(args.index) + "_%d" % (
                args.party_num) + ".pkl"
            torch.save(model.state_dict(), model_path)


        print("Test on local test set begin")

        with torch.no_grad():
            correct = 0
            total = 0

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
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print("Test accuracy of model on local test set of dataset: ", correct / total, args.dataset, args.split)
        scheduler.step()

    model_final = get_model(args)
    model_final.load_state_dict(torch.load(model_path))
    model_final = model_final.to(device)
    model_final.eval()
    with torch.no_grad():
        correct = 0
        total = 0

        for i in range(math.ceil(len(test_feature) / batch_size_test)):
            begin = i * batch_size_test
            end = (i + 1) * batch_size_test
            reshape_size = min(batch_size_test, len(test_feature[begin:end]))
            shape = get_dataset_shape(args.dataset)
            images = test_feature[begin:end].reshape(reshape_size, shape[0], shape[1], shape[2]).type(
                torch.FloatTensor).to(device)
            labels = test_target[begin:end].to(device)

            outputs = model_final(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f"Test accuracy of model {args.model} on local test set of dataset: ", correct / total, args.dataset, args.split)