# -*- coding: utf-8 -*-

import os
import pickle
import sys

import torch
import torch as torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np
import argparse

from model_config import get_model
from config import get_dataset_shape


def get_train_data_from_pkl(index, partition, party_num=20, dataset="emnist", split="default", batch=2):
    pkl_file = open("files/" + dataset + "_" + split + "_" + str(partition) + "_" + str(party_num) + "_b" + str(batch) + ".pkl", 'rb')
    data = pickle.load(pkl_file)
    feature = data[index]["train_X"]
    target = data[index]["train_y"]
    return feature, target

def get_test_data_from_pkl(index, partition, party_num=20, dataset="emnist", split="default", batch = 2):
    pkl_file = open("files/" + dataset + "_" + split + "_" + str(partition) + "_" + str(party_num) + "_b" + str(batch) + ".pkl", 'rb')
    data = pickle.load(pkl_file)
    feature = data[index]["test_X"]
    target = data[index]["test_y"]
    return feature, target

if __name__ == '__main__':

    num_epochs = 200
    batch_size_train = 128 # 100 for emnist, 128 for cifar10
    batch_size_test = 10
    log_interval = 500

    parser = argparse.ArgumentParser(description='index')
    parser.add_argument('--index', default=0, type=int, help='party index')
    parser.add_argument('--partition', default="iid-diff-quantity", type=str, help='partition methods')
    parser.add_argument('--device', default="cuda:7", type=str, help='partition methods')
    parser.add_argument('--dataset', default="cifar100", type=str, help='dataset name')
    parser.add_argument('--split', default="cifar100", type=str)
    parser.add_argument('--party_num', default=5, type=int)
    parser.add_argument('--batch', default=121, type=int, help='')
    parser.add_argument('--model', default="resnet50", type=str, help='model name')
    parser.add_argument('--input_channels', default=3, type=int, help='')
    parser.add_argument('--num_classes', default=100, type=int, help='')

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

    feature, target = get_train_data_from_pkl(args.index, args.partition, args.party_num, args.dataset, args.split, args.batch)
    transpose = (0, 3, 2, 1)
    if args.input_channels == 3:
        feature = np.transpose(feature,  transpose)
    feature = torch.Tensor(feature)
    target = torch.Tensor(target)

    test_feature, test_target = get_test_data_from_pkl(args.index, args.partition, args.party_num, args.dataset, args.split, args.batch)
    if args.input_channels == 3:
        test_feature = np.transpose(test_feature,  transpose)
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
            images = feature_train[begin:end].reshape(reshape_size, shape[0],  shape[1],  shape[2]).type(torch.FloatTensor).to(device)
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

        model.eval()
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

            best_accuracy = correct / total
            print('Training Accuracy {}: {} % Best: {} %'.format(args.model, 100 * correct / total,
                                                                           100 * best_accuracy))
            save_dir = "./models/%s_%s_%s_%s_b%d" %(args.dataset,args.split,args.model,args.partition, args.batch)
            best_epoch = epoch
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)

            model_path = save_dir + "/party_"+ str(args.index) + "_%d" % (
                args.party_num) + ".pkl"
            torch.save(model.state_dict(), model_path)
        scheduler.step()

    print("Test on local test set begin")

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
            images = test_feature[begin:end].reshape(reshape_size, shape[0],  shape[1],  shape[2]).type(torch.FloatTensor).to(device)
            labels = test_target[begin:end].to(device)

            outputs = model_final(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print("Test accuracy of best model on local test set: ", correct / total)
