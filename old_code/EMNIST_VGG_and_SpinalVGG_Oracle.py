# -*- coding: utf-8 -*-
"""
This Script contains the default and Spinal VGG code for EMNIST(Digits).

This code trains both NNs as two different models.

This code randomly changes the learning rate to get a good result.

@author: Dipu
"""
import os
import pickle

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
import matplotlib.pyplot as plt

def get_whole_training_set(partition, party_num=100, dataset="emnist", split="default", batch=10):
    pkl_file = open("files/" + dataset + "_" + split + "_" + str(partition) + "_" + str(party_num) + "_b" + str(batch) + ".pkl", 'rb')
    data = pickle.load(pkl_file)
    features, targets = data[0]["train_X"], data[0]["train_y"]
    for i in range(1, party_num):
        feature = data[i]["train_X"]
        target = data[i]["train_y"]
        features = np.concatenate((features, feature), axis=0)
        targets = np.concatenate((targets, target))

    return features, targets

def get_whole_validation_set(partition, party_num=100, dataset="emnist", split="default", batch=10):
    pkl_file = open("files/" + dataset + "_" + split + "_" + str(partition) + "_" + str(party_num) + "_b" + str(batch) + ".pkl", 'rb')
    data = pickle.load(pkl_file)
    features, targets = data[0]["validation_X"], data[0]["validation_y"]
    for i in range(1, party_num):
        feature = data[i]["validation_X"]
        target = data[i]["validation_y"]
        features = np.concatenate((features, feature), axis=0)
        targets = np.concatenate((targets, target))

    return features, targets


Half_width = 2048
layer_width = 128

if __name__ == '__main__':

    num_epochs = 200
    batch_size_train = 128 # 100 for emnist, 128 for cifar10
    batch_size_test = 10
    log_interval = 500

    parser = argparse.ArgumentParser(description='index')
    parser.add_argument('--index', default=-1, type=int, help='party index')
    parser.add_argument('--partition', default="noniid-#label45", type=str, help='partition methods')
    parser.add_argument('--device', default="cuda:2", type=str, help='partition methods')
    parser.add_argument('--dataset', default="cifar100", type=str, help='dataset name')
    parser.add_argument('--split', default="cifar100", type=str)
    parser.add_argument('--party_num', default=5, type=int)
    parser.add_argument('--batch', default=30, type=int, help='')
    parser.add_argument('--input_channels', default=3, type=int, help='')
    parser.add_argument('--num_classes', default=100, type=int, help='')
    parser.add_argument('--model', default="resnet50", type=str, help='model name')
    args = parser.parse_args()
    device = args.device
    print("Args", args)


    # For updating learning rate
    def update_lr(optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


    # Train the model
    # total_step = len(train_loader)
    # curr_lr1 = learning_rate

    learning_rate = 0.1
    print("Learning Rate", learning_rate)
    model2 = get_model(args).to(device)
    # optimizer2 = torch.optim.Adam(model2.parameters(), lr=learning_rate)
    optimizer2 = torch.optim.SGD(model2.parameters(), lr=learning_rate,
                      momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=200)
    curr_lr2 = learning_rate

    # model2 = SpinalCNN().to(device)
    feature, target = get_whole_training_set(args.partition, args.party_num, args.dataset, args.split, args.batch)
    transpose = (0,3,2,1)
    print(transpose)
    if args.input_channels == 3:
        feature = np.transpose(feature, transpose)  # 0 1 2 3/0 1 3 2/0 2 1 3/0 2 3 1/0 3 1 2/ 0 3 2 1
    # 0 1 2 3 Train Accuracy of SpinalNet: 66.42333333333333 % Best: 57.46 %
    # 0 1 3 2 Train Accuracy of SpinalNet: 87.28333333333333 % Best: 71.58666666666666 %
    # 0 2 3 1 Train Accuracy of SpinalNet: 94.62333333333333 % Best: 74.83999999999999 %
    # 0 3 2 1 Train Accuracy of SpinalNet: 99.90666666666667 % Best: 84.91333333333333 %
    # 0 2 1 3 Train Accuracy of SpinalNet: 58.79 % Best: 54.11333333333334 %
    # 0 3 1 2 -
    feature = torch.Tensor(feature)
    target = torch.Tensor(target)

    validation_feature, validation_target = get_whole_validation_set(args.partition, args.party_num, args.dataset, args.split, args.batch)
    if args.input_channels == 3:
        validation_feature = np.transpose(validation_feature,  transpose)
    validation_feature = torch.Tensor(validation_feature)
    validation_target = torch.Tensor(validation_target)

    test_feature, test_target = get_whole_test_set(-1, args.partition, args.party_num, args.dataset, args.split, args.batch)
    if args.input_channels == 3:
        test_feature = np.transpose(test_feature, transpose)
    test_feature = torch.Tensor(test_feature)
    test_target = torch.Tensor(test_target)
    # print(len(feature))
    # print(feature.shape)
    # print(target.shape)
    # print(type(feature))
    # print(type(target))
    # exit(0)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    # optimizer1 = torch.optim.Adam(model1.parameters(), lr=learning_rate)

    # Train the model
    # total_step = len(train_loader)
    total_step = math.ceil(len(feature) / batch_size_train)

    # best_accuracy1 = 0
    best_accuracy2 = 0

    for epoch in range(num_epochs):
        # model1.train()
        model2.train()
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

            outputs = model2(images)
            loss2 = criterion(outputs, labels.long())

            # Backward and optimize
            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()

            if i % 100 == 0:
                # print ("Ordinary Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                #        .format(epoch+1, num_epochs, i+1, total_step, loss1.item()))
                print("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss2.item()))

        # exit(0)
        # Test the model on validation set
        # model1.eval()
        model2.eval()
        with torch.no_grad():
            # correct1 = 0
            # total1 = 0
            correct2 = 0
            total2 = 0

            for i in range(math.ceil(len(feature) / batch_size_test)):
                begin = i * batch_size_test
                end = (i + 1) * batch_size_test
                reshape_size = min(batch_size_test, len(feature[begin:end]))
                shape = get_dataset_shape(args.dataset)
                images = feature[begin:end].reshape(reshape_size, shape[0], shape[1], shape[2]).type(
                    torch.FloatTensor).to(device)
                labels = target[begin:end].to(device)

                # outputs = model1(images)
                # _, predicted = torch.max(outputs.data, 1)
                # total1 += labels.size(0)
                # correct1 += (predicted == labels).sum().item()
                # print("prediction:",predicted)
                # print("labels:", labels)

                outputs = model2(images)
                _, predicted = torch.max(outputs.data, 1)
                total2 += labels.size(0)
                correct2 += (predicted == labels).sum().item()
            print(
                'Train Accuracy of {}: {} % Best: {} %'.format(args.model, 100 * correct2 / total2,
                                                                           100 * best_accuracy2))
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

                # outputs = model1(images)
                # _, predicted = torch.max(outputs.data, 1)
                # total1 += labels.size(0)
                # correct1 += (predicted == labels).sum().item()
                # print("prediction:",predicted)
                # print("labels:", labels)

                outputs = model2(images)
                _, predicted = torch.max(outputs.data, 1)
                total2 += labels.size(0)
                correct2 += (predicted == labels).sum().item()
                # print("prediction:",predicted)
                # print("labels:", labels)

            # if best_accuracy1>= correct1 / total1:
            #     curr_lr1 = learning_rate*np.asscalar(pow(np.random.rand(1),3))
            #     update_lr(optimizer1, curr_lr1)
            #     print('Test Accuracy of NN: {} % Best: {} %'.format(100 * correct1 / total1, 100*best_accuracy1))
            # else:
            #     best_accuracy1 = correct1 / total1
            #     net_opt1 = model1
            #     print('Test Accuracy of NN: {} % (improvement)'.format(100 * correct1 / total1))

            if best_accuracy2 >= correct2 / total2:
                # curr_lr2 = learning_rate * pow(np.random.rand(1), 3).item()
                # update_lr(optimizer2, curr_lr2)
                print(
                    'Validation Accuracy of {}: {} % Best: {} %'.format(args.model, 100 * correct2 / total2, 100 * best_accuracy2))
            else:
                best_accuracy2 = correct2 / total2
                net_opt2 = model2 # record the best model,这里存的是引用所以无法直接用等号copy
                print('Validation Accuracy of {}: {} % (improvement)'.format(args.model, 100 * correct2 / total2))
                save_dir = "./models/%s_%s_%s_%s_b%d" %(args.dataset,args.split,args.model,args.partition, args.batch)
                best_epoch = epoch
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)

                model2_path = save_dir + "/party_"+ str(args.index) + "_%d" % (
                    args.party_num) + ".pkl"
                torch.save(model2.state_dict(), model2_path)


        print("Test on local test set begin")

        # model_final = get_model(args)
        #
        # model_final.load_state_dict(torch.load(model2_path))
        # model_final = model_final.to(device)
        # model_final.eval()

        with torch.no_grad():

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

                outputs = model2(images)
                _, predicted = torch.max(outputs.data, 1)
                total2 += labels.size(0)
                correct2 += (predicted == labels).sum().item()
            print("Best Epoch:", best_epoch)
            print("Validation accuracy on local validation set: ", correct2 / total2)

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

                outputs = model2(images)
                _, predicted = torch.max(outputs.data, 1)
                total2 += labels.size(0)
                correct2 += (predicted == labels).sum().item()

            print("Test accuracy of model on local test set of dataset: ", correct2 / total2, args.dataset, args.split)
        # model1_path = "./models/model1_" + str(args.index) + ".pkl"
        # torch.save(model1, model1_path)
        scheduler.step()

    model_final = get_model(args)
    model_final.load_state_dict(torch.load(model2_path))
    model_final = model_final.to(device)
    model_final.eval()
    print("Best Epoch:", best_epoch)
    with torch.no_grad():

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

            outputs = model_final(images)
            _, predicted = torch.max(outputs.data, 1)
            total2 += labels.size(0)
            correct2 += (predicted == labels).sum().item()
        print("Epoch:", best_epoch)
        print("Best validation accuracy on local validation set: ", correct2 / total2)

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

            outputs = model_final(images)
            _, predicted = torch.max(outputs.data, 1)
            total2 += labels.size(0)
            correct2 += (predicted == labels).sum().item()

        print("Best test accuracy of Model on local test set of dataset: ", correct2 / total2, args.dataset, args.split)

    # model1_path = "./models/model1_" + str(args.index) + ".pkl"