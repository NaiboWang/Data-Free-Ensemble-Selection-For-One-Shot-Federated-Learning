import copy
import csv
import json
import os
import pickle
import random
import time

import math
import numpy as np
import torch
from commandline_config import Config
from prettytable import PrettyTable
from sklearn.decomposition import PCA, KernelPCA
from sklearn.metrics import classification_report, cohen_kappa_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
from model_config import get_model
from config import get_label_numbers, exp_config
from dbconfig import get_path
from dbconfig import ensemble_selection_exp_results, ensemble_selection_results
from ensemble import main
from test_model import generate_timestamp, get_whole_test_set, convert_report_to_json

DATASET_DIR, MODEL_DIR = get_path()

def repeat(selection_method=None, repeat_times=50, config=None, qualifier=""):
    start_time = time.time()
    average_accuracy = 0.0
    average_fscore = 0.0
    reports = []
    for i in range(repeat_times):  # repeat repeat_times times
        indexes, funcName, additional_information = selection_method(config)
        print("indexes", indexes)
        config["indexes"] = indexes
        if config.avg == "none": # if not conduct model averaging:
            report = main(config)
        else:
            report = model_averging(config)
        average_accuracy += report["accuracy"]
        average_fscore += report['macro avg']["f1-score"]
        reports.append({"indexes": indexes,"additional_information":additional_information, "report": report})
    print(funcName)
    average_accuracy /= repeat_times
    average_fscore /= repeat_times
    config["indexes"] = []
    # print(config)
    print("Average accuracy, f1-score for %s:" % funcName, average_accuracy, average_fscore)
    print("")
    end_time = time.time()
    ts = generate_timestamp()
    if config.avg != "none":
        qualifier = config.avg + "_" + qualifier
    output_info = {
        "batch": config["batch_ensemble"],
        "timestamp": ts,
        "method": qualifier + funcName,
        "config": config,
        "device": "cpu",
        "duration":[start_time, end_time],
        "repeat_times": repeat_times,
        "average_accuracy": average_accuracy,
        "average_fscore": average_fscore,
        "reports": reports,
    }
    with open(MODEL_DIR + "exp_results/data/%s.json" % ts, "w") as f:
        ensemble_selection_exp_results.insert_one(output_info)
        json.dump(output_info, f)

def read_distribution(config):
    party_num = config["party_num"]
    party_dataset_distribution = []
    pkl_file = open(
        DATASET_DIR + "files/" + config["dataset"] + "_" + config["split"] + "_" + str(config["partition"]) + "_" + str(
            party_num) + "_b" + str(config.batch) + ".pkl",
        'rb')
    data = pickle.load(pkl_file)
    num_labels = config.num_classes
    # t = []
    # for i in range(num_labels):
    #     t.append(i)
    # output = PrettyTable(t)
    indexes = get_outliers(config)
    filtered_indexes = []
    for index in range(party_num):
        if index not in indexes:
            filtered_indexes.append(index)
            targets = data[index]["train_y"]
            # print(targets)

            weight = []
            for i in range(num_labels):
                weight.append(0)
            for i in range(len(targets)):
                weight[targets[i]] += 1
            # print(weight)
            party_dataset_distribution.append(weight)
        # for key in party_dataset_distribution[-1]:
        #     t2 = []
        #     for i in range(47):
        #         t2.append(party_dataset_distribution[-1][i])
        #     # output += key + ":" + str(self[key]) + "\n"
        #     output.add_row(t2)
    scaler = StandardScaler()
    # TODO: Explain Why we want to use standard scaler, and why model parameters use minmax scaler
    party_dataset_distribution = scaler.fit_transform(party_dataset_distribution)
    return party_dataset_distribution, filtered_indexes

def box_plot_outliers(s):
    q1, q3 = s.quantile(.1), s.quantile(.75)
    iqr = q3 - q1
    print(iqr)
    low = q1 - 1.5 * iqr
    outlier = s.mask(s<low)
    indexes = np.where(pd.isna(np.array(outlier)))
    return outlier, indexes


def get_outliers(config):
    # This part is used to filter out models that are considered as low outliers
    meta_data = "%s_%s_%s_%s_b%d" % (config.dataset, config.split, config.model, config.partition, config.batch)
    test_results = list(ensemble_selection_results.find(
        {"meta_data": meta_data, "model": config.model, "batch": config.batch, "party_num": config.party_num,
         "ensemble": None,
         "parties": {"$size": 1  # Find the test results in the combination contains only one model
                     }})) 
    party_local_validation_accuracies = []
    for index in range(len(test_results)):
        local_validation_accuracy = test_results[index]["local_validation_accuracy"]
        party = int(test_results[index]["parties"][0])
        if party != -1:  # skip -1 which is the oracle
            party_local_validation_accuracies.append((party, local_validation_accuracy))
    party_local_validation_accuracies.sort(key=lambda x: x[0])
    assert len(party_local_validation_accuracies) == config.party_num
    vaccs = []
    for (i, acc) in party_local_validation_accuracies:
        vaccs.append(acc)
    df = pd.DataFrame(vaccs)
    outliers = df.apply(box_plot_outliers)
    indexes = outliers[0][1][0]
    return indexes


def read_parameters(config, flatten = True):
    all_weights, flatten_weights = [], []
    party_num = config["party_num"]
    meta_data = "%s_%s_%s_%s_b%d" % (config["dataset"], config["split"], config.model, config["partition"], config["batch"])
    save_dir = MODEL_DIR + "models/" + meta_data
    if not os.path.exists(DATASET_DIR + "files/cluster_preprocess"):
        os.mkdir(DATASET_DIR + "files/cluster_preprocess")
    if config["normalization"] == 1:
        file_name = meta_data + f"_be{config.batch_ensemble}_" + str(party_num) + "_" + config.dr_method[0]
    else:
        file_name = meta_data + f"_be{config.batch_ensemble}_" + str(party_num) + "_" + config.dr_method[0] + "_noNormalization"
    if config.last_layer:
        file_name += "_lastLayer"
    if config.layer != 0:
        file_name += "_layer%d" % config.layer
    if not flatten:
        file_name += "_original"
    file_name += ".pkl"
    if not os.path.exists(DATASET_DIR + "files/cluster_preprocess/"+file_name):
        indexes = get_outliers(config)
        for index in range(party_num):
            model_path = save_dir + "/party_%d_%d.pkl" % (index, party_num)
            # print(model_path)
            config_c = copy.deepcopy(config)
            config_c.device = "cpu"
            try:
                model = get_model(config_c)
                model.load_state_dict(torch.load(model_path, map_location="cpu"))
            except:
                print("Change model class number to 62")
                model = get_model(config_c, 62)
                model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu'))) # ensure models are loaded via cpu and main memory, not gpu cuda memory
            weights = model.state_dict()
            all_weights.append((index,copy.deepcopy(weights)))
        filtered_weights = []
        filtered_indexes = []
        for i in range(len(all_weights)):
            index = all_weights[i][0]
            if index not in indexes:
                filtered_weights.append(all_weights[i])
                filtered_indexes.append(index)
        # for key in model.state_dict().keys():
        #     print(key)
        if flatten:
            all_keys = filtered_weights[0][1].keys()
            random_indexes = list(random.sample(range(len(all_keys) - 1), min(len(filtered_weights), int(len(all_keys) * 0.1))))
            random_indexes.sort()
            random_keys = list(np.asarray(list(all_keys))[random_indexes])
            if config.layer == -1:
                print("random keys:", random_keys)
            for i in range(len(filtered_weights)):
                index = filtered_weights[i][0]
                weights = []
                for key in filtered_weights[i][1].keys():
                    if config.last_layer:
                        if config.model == "SpinalNet":
                            layer_name =  ["fc_out.1.weight","fc_out.1.bias"]
                        elif config.model == "effenetv2_l":
                            layer_name = ['classifier.weight', 'classifier.bias']
                        # elif config.model == "efficientnet-b7":
                        #     layer_name = ['_fc.weight', '_fc.bias']
                        else:
                            layer_name = ['linear.weight', 'linear.bias']
                        if key in layer_name:
                            weight = filtered_weights[i][1][key].cpu().numpy().flatten()
                            weights.extend(weight)
                    else:
                        if config.layer == 0: # all layers
                            weight = filtered_weights[i][1][key].cpu().numpy().flatten()
                            weights.extend(weight)
                        elif config.layer == -1: # random layers
                            if key in random_keys:
                                weight = filtered_weights[i][1][key].cpu().numpy().flatten()
                                weights.extend(weight)
                        else: # TODO: Automatically generate model layer names
                            if config.model == "SpinalNet":
                                layer_name = [['l1.0.weight', 'l1.0.bias', 'l1.1.weight', 'l1.1.bias', 'l1.1.running_mean', 'l1.1.running_var', 'l1.1.num_batches_tracked', 'l1.3.weight', 'l1.3.bias', 'l1.4.weight', 'l1.4.bias', 'l1.4.running_mean', 'l1.4.running_var', 'l1.4.num_batches_tracked'],['fc_spinal_layer1.1.weight', 'fc_spinal_layer1.1.bias', 'fc_spinal_layer1.2.weight', 'fc_spinal_layer1.2.bias', 'fc_spinal_layer1.2.running_mean', 'fc_spinal_layer1.2.running_var', 'fc_spinal_layer1.2.num_batches_tracked'],['fc_spinal_layer4.1.weight', 'fc_spinal_layer4.1.bias', 'fc_spinal_layer4.2.weight', 'fc_spinal_layer4.2.bias', 'fc_spinal_layer4.2.running_mean', 'fc_spinal_layer4.2.running_var', 'fc_spinal_layer4.2.num_batches_tracked']]
                            elif config.model == "densenet":
                                layer_name = [['conv1.weight', 'dense1.0.bn1.weight', 'dense1.0.bn1.bias', 'dense1.0.bn1.running_mean', 'dense1.0.bn1.running_var', 'dense1.0.bn1.num_batches_tracked', 'dense1.0.conv1.weight', 'dense1.0.bn2.weight', 'dense1.0.bn2.bias', 'dense1.0.bn2.running_mean', 'dense1.0.bn2.running_var', 'dense1.0.bn2.num_batches_tracked', 'dense1.0.conv2.weight'],['dense2.8.bn1.weight', 'dense2.8.bn1.bias', 'dense2.8.bn1.running_mean', 'dense2.8.bn1.running_var', 'dense2.8.bn1.num_batches_tracked', 'dense2.8.conv1.weight', 'dense2.8.bn2.weight', 'dense2.8.bn2.bias', 'dense2.8.bn2.running_mean', 'dense2.8.bn2.running_var', 'dense2.8.bn2.num_batches_tracked', 'dense2.8.conv2.weight'], ['dense4.13.bn1.weight', 'dense4.13.bn1.bias', 'dense4.13.bn1.running_mean', 'dense4.13.bn1.running_var', 'dense4.13.bn1.num_batches_tracked', 'dense4.13.conv1.weight', 'dense4.13.bn2.weight', 'dense4.13.bn2.bias', 'dense4.13.bn2.running_mean', 'dense4.13.bn2.running_var', 'dense4.13.bn2.num_batches_tracked', 'dense4.13.conv2.weight']]
                            elif config.model == "resnet50":
                                layer_name = [['conv1.weight', 'bn1.weight', 'bn1.bias', 'bn1.running_mean', 'bn1.running_var', 'bn1.num_batches_tracked', 'layer1.0.conv1.weight', 'layer1.0.bn1.weight', 'layer1.0.bn1.bias', 'layer1.0.bn1.running_mean', 'layer1.0.bn1.running_var', 'layer1.0.bn1.num_batches_tracked', 'layer1.0.conv2.weight', 'layer1.0.bn2.weight', 'layer1.0.bn2.bias', 'layer1.0.bn2.running_mean', 'layer1.0.bn2.running_var', 'layer1.0.bn2.num_batches_tracked', 'layer1.0.conv3.weight', 'layer1.0.bn3.weight', 'layer1.0.bn3.bias', 'layer1.0.bn3.running_mean', 'layer1.0.bn3.running_var', 'layer1.0.bn3.num_batches_tracked', 'layer1.0.shortcut.0.weight', 'layer1.0.shortcut.1.weight', 'layer1.0.shortcut.1.bias', 'layer1.0.shortcut.1.running_mean', 'layer1.0.shortcut.1.running_var', 'layer1.0.shortcut.1.num_batches_tracked'],['layer3.0.conv1.weight', 'layer3.0.bn1.weight', 'layer3.0.bn1.bias', 'layer3.0.bn1.running_mean', 'layer3.0.bn1.running_var', 'layer3.0.bn1.num_batches_tracked', 'layer3.0.conv2.weight', 'layer3.0.bn2.weight', 'layer3.0.bn2.bias', 'layer3.0.bn2.running_mean', 'layer3.0.bn2.running_var', 'layer3.0.bn2.num_batches_tracked', 'layer3.0.conv3.weight', 'layer3.0.bn3.weight', 'layer3.0.bn3.bias', 'layer3.0.bn3.running_mean', 'layer3.0.bn3.running_var', 'layer3.0.bn3.num_batches_tracked', 'layer3.0.shortcut.0.weight', 'layer3.0.shortcut.1.weight', 'layer3.0.shortcut.1.bias', 'layer3.0.shortcut.1.running_mean', 'layer3.0.shortcut.1.running_var', 'layer3.0.shortcut.1.num_batches_tracked'], ['layer4.2.conv1.weight', 'layer4.2.bn1.weight', 'layer4.2.bn1.bias', 'layer4.2.bn1.running_mean', 'layer4.2.bn1.running_var', 'layer4.2.bn1.num_batches_tracked', 'layer4.2.conv2.weight', 'layer4.2.bn2.weight', 'layer4.2.bn2.bias', 'layer4.2.bn2.running_mean', 'layer4.2.bn2.running_var', 'layer4.2.bn2.num_batches_tracked', 'layer4.2.conv3.weight', 'layer4.2.bn3.weight', 'layer4.2.bn3.bias', 'layer4.2.bn3.running_mean', 'layer4.2.bn3.running_var', 'layer4.2.bn3.num_batches_tracked']]
                            elif config.model == "dla":
                                layer_name = [['base.0.weight', 'base.1.weight', 'base.1.bias', 'base.1.running_mean', 'base.1.running_var', 'base.1.num_batches_tracked', 'layer1.0.weight', 'layer1.1.weight', 'layer1.1.bias', 'layer1.1.running_mean', 'layer1.1.running_var', 'layer1.1.num_batches_tracked', 'layer2.0.weight', 'layer2.1.weight', 'layer2.1.bias', 'layer2.1.running_mean', 'layer2.1.running_var', 'layer2.1.num_batches_tracked'],['layer5.root.conv.weight', 'layer5.root.bn.weight', 'layer5.root.bn.bias', 'layer5.root.bn.running_mean', 'layer5.root.bn.running_var', 'layer5.root.bn.num_batches_tracked', 'layer5.level_1.root.conv.weight', 'layer5.level_1.root.bn.weight', 'layer5.level_1.root.bn.bias', 'layer5.level_1.root.bn.running_mean', 'layer5.level_1.root.bn.running_var', 'layer5.level_1.root.bn.num_batches_tracked', 'layer5.level_1.left_node.conv1.weight', 'layer5.level_1.left_node.bn1.weight', 'layer5.level_1.left_node.bn1.bias', 'layer5.level_1.left_node.bn1.running_mean', 'layer5.level_1.left_node.bn1.running_var', 'layer5.level_1.left_node.bn1.num_batches_tracked', 'layer5.level_1.left_node.conv2.weight', 'layer5.level_1.left_node.bn2.weight', 'layer5.level_1.left_node.bn2.bias', 'layer5.level_1.left_node.bn2.running_mean', 'layer5.level_1.left_node.bn2.running_var', 'layer5.level_1.left_node.bn2.num_batches_tracked', 'layer5.level_1.left_node.shortcut.0.weight', 'layer5.level_1.left_node.shortcut.1.weight', 'layer5.level_1.left_node.shortcut.1.bias', 'layer5.level_1.left_node.shortcut.1.running_mean', 'layer5.level_1.left_node.shortcut.1.running_var', 'layer5.level_1.left_node.shortcut.1.num_batches_tracked', 'layer5.level_1.right_node.conv1.weight', 'layer5.level_1.right_node.bn1.weight', 'layer5.level_1.right_node.bn1.bias', 'layer5.level_1.right_node.bn1.running_mean', 'layer5.level_1.right_node.bn1.running_var', 'layer5.level_1.right_node.bn1.num_batches_tracked', 'layer5.level_1.right_node.conv2.weight', 'layer5.level_1.right_node.bn2.weight', 'layer5.level_1.right_node.bn2.bias', 'layer5.level_1.right_node.bn2.running_mean', 'layer5.level_1.right_node.bn2.running_var', 'layer5.level_1.right_node.bn2.num_batches_tracked'], ['layer6.root.bn.bias', 'layer6.root.bn.running_mean', 'layer6.root.bn.running_var', 'layer6.root.bn.num_batches_tracked', 'layer6.left_node.conv1.weight', 'layer6.left_node.bn1.weight', 'layer6.left_node.bn1.bias', 'layer6.left_node.bn1.running_mean', 'layer6.left_node.bn1.running_var', 'layer6.left_node.bn1.num_batches_tracked', 'layer6.left_node.conv2.weight', 'layer6.left_node.bn2.weight', 'layer6.left_node.bn2.bias', 'layer6.left_node.bn2.running_mean', 'layer6.left_node.bn2.running_var', 'layer6.left_node.bn2.num_batches_tracked', 'layer6.left_node.shortcut.0.weight', 'layer6.left_node.shortcut.1.weight', 'layer6.left_node.shortcut.1.bias', 'layer6.left_node.shortcut.1.running_mean', 'layer6.left_node.shortcut.1.running_var', 'layer6.left_node.shortcut.1.num_batches_tracked', 'layer6.right_node.conv1.weight', 'layer6.right_node.bn1.weight', 'layer6.right_node.bn1.bias', 'layer6.right_node.bn1.running_mean', 'layer6.right_node.bn1.running_var', 'layer6.right_node.bn1.num_batches_tracked', 'layer6.right_node.conv2.weight', 'layer6.right_node.bn2.weight', 'layer6.right_node.bn2.bias', 'layer6.right_node.bn2.running_mean', 'layer6.right_node.bn2.running_var', 'layer6.right_node.bn2.num_batches_tracked']]
                            if key in layer_name[config.layer-1]:
                                weight = filtered_weights[i][1][key].cpu().numpy().flatten()
                                weights.extend(weight)
                flatten_weights.append(weights)
            # n_components = int(config.dr_method[1] * len(flatten_weights[0]))
            if config.dr_method[0] == "PCA":
                pca = PCA(n_components=len(filtered_weights))
                flatten_weights = pca.fit(flatten_weights).transform(flatten_weights)
            elif config.dr_method[0] == "Kernel_PCA":
                kernel_pca = KernelPCA(n_components=len(filtered_weights))
                # c = kernel_pca.fit(flatten_weights)
                # d = c.transform(flatten_weights)
                flatten_weights = kernel_pca.fit(flatten_weights).transform(flatten_weights)

            if config["normalization"] == 1:
                mm = MinMaxScaler()
                flatten_weights = mm.fit_transform(flatten_weights)
            with open(DATASET_DIR + "files/cluster_preprocess/" + file_name, 'wb') as fid:
                pickle.dump(flatten_weights, fid)
            with open(DATASET_DIR + "files/cluster_preprocess/indexes_" + file_name, 'wb') as fid:
                pickle.dump(filtered_indexes, fid)
            print("Min Max Transformed and saved to %s." % file_name)
            return flatten_weights, filtered_indexes
        else:
            with open(DATASET_DIR + "files/cluster_preprocess/" + file_name, 'wb') as fid:
                pickle.dump(all_weights, fid)
            return all_weights
    else:
        pkl_file = open(DATASET_DIR + "files/cluster_preprocess/" + file_name, 'rb')
        weights = pickle.load(pkl_file)
        pkl_file = open(DATASET_DIR + "files/cluster_preprocess/indexes_" + file_name, 'rb')
        indexes = pickle.load(pkl_file)
        print("Load preprocessed weights.")
        return weights, indexes


def get_dataset_amount(config):
    party_dataset_amount = []
    party_num = config["party_num"]
    total = 0
    pkl_file = open(
        DATASET_DIR + "files/" + config["dataset"] + "_" + config["split"] + "_" + str(config["partition"]) + "_" + str(party_num) + "_b" + str(config.batch) + ".pkl",
        'rb')
    data = pickle.load(pkl_file)
    for index in range(party_num):
        targets = data[index]["train_y"]
        party_dataset_amount.append((index, len(targets)))
        # print(index, len(targets))
        total += len(targets)
    return party_dataset_amount

def get_validation_accuracies(config):
    party_local_validation_accuracies = []
    # meta_data = "%s_%s_%s_b%d" % (config["dataset"], config["split"], config["partition"], config["batch"])
    meta_data = "%s_%s_%s_%s_b%d" % (config.dataset, config.split, config.model, config.partition, config.batch)
    test_results = list(ensemble_selection_results.find(
        {"meta_data": meta_data, "model": config.model, "batch": config.batch, "party_num": config.party_num, "ensemble": None,
         "parties": {"$size": 1  # Find the test results in the combination contains only one model
                     }})) 
    length_test = len(test_results)
    try:
        assert length_test == config.party_num + 1
    except:
        print("length", length_test)
    # print(conditions, length_test)
    for index in range(length_test):
        party = int(test_results[index]["parties"][0])
        if party != -1:
            try:
                local_validation_accuracy = test_results[index]["local_validation_accuracy"]
                party_local_validation_accuracies.append((party, local_validation_accuracy))
            except Exception as e:
                print(e)
                t = copy.deepcopy(test_results[index])
                del t["report"]
                print(index, party, t)
                exit(-1)
    if len(party_local_validation_accuracies) != config.party_num:
        raise AssertionError("Not all models are testsed!\n\n\n\n\n")
    party_local_validation_accuracies = sorted(party_local_validation_accuracies,
                                               key=lambda party_acc: party_acc[0], reverse=False)
    return party_local_validation_accuracies

def model_averging(config):
    """
    Weights averaging function.
    Args:
        w: weights of all selected clients
        n_samples: number of samples for every client
        args: arguments from commandline

    Returns:
         the average of the weights.
    """
    if config.indexes == [-1]:
        return {
            "accuracy": -1,
            "macro avg": {
                "precision": -1,
                "recall": -1,
                "f1-score": -1,
                "support": -1
            },
            "weighted avg": {
                "precision": -1,
                "recall": -1,
                "f1-score": -1,
                "support": -1
            }
        } # skip oracle model averaging
    all_weights = read_parameters(config, flatten=False)
    w = []
    party_dataset_amount = get_dataset_amount(config)
    n_samples = []
    for i in range(len(party_dataset_amount)):
        if i in config.indexes:
            w.append(all_weights[i][1])
            n_samples.append(party_dataset_amount[i][1])
    n_total = sum(n_samples)
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        if config.avg == "fed_avg":
            w_avg[key] = (n_samples[0] / n_total) * w[0][key]
        elif config.avg == "mean_avg":
            w_avg[key] = (1/len(config.indexes)) * w[0][key]

        for i in range(1, len(w)):
            if config.avg == "fed_avg":
                w_avg[key] += (n_samples[i] / n_total) * w[i][key]
            elif config.avg == "mean_avg":
                w_avg[key] += (1/len(config.indexes)) * w[i][key]
    # if config.model == "SpinalNet":
    #     model = SpinalVGG(config.num_classes, config.input_channels)
    # elif config.model == "effenetv2_l":
    #     model = effnetv2_l(config.num_classes)
    # elif config.model == "efficientnet-b7":
    #     model = EfficientNet.from_name('efficientnet-b7', in_channels=config.input_channels,num_classes=config.num_classes)
    try:
        model = get_model(config)
        model.load_state_dict(w_avg)
    except:
        print("Change model class number to 62")
        model = get_model(config, 62)
        model.load_state_dict(w_avg)
    # model = model.to(config.device)
    model.eval()
    batch_size_test = 10

    # test_features, test_targets = get_whole_test_set(-1, config.partition, config.party_num, config.dataset, config.split, config.batch)
    # test_features = torch.Tensor(test_features)
    # test_targets = torch.Tensor(test_targets)
    from train_model import test_on_dataset, convert_dataset, get_dataset_shape
    test_whole_feature, test_whole_target = get_whole_test_set(-1, config.partition, config.party_num, config.dataset,
                  config.split, config.batch)
    test_features, test_targets = convert_dataset(config, test_whole_feature, test_whole_target)

    with torch.no_grad():
        correct = 0
        total = 0
        results = [[], []]

        for i in range(math.ceil(len(test_features) / batch_size_test)):
            begin = i * batch_size_test
            end = (i + 1) * batch_size_test
            reshape_size = min(batch_size_test, len(test_features[begin:end]))
            shape = get_dataset_shape(config.dataset)
            if config.device == "cpu":
                raise EnvironmentError("CPU is not supported, please change to GPU!")
            images = test_features[begin:end].reshape(reshape_size, shape[0], shape[1], shape[2]).type(
                torch.FloatTensor).to(config.device)
            labels = test_targets[begin:end].to(config.device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            results[0].extend(predicted.cpu().numpy().tolist())
            results[1].extend(labels.cpu().numpy().tolist())
        # print(correct,total, len(results[0]),len(results[1]))
        whole_test_accuracy = correct / total
        print("Whole test accuracy:", whole_test_accuracy)
        report = convert_report_to_json(
            classification_report(np.asarray(results)[1, :], np.asarray(results)[0, :], output_dict=True))
        return report


def convert_int(l):
    ll = list(l)
    for i in range(len(ll)):
        ll[i] = int(ll[i])
    return ll

def diversity_calculation(config):
    meta_data = "%s_%s_%s_%s_b%d" % (config.dataset, config.split, config.model, config.partition, config.batch)
    predictions_dir = MODEL_DIR + "models/" + meta_data + "/predictions/"
    disagreements = []
    cohen_kappas = []
    cohen_kappas2 = []
    for i in range(len(config.indexes)):
        prediction_1 = []
        with open(predictions_dir + "%d_%d.csv" % (config.indexes[i], config.party_num), "r") as f:
            csv_reader = csv.reader(f)
            predictions_i = list(csv_reader)
            for prediction in predictions_i:
                prediction_1.append(float(prediction[0]))
            # print(prediction_1)
        for j in range(i + 1, len(config.indexes)):
            prediction_2 = []
            with open(predictions_dir + "%d_%d.csv" % (config.indexes[j], config.party_num), "r") as f:
                csv_reader = csv.reader(f)
                predictions_i = list(csv_reader)
                for prediction in predictions_i:
                    prediction_2.append(float(prediction[0]))
                # print(prediction_2)
            p1 = np.asarray(prediction_1)
            p2 = np.asarray(prediction_2)

            difference = p1 != p2

            # Binary Disagreement
            disagreement = np.sum(difference) / len(difference)
            disagreements.append(disagreement)

            # Cohen's Kappa
            cohen_kappa = cohen_kappa_score(p1, p2)
            cohen_kappas.append(cohen_kappa)

            #


    print("BD:",len(disagreements),disagreements)
    print("Cohen's Kappa:",len(cohen_kappas),cohen_kappas)
    return np.mean(disagreements), np.mean(cohen_kappas)

if __name__ == '__main__':
    c = Config(exp_config)
    print(diversity_calculation(c))