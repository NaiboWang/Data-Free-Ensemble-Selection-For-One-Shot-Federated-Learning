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
        json.dump(output_info, f)
        ensemble_selection_exp_results.insert_one(output_info)

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
    for index in range(party_num):
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
    return party_dataset_distribution


def read_parameters(config, flatten = True):
    all_weights, flatten_weights = [], []
    party_num = config["party_num"]
    meta_data = "%s_%s_%s_%s_b%d" % (config["dataset"], config["split"], config.model, config["partition"], config["batch"])
    save_dir = MODEL_DIR + "models/" + meta_data
    if not os.path.exists(DATASET_DIR + "files/cluster_preprocess"):
        os.mkdir(DATASET_DIR + "files/cluster_preprocess")
    if config["normalization"] == 1:
        file_name = meta_data + "_" + str(party_num) + "_" + config.dr_method[0]
    else:
        file_name = meta_data + "_" + str(party_num) + "_" + config.dr_method[0] + "_noNormalization"
    if config.last_layer:
        file_name += "_lastLayer"
    if config.layer != 0:
        file_name += "_layer%d" % config.layer
    if not flatten:
        file_name += "_original"
    file_name += ".pkl"
    if not os.path.exists(DATASET_DIR + "files/cluster_preprocess/"+file_name):
    # if True:
        for index in range(party_num):
            model_path = save_dir + "/party_%d_%d.pkl" % (index, party_num)
            # print(model_path)
            config_c = copy.deepcopy(config)
            config_c.device = "cpu"
            model = get_model(config_c)
            model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu'))) # ensure models are loaded via cpu and main memory, not gpu cuda memory
            weights = model.state_dict()
            all_weights.append((index,copy.deepcopy(weights)))
        for key in model.state_dict().keys():
            print(key)
        if flatten:
            # 第一种方式，把所有的权值展平成一层，形成一个mXn的二维矩阵，m是party number，n是单个模型所有的节点数量，即纬度数
            all_keys = all_weights[0][1].keys()
            random_indexes = list(random.sample(range(len(all_keys) - 1), min(party_num, int(len(all_keys) * 0.1))))
            random_indexes.sort()
            random_keys = list(np.asarray(list(all_keys))[random_indexes])
            if config.layer == -1:
                print("random keys:", random_keys)
            for index in range(party_num):
                weights = []
                for key in all_weights[index][1].keys():
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
                            weight = all_weights[index][1][key].cpu().numpy().flatten()
                            weights.extend(weight)
                    else:
                        if config.layer == 0: # all layers
                            weight = all_weights[index][1][key].cpu().numpy().flatten()
                            weights.extend(weight)
                        elif config.layer == -1: # random layers
                            if key in random_keys:
                                weight = all_weights[index][1][key].cpu().numpy().flatten()
                                weights.extend(weight)
                        else: # TODO: Automatically generate model layer names
                            if config.model == "SpinalNet":
                                layer_name = [['l1.0.weight', 'l1.0.bias', 'l1.1.weight', 'l1.1.bias', 'l1.1.running_mean', 'l1.1.running_var', 'l1.1.num_batches_tracked', 'l1.3.weight', 'l1.3.bias', 'l1.4.weight', 'l1.4.bias', 'l1.4.running_mean', 'l1.4.running_var', 'l1.4.num_batches_tracked'],['fc_spinal_layer1.1.weight', 'fc_spinal_layer1.1.bias', 'fc_spinal_layer1.2.weight', 'fc_spinal_layer1.2.bias', 'fc_spinal_layer1.2.running_mean', 'fc_spinal_layer1.2.running_var', 'fc_spinal_layer1.2.num_batches_tracked'],['fc_spinal_layer4.1.weight', 'fc_spinal_layer4.1.bias', 'fc_spinal_layer4.2.weight', 'fc_spinal_layer4.2.bias', 'fc_spinal_layer4.2.running_mean', 'fc_spinal_layer4.2.running_var', 'fc_spinal_layer4.2.num_batches_tracked']]
                            elif config.model == "effenetv2_l":
                                layer_name = [
                                    ['features.0.0.weight', 'features.0.1.weight', 'features.0.1.bias', 'features.0.1.running_mean', 'features.0.1.running_var', 'features.0.1.num_batches_tracked'],
                                    ['features.35.conv.1.weight', 'features.35.conv.1.bias', 'features.35.conv.1.running_mean', 'features.35.conv.1.running_var', 'features.35.conv.1.num_batches_tracked', 'features.35.conv.3.weight', 'features.35.conv.4.weight', 'features.35.conv.4.bias', 'features.35.conv.4.running_mean', 'features.35.conv.4.running_var', 'features.35.conv.4.num_batches_tracked', 'features.35.conv.6.fc.0.weight', 'features.35.conv.6.fc.0.bias', 'features.35.conv.6.fc.2.weight', 'features.35.conv.6.fc.2.bias', 'features.35.conv.7.weight', 'features.35.conv.8.weight', 'features.35.conv.8.bias', 'features.35.conv.8.running_mean', 'features.35.conv.8.running_var', 'features.35.conv.8.num_batches_tracked', ],
                                    ['classifier.weight', 'classifier.bias']]
                            elif config.model == "efficientnet-b7":
                                layer_name = [
                                    ['_conv_stem.weight', '_bn0.weight', '_bn0.bias', '_bn0.running_mean', '_bn0.running_var', '_bn0.num_batches_tracked', '_blocks.0._depthwise_conv.weight', '_blocks.0._bn1.weight', '_blocks.0._bn1.bias', '_blocks.0._bn1.running_mean', '_blocks.0._bn1.running_var', '_blocks.0._bn1.num_batches_tracked', '_blocks.0._se_reduce.weight', '_blocks.0._se_reduce.bias', '_blocks.0._se_expand.weight', '_blocks.0._se_expand.bias', '_blocks.0._project_conv.weight', '_blocks.0._bn2.weight', '_blocks.0._bn2.bias', '_blocks.0._bn2.running_mean', '_blocks.0._bn2.running_var', '_blocks.0._bn2.num_batches_tracked', ],['_blocks.25._expand_conv.weight', '_blocks.25._bn0.weight', '_blocks.25._bn0.bias', '_blocks.25._bn0.running_mean', '_blocks.25._bn0.running_var', '_blocks.25._bn0.num_batches_tracked', '_blocks.25._depthwise_conv.weight', '_blocks.25._bn1.weight', '_blocks.25._bn1.bias', '_blocks.25._bn1.running_mean', '_blocks.25._bn1.running_var', '_blocks.25._bn1.num_batches_tracked', '_blocks.25._se_reduce.weight', '_blocks.25._se_reduce.bias', '_blocks.25._se_expand.weight', '_blocks.25._se_expand.bias', '_blocks.25._project_conv.weight', '_blocks.25._bn2.weight', '_blocks.25._bn2.bias', '_blocks.25._bn2.running_mean', '_blocks.25._bn2.running_var', '_blocks.25._bn2.num_batches_tracked',],['_blocks.54._expand_conv.weight', '_blocks.54._bn0.weight', '_blocks.54._bn0.bias', '_blocks.54._bn0.running_mean', '_blocks.54._bn0.running_var', '_blocks.54._bn0.num_batches_tracked', '_blocks.54._depthwise_conv.weight', '_blocks.54._bn1.weight', '_blocks.54._bn1.bias', '_blocks.54._bn1.running_mean', '_blocks.54._bn1.running_var', '_blocks.54._bn1.num_batches_tracked', '_blocks.54._se_reduce.weight', '_blocks.54._se_reduce.bias', '_blocks.54._se_expand.weight', '_blocks.54._se_expand.bias', '_blocks.54._project_conv.weight', '_blocks.54._bn2.weight', '_blocks.54._bn2.bias', '_blocks.54._bn2.running_mean', '_blocks.54._bn2.running_var', '_blocks.54._bn2.num_batches_tracked']
                                ]
                            if key in layer_name[config.layer-1]:
                                weight = all_weights[index][1][key].cpu().numpy().flatten()
                                weights.extend(weight)
                flatten_weights.append(weights)
            # n_components = int(config.dr_method[1] * len(flatten_weights[0]))
            if config.dr_method[0] == "PCA":
                pca = PCA(n_components=party_num)
                flatten_weights = pca.fit(flatten_weights).transform(flatten_weights)
            elif config.dr_method[0] == "Kernel_PCA":
                kernel_pca = KernelPCA(n_components=party_num)
                # c = kernel_pca.fit(flatten_weights)
                # d = c.transform(flatten_weights)
                flatten_weights = kernel_pca.fit(flatten_weights).transform(flatten_weights)

            if config["normalization"] == 1:
                mm = MinMaxScaler()
                flatten_weights = mm.fit_transform(flatten_weights)
            with open(DATASET_DIR + "files/cluster_preprocess/" + file_name, 'wb') as fid:
                pickle.dump(flatten_weights, fid)
            print("Min Max Transformed and saved to %s." % file_name)
            return flatten_weights
        else:
            with open(DATASET_DIR + "files/cluster_preprocess/" + file_name, 'wb') as fid:
                pickle.dump(all_weights, fid)
            return all_weights
    else:
        pkl_file = open(DATASET_DIR + "files/cluster_preprocess/" + file_name, 'rb')
        weights = pickle.load(pkl_file)
        print("Load preprocessed weights.")
        return weights


def get_dataset_amount(config):
    party_dataset_amount = []
    K = config["K"]
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
                     }}))  # 查找原始测试数据集的测试结果
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
    model = get_model(config)
    # TODO:实验看结果是否正确
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