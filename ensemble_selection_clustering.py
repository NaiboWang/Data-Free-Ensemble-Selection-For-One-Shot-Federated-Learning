import copy
import json
import logging
import os
import pickle
import random
import sys

import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import SpectralClustering
from config import get_noniid_label_number_split_name, exp_config
from commandline_config import Config
from sklearn.decomposition import PCA, KernelPCA
from common_tools import repeat, read_parameters, get_dataset_amount, convert_int, get_validation_accuracies, \
    read_distribution
from dbconfig import ensemble_selection_exp_results, ensemble_selection_results
from ensemble import main
# from .test import generate_timestamp
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN

all_weights = []
flatten_weights = []
flatten_indexes = []
party_local_validation_accuracies = []
party_dataset_amount = []


def get_indexes_by_CV(config, wheres):
    K = config["K"]
    cluster_indexes = []

    for i in range(K):
        where = wheres[i]
        if len(where) > 0:  
            partial_accuracies = []
            for j in range(len(where)):
                index_ts = where[j]
                index_t = flatten_indexes[index_ts]
                if index_t == party_local_validation_accuracies[index_t][0]:
                    partial_accuracies.append(party_local_validation_accuracies[index_t])
                else:
                    logging.CRITICAL("Index does not match!", index_t)
            partial_accuracies = sorted(partial_accuracies, key=lambda party_acc: party_acc[1],
                                        reverse=True)
            index = partial_accuracies[0][0]
            cluster_indexes.append(int(index))  # avoid int64 error
    cluster_indexes.sort()
    # print(cluster_indexes)
    return cluster_indexes

def get_indexes_by_data(config, wheres):
    global party_dataset_amount
    # print(party_dataset_amount)  
    cluster_indexes = []
    K = config["K"]
    for i in range(K):
        where = wheres[i]
        if len(where) > 0:  # in case cannot have K clusters
            partial_dataset_amounts = []
            for j in range(len(where)):
                index_ts = where[j]
                index_t = flatten_indexes[index_ts]
                if index_t == party_dataset_amount[index_t][0]:
                    partial_dataset_amounts.append(party_dataset_amount[index_t])
                else:
                    logging.CRITICAL("Index does not match!", index_t)
            partial_dataset_amounts = sorted(partial_dataset_amounts, key=lambda party_amount: party_amount[1],
                                        reverse=True)
            index = partial_dataset_amounts[0][0]
            cluster_indexes.append(int(index))  
    cluster_indexes.sort()
    # print(cluster_indexes)
    return cluster_indexes


def get_indexes_by_mixed(config, wheres, difference_threshold=2):
    global party_dataset_amount, party_local_validation_accuracies
    K = config["K"]
    cluster_indexes = []

    for i in range(K):
        where = wheres[i]
        if len(where) > 0:  
            partial_accuracies = []
            partial_dataset_amounts = []
            for j in range(len(where)):
                index_ts = where[j]
                index_t = flatten_indexes[index_ts]
                if index_t != index_ts:
                    print("Index does not match:", index_ts, index_t)
                if index_t == party_local_validation_accuracies[index_t][0]:
                    partial_accuracies.append(party_local_validation_accuracies[index_t])
                else:
                    logging.CRITICAL("Index does not match!", index_t)
                if index_t == party_dataset_amount[index_t][0]:
                    partial_dataset_amounts.append(party_dataset_amount[index_t])
                else:
                    logging.CRITICAL("Index does not match!", index_t)

            partial_accuracies = sorted(partial_accuracies, key=lambda party_acc: party_acc[1],
                                        reverse=True)
            partial_dataset_amounts = sorted(partial_dataset_amounts, key=lambda party_amount: party_amount[1],
                                             reverse=True)

            # If the party with the largest amount of data has more than 3 times the amount of data than the party with the middle largest amount of data, conduct data selection, otherwise conduct CV selection.
            cluster_length = len(partial_dataset_amounts)
            index_half = cluster_length // 2
            if len(partial_dataset_amounts) > 1 and partial_dataset_amounts[0][1] / partial_dataset_amounts[index_half][1] > difference_threshold:
                index = partial_dataset_amounts[0][0]
            else:
                index = partial_accuracies[0][0]
            cluster_indexes.append(int(index))  
    cluster_indexes.sort()
    # print(cluster_indexes)
    return cluster_indexes


def hierarchical_clustering_selection(config):
    K = config["K"]
    party_num = config["party_num"]

    # print("Length of weights:",len(weights))
    Z = linkage(flatten_weights, method=config["cluster_method"])
    f = fcluster(Z, t=K, criterion='maxclust') 
    try:
        fig = plt.figure(figsize=(15, 10))
        dn = dendrogram(Z)
        # print('Z:\n', Z)
        # print('f:\n', f)
        plt.title(config["partition"] + "_" + config["cluster_method"])
        plt.show()
    except: 
        pass
    # cluster_indexes = []
    
    wheres = []
    for i in range(K):
        wheres.append(np.where(f==i+1)[0])

    if config["selection_method"] == "CV":
        cluster_indexes = get_indexes_by_CV(config, wheres)
    elif config["selection_method"] == "data":
        cluster_indexes = get_indexes_by_data(config, wheres)
    elif config["selection_method"] == "mixed":
        cluster_indexes = get_indexes_by_mixed(config, wheres)

    # print(cluster_indexes)
    # funcName = sys._getframe().f_back.f_code.co_name 
    funcName = sys._getframe().f_code.co_name  
    return cluster_indexes, funcName, convert_int(f)

def kmeans_clustering_selection(config):
    K = config["K"]
    party_num = config["party_num"]

    # Conduct KMeans Clustering
    cluster = KMeans(n_clusters=K)
    y_pred = cluster.fit_predict(flatten_weights)  
    # print(y_pred)
    # print(cluster.cluster_centers_)  
    # y_pred = cluster.predict(flatten_weights)
    # print(y_pred)  

    try:
        fig, ax1 = plt.subplots(1)  
        np_flatten_weights = np.asarray(flatten_weights)
        for i in range(K):
            ax1.scatter(np_flatten_weights[y_pred == i, 0], np_flatten_weights[y_pred == i, 1],  
                        marker='o',  
                        s=8,  
                        )  
        
        ax1.scatter(cluster.cluster_centers_[:, 0], cluster.cluster_centers_[:, 1],  
                    marker='x',  
                    s=15,  
                    c='yellow')  

        plt.title('Result after Spectral Clustering')
        plt.show()
    except: 
        pass

    wheres = []
    for i in range(K):
        wheres.append(np.where(y_pred==i)[0])

    if config["selection_method"] == "CV":
        cluster_indexes = get_indexes_by_CV(config, wheres)
    elif config["selection_method"] == "data":
        cluster_indexes = get_indexes_by_data(config, wheres)
    elif config["selection_method"] == "mixed":
        cluster_indexes = get_indexes_by_mixed(config, wheres)


    # print(cluster_indexes)
    # funcName = sys._getframe().f_back.f_code.co_name 
    funcName = sys._getframe().f_code.co_name  
    return cluster_indexes, funcName, convert_int(y_pred)

def dbscan_clustering_selection(config):
    K = config["K"]
    party_num = config["party_num"]

    # Conduct DBSCAN Clustering
    cluster = DBSCAN()
    # cluster = cluster.fit(flatten_weights)  
    # pred = cluster.labels_
    # unique_cluster_labels = set(pred)
    # n_clusters = len(unique_cluster_labels) - (-1 in pred)
    # print(cluster.cluster_centers_)  
    y_pred = cluster.fit_predict(flatten_weights)
    # print(y_pred)  

    try:
        fig, ax1 = plt.subplots(1)  
        np_flatten_weights = np.asarray(flatten_weights)
        for i in range(K):
            ax1.scatter(np_flatten_weights[y_pred == i, 0], np_flatten_weights[y_pred == i, 1],  
                        marker='o',  
                        s=8,  
                        )  
        
        ax1.scatter(cluster.cluster_centers_[:, 0], cluster.cluster_centers_[:, 1],  
                    marker='x',  
                    s=15,  
                    c='yellow')  

        plt.title('Clustering result after DBSCAN')
        plt.show()
    except: 
        pass
    cluster_indexes = []
    
    for i in range(K):
        where = np.where(y_pred==i)[0]
        if len(where)>0: 
            index = where[0]
            cluster_indexes.append(int(index)) 
    cluster_indexes.sort()
    # print(cluster_indexes)
    # funcName = sys._getframe().f_back.f_code.co_name 
    funcName = sys._getframe().f_code.co_name  
    return cluster_indexes, funcName, convert_int(y_pred)

def spectral_clustering_selection(config):
    K = config["K"]
    party_num = config["party_num"]

    # Conduct KMeans Clustering
    cluster = SpectralClustering(n_clusters=K)
    y_pred = cluster.fit_predict(flatten_weights)  
    # print(y_pred)
    # print(cluster.cluster_centers_)  
    # y_pred = cluster.predict(flatten_weights)
    # print(y_pred)  

    try:
        fig, ax1 = plt.subplots(1)  
        np_flatten_weights = np.asarray(flatten_weights)
        for i in range(K):
            ax1.scatter(np_flatten_weights[y_pred == i, 0], np_flatten_weights[y_pred == i, 1],  
                        marker='o',  
                        s=8,  
                        )  
        plt.title('Clustering result after KMeans')
        plt.show()
    except: 
        pass

    wheres = []
    for i in range(K):
        wheres.append(np.where(y_pred==i)[0])

    if config["selection_method"] == "CV":
        cluster_indexes = get_indexes_by_CV(config, wheres)
    elif config["selection_method"] == "data":
        cluster_indexes = get_indexes_by_data(config, wheres)
    elif config["selection_method"] == "mixed":
        cluster_indexes = get_indexes_by_mixed(config, wheres)


    # print(cluster_indexes)
    # funcName = sys._getframe().f_back.f_code.co_name 
    funcName = sys._getframe().f_code.co_name  
    return cluster_indexes, funcName, convert_int(y_pred)

if __name__ == '__main__':
    c = Config(exp_config)
    qualifier = ""
    if c.last_layer:
        qualifier += "last_layer_"
    if c.layer != 0:
        qualifier += "layer%d_" % c.layer
    if c.dr_method[0] != "noDimensionReduction":
        qualifier += c.dr_method[0] + "_"
    if c.label_distribution:
        qualifier += "label_distribution_"
    partitions = ["iid-diff-quantity", "homo", "noniid-labeldir", get_noniid_label_number_split_name(c.split)]
    if c.partitions[-1] == "-1":
        c.partitions = partitions
    # partitions = [get_noniid_label_number_split_name(c.split)]
    hierarchical_cluster_methods = ["centroid", "single", 'average', 'complete', 'ward',]
    # hierarchical_cluster_methods = ['ward']
    for partition in c.partitions:
        c.partition = partition
        print(c)
        if not c.label_distribution:
            flatten_weights, flatten_indexes = read_parameters(c) # read model parameters
        else:
            flatten_weights, flatten_indexes = read_distribution(c)

        if c.selection_method == "CV":
            party_local_validation_accuracies = get_validation_accuracies(c) # Get local validation accuracies for this partition
        elif c.selection_method == "data":
            party_dataset_amount = get_dataset_amount(c)
        elif c.selection_method == "mixed":
            party_local_validation_accuracies = get_validation_accuracies(c)
            party_dataset_amount = get_dataset_amount(c)

        for cluster_method in hierarchical_cluster_methods:
            c.cluster_method = cluster_method
            print(partition, cluster_method)
            repeat(hierarchical_clustering_selection, 1, config=c, qualifier=qualifier)

        # Spectral Clustering
        c.cluster_method = "spectral"
        print(partition, c["cluster_method"])
        repeat(spectral_clustering_selection, repeat_times=1, config=c, qualifier=qualifier)


        # KMeans
        c.cluster_method = "KMeans"
        print(partition, c["cluster_method"])
        repeat(kmeans_clustering_selection,repeat_times=1, config=c, qualifier=qualifier)

        # DBSCAN
        """
        Because DBSCAN need to specify min_samples and eps which is unknown and hard to measure for model with 3M+ parameters, so the clustering results are very bad, basically every model are classified as an individual cluster, so give this method up. 
        """
        # c.cluster_method = "DBSCAN"
        # print(partition, c["cluster_method"])
        # repeat(dbscan_clustering_selection,repeat_times=1, config=c, qualifier=qualifier)

        # Spectral Clustering
        # Representative model inside every cluster
        # TODO: Try variance to select
        # Select with CV Selection/Data Selection/Most Centralized model
        # PCA/LDA dimension reduction (Also, try SVD?)
        # TODO: efficientnet dimension of 118094767
        """
        NOTE: It is really a good research point of how to reduce the dimension of a deep learning model to get better clustering results! Not just use PCA/t-NSE/LDA, and how to decide the correct dimension to reduce to is another problem.
        So we need to use techs such as knowledge distillation to help us do this.
        """
        # Parameters Normalization
        # TODO: Compare with public dataset
        # Record running time
        # Memory Run Out
        # Compare with label distribution similarity, and compare indexes of label distribution with clustering indexes
        # Compare with only the clustering results of the last layer parameters
        # Maybe adjust the different proportion of dataset partition such as 3:7 to 5:5 for parties with lesser data
        # Do model averaging the see if the results are good enough, it is another aspect to compare the results (such as FedKT) VS model averaging for all models VS model averging for K models with equal weights and FedAvg weights
        # TODO: Compare with diversity methods by dataset, to see if the clustering results are good and the clustered methods really can cluster similar models together; Compute the diversity of the whole selected group;
        # TODO: Give more figures to show your results: time VS accuracy, model cluster results VS data cluster results
        # TODO: Try to explain why clustering is good

        # More dataset: at least Cifar10 and Cifar100, maybe more with tiny-imagenet
        # Spinalnet with Cifar10 or Cifar100; A new model structure on a new and old dataset.
        # Try random selected layer to see the results, such as the 4th layer of the model structure
        # TODO: try all selection groups for cifar 100 of 5, 10 and 20 parties and compare with our method