import os
import pickle

import numpy as np

if __name__ == '__main__':
    with open('files/cifar100_cifar100_homo_20_b121.pkl', 'rb') as f:
        data = pickle.load(f)
        print(data[0]['train_X'].shape)
        print(data[0]['train_y'].shape)
        print(data[0]['test_X'].shape)
        print(data[0]['test_y'].shape)
        net_cls_counts = []
        for index in range(len(data)): # Index is the party number
            target_train = data[index]["train_y"]
            target_test = data[index]["test_y"]
            unq, unq_cnt = np.unique(target_train, return_counts=True)
            tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
            net_cls_counts.append((index, tmp))
            # Find how many labels do the party have, e.g., set_train is (3,8,9) which means this party only have data with label 3,8,9 in its training set
            set_train = set(target_train)
            set_test = set(target_test)
            overall = set_train | set_test  
            # To find whether the test set have labels that don't included in the training set
        print(net_cls_counts)