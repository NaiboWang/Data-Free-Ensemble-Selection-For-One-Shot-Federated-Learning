import json
import logging
import os
import pickle
import random
import re
from collections import Counter


import commandline_config
import numpy as np

from config import get_noniid_label_number_split_name

np.set_printoptions(suppress=True)
import torch
import torchvision

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class Cutout:
    def __init__(self, size=16, p=0.5):
        self.size = size
        self.half_size = size // 2
        self.p = p

    def __call__(self, image):
        if torch.rand([1]).item() > self.p:
            return image

        left = torch.randint(-self.half_size, image.size(1) - self.half_size, [1]).item()
        top = torch.randint(-self.half_size, image.size(2) - self.half_size, [1]).item()
        right = min(image.size(1), left + self.size)
        bottom = min(image.size(2), top + self.size)

        image[:, max(0, left): right, max(0, top): bottom] = 0
        return image

def convert_pkl(dataset="emnist",split="digits"):
    if os.path.exists("files/" + dataset +"_"+ split + ".pkl"):
        print("Dataset %s already exists, stop converting original emnist dataset to pkl file." % dataset)
        return
    if dataset == "emnist":
        train_set = torchvision.datasets.EMNIST('files/', split=split, train=True, download=True,
                                                transform=torchvision.transforms.Compose([
                                                    torchvision.transforms.RandomPerspective(),
                                                    torchvision.transforms.RandomRotation(10, fill=(0,)),
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize(
                                                        (0.1307,), (0.3081,))
                                                ]))

        test_set = torchvision.datasets.EMNIST('files/', split=split, train=False, download=True,
                                               transform=torchvision.transforms.Compose([
                                                   torchvision.transforms.ToTensor(),
                                                   torchvision.transforms.Normalize(
                                                       (0.1307,), (0.3081,))
                                               ]))
    elif dataset == "cifar10":
        transform_train = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_set = torchvision.datasets.CIFAR10('files/', train=True, download=True,
                                                # transform=torchvision.transforms.Compose([
                                                #     torchvision.transforms.ToTensor(),
                                                #     torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                                # ])
                                                 transform=transform_train,
                                                 )
        test_set = torchvision.datasets.CIFAR10('files/', train=False, download=True,
                                                # transform=torchvision.transforms.Compose([
                                                #     torchvision.transforms.ToTensor(),
                                                #     torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                                # ])
                                                transform=transform_test,
                                                )
    elif dataset == "cifar100":
        CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
        CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
        # stats = ((0.5074, 0.4867, 0.4411), (0.2011, 0.1987, 0.2025))
        stats = (CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(*stats)
        ])

        test_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(*stats)
        ])
        train_set = torchvision.datasets.CIFAR100('files/', train=True, download=True,
                                                transform=train_transform)

        test_set = torchvision.datasets.CIFAR100('files/', train=False, download=True,
                                                transform=test_transform)
    elif dataset == "tinyimagenet":
        pass
    train_set_X = train_set.data.tolist()
    test_set_X = test_set.data.tolist()

    if commandline_config.commandline_config.check_type(train_set.targets) == "list":
        train_set_y = train_set.targets
        test_set_y = test_set.targets
    else:
        train_set_y = train_set.targets.tolist()
        test_set_y = test_set.targets.tolist()
    train_set_X.extend(test_set_X)
    train_set_y.extend(test_set_y)
    emnist = {
        "X": train_set_X,
        "y": train_set_y,
    }
    with open("files/" + dataset +"_"+ split +".pkl", 'wb') as fid:
        pickle.dump(emnist, fid)
    print("Done!")


def record_net_data_stats(y_train, net_dataidx_map, logdir):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp

    """
    Data statistics: {0: {0: 269, 1: 283, 2: 250, 3: 326, 4: 293, 5: 267, 6: 289, 7: 273, 8: 272, 9: 278}, 1: {0: 255, 1: 309, 2: 253, 3: 276, 4: 309, 5: 284, 6: 251, 7: 273, 8: 287, 9: 303}}
    """
    logger.info('Data statistics: %s' % str(net_cls_counts))

    return net_cls_counts


def partition_data(dataset="emnist", datadir="data/", split="default", logdir="logs/", partition="homo", n_parties=20, beta=0.4, num_label =10):
    # np.random.seed(2020)
    # torch.manual_seed(2020)
    pkl_file = open("files/" + dataset + "_"+ split + '.pkl', 'rb')
    data = pickle.load(pkl_file)
    X_train = np.asarray(data["X"])
    if split == "letters":
        y_train = np.asarray(data["y"]) - 1
    else:
        y_train = np.asarray(data["y"])

    n_train = y_train.shape[0]

    if partition == "homo":
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, n_parties)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}

    elif partition == "noniid-labeldir":
        min_size = 0
        min_require_size = 120
        K = num_label  # 标签数量

        N = y_train.shape[0]
        np.random.seed(2020)
        net_dataidx_map = {}

        # 重复分配直到每个party的数据量都达到了最小需要的size
        while min_size < min_require_size:
            idx_batch = [[] for _ in range(n_parties)]
            # ?这里有时候会分配到某个party导致缺少一些标签，是否正常？比如party 0 只包含012345没有6789
            for k in range(K):
                # np.where 只有条件 (condition)，没有x和y，则输出满足条件 (标签为k) 元素的坐标
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                # np.repeat 将第一个参数重复第二个参数次生成一个数组

                proportions = np.random.dirichlet(np.repeat(beta, n_parties))
                # logger.info("proportions1: ", proportions)
                # logger.info("sum pro1:", np.sum(proportions))
                ## Balance
                # 下面公式先判断idx_j的长度是否小于N/n_parties，如果是返回1否则0，再乘p
                # zip(a,b)根据a和b的最短长度决定zip大小，然后分别输出a和b的对应元素值
                # 下面这行的意思是，如果当前party的元素数量已经超过了N/n_parties则返回0，即不往数组里继续添加当前元素，否则返回对应比例
                proportions = np.array([p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])
                # logger.info("proportions2: ", proportions)
                proportions = proportions / proportions.sum()
                # logger.info("proportions3: ", proportions)
                # cumsum([1,2,3,1]) => [1,3,6,7], [:-1]去掉最后一个元素
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                # logger.info("proportions4: ", proportions)
                # split(ary, indices_or_sections, axis=0) :把一个数组从左到右按顺序切分,下面一行就是将idx_k这个标签对应的元素的数组按照生成好的分配规则切分之后放在idx_batch数组中
                # https://blog.csdn.net/mingyuli/article/details/81227629
                # [] + [1,2] = [1,2], [1] + [2,3] = [1,2,3]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                # 小bug，不应该写在这里
                min_size = min([len(idx_j) for idx_j in idx_batch])
                # if K == 2 and n_parties <= 10:
                #     if np.min(proportions) < 200:
                #         min_size = 0
                #         break

        for j in range(n_parties):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    elif partition > "noniid-#label0" and partition <= "noniid-#label999":
        num = eval(partition[13:])  # 得到#label后面的数字
        num -= 1
        K = num_label
        assert (num+1) * n_parties > num_label # ensure all labels are covered by n_parties
        # 如何label后面的值为K，如数字数据集为10的时候，即需要给每个party分配所有的类别的数据
        if num == K:
            # 每个party初始化一个空int数组
            net_dataidx_map = {i: np.ndarray(0, dtype=np.int64) for i in range(n_parties)}
            for i in range(K):
                idx_k = np.where(y_train == i)[0]
                np.random.shuffle(idx_k)
                # 把idx_k平均切分成n_parties份
                split = np.array_split(idx_k, n_parties)
                # 往net_dataidx_map数组中添加split后的indexes
                for j in range(n_parties):
                    net_dataidx_map[j] = np.append(net_dataidx_map[j], split[j])
        else: # 否则，给每个party分别指定数量类别的数据
            times = [0 for i in range(K)]
            contain = []
            ## BUG: 如果party数目小于类别数目，会存在有个别类别没有被任何一个party选中的情况，此时需要重选
            while 0 in times:
                print("reselect of noniid-#label")
                times = [0 for i in range(K)]
                contain = []
                for i in range(n_parties):
                    current = [i%K]
                    times[i % K] += 1  # 第i%K个标签值+1，即times里存储的是当前label值有多少个party在选
                    j = 0
                    # 因为num的意思是每个party所拥有的label数量，所以要保证current数组里有num个元素，每个元素代表一个标签值，不停的生成随机数直到满足条件为止
                    while (j < num):
                        ind = random.randint(0, K - 1)  # 返回0到K-1的随机值，左右都包含，即有可能抽中0到K-1的任意一个值
                        # bug 下面的括号没必要加
                        if ind not in current:  # 如果随机出的标签值是新标签，加入current数组中并且times对应label的party数量+1
                            j = j + 1
                            current.append(ind)
                            times[ind] += 1
                    contain.append(current)  # 把数组添加到contain中去,contain数组最终的样式为：
                    # [[1,3],[2,3],...,[2,4]] 如果num为2
                # print("contain: ", contain)
            net_dataidx_map = {i: np.ndarray(0, dtype=np.int64) for i in range(n_parties)}
            for i in range(K):
                idx_k = np.where(y_train == i)[0]
                np.random.shuffle(idx_k)
                # times[i]代表当前label值有几个party选中了，然后把idx_k平分为几份
                split = np.array_split(idx_k, times[i])
                ids = 0
                for j in range(n_parties):
                    if i in contain[j]: # i from 0 to num_label-1
                        net_dataidx_map[j] = np.append(net_dataidx_map[j], split[ids])
                        ids += 1

    elif partition == "iid-diff-quantity":
        # 直接对所有的训练数据，不分标签进行打乱重组然后切分，与homo的区别就在于有的party数据多有的少，homo是均分
        idxs = np.random.permutation(n_train)
        min_size = 0
        if n_parties < 500:
            MIN = 120
        else:
            MIN = 12
        min_proprotion = MIN * 2 / len(idxs) # Ensure every party can get minimum number of samples
        while min_size < MIN:
            proportions = np.random.dirichlet(np.repeat(beta, n_parties))
            # print(proportions)
            proportions += min_proprotion
            # print(proportions)
            proportions = proportions / proportions.sum()
            # print(proportions)
            print(len(idxs),proportions * len(idxs))
            min_size = np.min(proportions * len(idxs))
            print(min_size)
        proportions = (np.cumsum(proportions) * len(idxs)).astype(int)[:-1]
        batch_idxs = np.split(idxs, proportions)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}

    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map, logdir)
    return X_train, y_train, net_dataidx_map, traindata_cls_counts


def read_data_partition():
    files = os.listdir("files/")
    print(files)
    for file in files:
        if 'pkl' in file and len(file.split("_"))>2:
            pkl_file = open("files/" + file, 'rb')
            data = pickle.load(pkl_file)
            print(file)
            """
            Data format
            e.g., emnist_noniid-#label3_50.pkl
            一个列表包含了50个dict，每个dict为一个party的数据
            每个dict里有四个数组项：train_X, train_y, test_X, test_y，即party包含的数据信息，有n条
            """
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
                overall = set_train | set_test  # 求并集
                # To find whether the test set have labels that don't included in the training set, 错误检查
                if len(overall - set_train) != 0:
                    print(file, index, len(target_train), len(target_test))
            print(net_cls_counts)


if __name__ == '__main__':
    # read_data_partition()
    config_emnist_digits = [201, "emnist", 'digits', 10, [5,10,100,200,400]]
    config_emnist_letters = [201, "emnist", 'letters', 26, [5,10,100,200,400]]
    config_emnist_balanced = [201, "emnist", 'balanced', 47, [5,10,100,200,400]]
    config_cifar10 = [211, "cifar10", 'cifar10', 10, [5,10,50,100,200]]
    config_cifar100 = [221, "cifar100", 'cifar100', 100, [5,10,20]]
    
    c = commandline_config.Config({
        "dataset": "emnist",
        "split": "digits",
        "ID": 0,
        "num_clients":[5,10,100,200,400],
        "ratio": [0.7,0.1,0.2], # partition ratio for training/validation/test set
    })

    if c.split == "digits":
        config = config_emnist_digits
    elif c.split == "letters":
        config = config_emnist_letters
    elif c.split == "balanced":
        config = config_emnist_balanced
    elif c.split == "cifar10":
        config = config_cifar10
    elif c.split == "cifar100":
        config = config_cifar100
    config[0] = c.ID
    config[4] = c.num_clients

    batch = config[0]
    dtset = config[1]
    split = config[2]
    num_label = config[3] # 10, 26, 47 for digits, letters, balanced of emnist
    party_nums = config[4]
    # party_nums = [20]
    convert_pkl(dataset=dtset, split=split)

    for n_parties in party_nums:
        partitions = [get_noniid_label_number_split_name(split),  "homo","noniid-labeldir", "iid-diff-quantity",  ]
        # partitions = ["iid-diff-quantity"]
        # partitions = [get_noniid_label_number_split_name(split), "homo", "iid-diff-quantity", ]
        # partitions = ["noniid-#label3", "noniid-labeldir", "homo"]
        for partition in partitions:
            file_name = "files/%s_%s_" % (dtset, split) + str(partition) + "_" + str(n_parties) + "_b" + str(batch) + ".pkl"
            if os.path.exists(file_name):
                print("Dataset partition %s_%s_%s_%d.pkl already exists, work done." % (dtset, split, partition, n_parties))
            else:
                """
                net_dataidx_map 是一个dict，有num_party个项，每个项是一个数组，比如0这个项对应的数组就是party 0有2800个数据，每个数据是一个索引值，意思是party 0有2800个数据（包括训练和测试集合），对应原来dataset的索引值
                """
                X, y, net_dataidx_map, traindata_cls_counts = partition_data(dataset=dtset, partition=partition,split=split,
                                                     n_parties=n_parties,num_label=num_label)

                data = []
                traindata_cls_counts = {}
                for i in range(n_parties):
                    map = np.asarray(net_dataidx_map[i])
                    n = map.shape[0]
                    ratio = c.ratio
                    while True:
                        idxs = np.random.permutation(n)  # 随机打乱
                        map = map[idxs]
                        train_num = int(n * ratio[0])
                        train_idx = map[0:train_num]
                        validation_num = int(n*(ratio[0]+ratio[1]))
                        validation_idx = map[train_num:validation_num]
                        test_idx = map[validation_num:]
                        train_X = X[train_idx]
                        train_y = y[train_idx]
                        validation_X = X[validation_idx]
                        validation_y = y[validation_idx]
                        test_X = X[test_idx]
                        test_y = y[test_idx]
                        set_train = set(train_y)
                        set_validation = set(validation_y)
                        set_test = set(test_y)
                        overall = set_train | set_test | set_validation  # 求并集
                        if len(overall - set_train) == 0:  # 必须保证训练集里包含所有的标签数据
                            break
                        else:
                            # break
                            print(overall - set_train, set_train)
                            print(f"Retry Partition for party {i} with n={n}")
                    result = {
                        "train_X": train_X,
                        "train_y": train_y,
                        "validation_X": validation_X,
                        "validation_y": validation_y,
                        "test_X": test_X,
                        "test_y": test_y,
                    }
                    # count_result一个dict统计了各个类数据的数量
                    count_result = dict(Counter(train_y))
                    counts = {}
                    for j in range(num_label):
                        if j in count_result:
                            counts[str(j)] = count_result[j]
                    traindata_cls_counts[str(i)] = counts
                    data.append(result)
                print(traindata_cls_counts)
                json_file = "dataset_partition_info.json"
                dataset_partition_name = "%s_%s_%s_%d" % (dtset, split, partition, n_parties)
                # traindata_cls_counts = json.loads(re.sub(r'(\d+):', r'"\1":', str(traindata_cls_counts)))
                dpit = {dataset_partition_name: traindata_cls_counts}
                # if os.path.exists(json_file):
                #     with open(json_file, 'r') as file_obj:
                #         content = file_obj.read()
                #         dpi = json.loads(content)
                # else:
                #     dpi = []
                # dpi.append(dpit)
                # with open(json_file, 'w') as file_obj:
                #     json.dump(dpi, file_obj)
                with open(file_name, 'wb') as fid:
                    pickle.dump(data, fid)
            print("Done with partition %s" % partition)
        print("Done with dataset %s, partyNum: %d" % (dtset, n_parties))
    print("Done with dataset %s" % dtset)
