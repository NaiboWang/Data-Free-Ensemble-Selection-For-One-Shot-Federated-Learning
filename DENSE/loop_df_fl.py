#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import argparse
import copy
import os
import pickle
import shutil
import sys
import warnings
# import torchvision.models as models
import numpy as np
from tqdm import tqdm
import pdb
from model.SpinalNet import SpinalVGG
from model.resnet import ResNet50, ResNet18
from helpers.datasets import partition_data
from helpers.synthesizers import AdvSynthesizer
from helpers.utils import get_dataset, average_weights, DatasetSplit, KLDiv, setup_seed, test, test_on_dataset
from models.generator import Generator
from models.nets import CNNCifar, CNNMnist, CNNCifar100
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

from models.resnet import resnet18
from models.vit import deit_tiny_patch16_224
# import wandb

warnings.filterwarnings('ignore')
upsample = torch.nn.Upsample(mode='nearest', scale_factor=7)



class LocalUpdate(object):
    def __init__(self, args, dataset, idxs):
        self.args = args
        self.train_loader = DataLoader(DatasetSplit(dataset, idxs),
                                       batch_size=self.args.local_bs, shuffle=True, num_workers=4)

    def update_weights(self, model, client_id):
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                    momentum=0.9)
        # label_list = [0] * 100
        # for batch_idx, (images, labels) in enumerate(self.train_loader):
        #     for i in range(100):
        #         label_list[i] += torch.sum(labels == i).item()
        # print(label_list)
        local_acc_list = []
        for iter in tqdm(range(self.args.local_ep)):
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images, labels = images.cuda(), labels.cuda()
                model.zero_grad()
                # ---------------------------------------
                output = model(images)
                loss = F.cross_entropy(output, labels)
                # ---------------------------------------
                loss.backward()
                optimizer.step()
            acc, test_loss = test(model, test_loader)
            # if client_id == 0:
            #     wandb.log({'local_epoch': iter})
            # wandb.log({'client_{}_accuracy'.format(client_id): acc})
            local_acc_list.append(acc)
        return model.state_dict(), np.array(local_acc_list)


def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--epochs', type=int, default=10,
                        help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=5,
                        help="number of users: K")
    parser.add_argument('--frac', type=float, default=1,
                        help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=100,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=128,
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.5)')
    # other arguments
    parser.add_argument('--dataset', type=str, default='cifar10', help="name \
                        of dataset")
    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID.')

    # Data Free
    parser.add_argument('--adv', default=0, type=float, help='scaling factor for adv loss')

    parser.add_argument('--bn', default=0, type=float, help='scaling factor for BN regularization')
    parser.add_argument('--oh', default=0, type=float, help='scaling factor for one hot loss (cross entropy)')
    parser.add_argument('--act', default=0, type=float, help='scaling factor for activation loss used in DAFL')
    parser.add_argument('--save_dir', default='run/synthesis', type=str)
    parser.add_argument('--partition', default='dirichlet', type=str)
    parser.add_argument('--beta', default=0.5, type=float,
                        help=' If beta is set to a smaller value, '
                             'then the partition is more unbalanced')

    # Basic
    parser.add_argument('--lr_g', default=1e-3, type=float,
                        help='initial learning rate for generation')
    parser.add_argument('--input_channels', default=3, type=int, help='')
    parser.add_argument('--T', default=1, type=float)
    parser.add_argument('--g_steps', default=20, type=int, metavar='N',
                        help='number of iterations for generation')
    parser.add_argument('--batch_size', default=256, type=int, metavar='N',
                        help='number of total iterations in each epoch')
    parser.add_argument('--nz', default=256, type=int, metavar='N',
                        help='number of total iterations in each epoch')
    parser.add_argument('--synthesis_batch_size', default=256, type=int)
    # Misc
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training.')
    parser.add_argument('--device', default="cuda:1", type=str,
                        help='seed for initializing training.')
    parser.add_argument('--type', default="pretrain", type=str,
                        help='seed for initializing training.')
    parser.add_argument('--model', default="", type=str,
                        help='seed for initializing training.')
    parser.add_argument('--other', default="", type=str,
                        help='seed for initializing training.')
    args = parser.parse_args()
    return args


class Ensemble(torch.nn.Module):
    def __init__(self, model_list):
        super(Ensemble, self).__init__()
        self.models = model_list

    def forward(self, x):
        logits_total = 0
        for i in range(len(self.models)):
            logits = self.models[i](x)
            logits_total += logits
        logits_e = logits_total / len(self.models)

        return logits_e


def kd_train(synthesizer, model, criterion, optimizer, device):
    student, teacher = model
    student.train()
    teacher.eval()
    description = "loss={:.4f} acc={:.2f}%"
    total_loss = 0.0
    correct = 0.0
    with tqdm(synthesizer.get_data()) as epochs:
        for idx, (images) in enumerate(epochs):
            optimizer.zero_grad()
            images = images.cuda(device)
            with torch.no_grad():
                t_out = teacher(images)
            s_out = student(images.detach())
            loss_s = criterion(s_out, t_out.detach())

            loss_s.backward()
            optimizer.step()

            total_loss += loss_s.detach().item()
            avg_loss = total_loss / (idx + 1)
            pred = s_out.argmax(dim=1)
            target = t_out.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum().item()
            acc = correct / len(synthesizer.data_loader.dataset) * 100

            epochs.set_description(description.format(avg_loss, acc))


def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    if is_best:
        torch.save(state, filename)


def get_model(args):
    if args.model == "mnist_cnn":
        global_model = CNNMnist().cuda()
    elif args.model == "fmnist_cnn":
        global_model = CNNMnist().cuda()
    elif args.model == "cnn":
        global_model = CNNCifar().cuda(int(args.device[-1]))
    elif args.model == "resnet50":
        global_model = ResNet50(num_classes=10, input_channels=3).to(args.device)
    elif args.model == "svhn_cnn":
        global_model = CNNCifar().cuda()
    elif args.model == "SpinalNet":
        global_model = SpinalVGG(10, 1).to(args.device)
    elif args.model == "cifar100_cnn":
        global_model = CNNCifar100().cuda()
    elif args.model == "res":
        # global_model = resnet18()
        global_model = resnet18(num_classes=100).cuda()

    elif args.model == "vit":
        global_model = deit_tiny_patch16_224(num_classes=1000,
                                             drop_rate=0.,
                                             drop_path_rate=0.1)
        global_model.head = torch.nn.Linear(global_model.head.in_features, 10)
        global_model = global_model.cuda()
        global_model = torch.nn.DataParallel(global_model)
    return global_model

def get_whole_test_set(dataset_name):
    party_num = int(dataset_name.split("_")[-2])
    pkl_file = open(dataset_name, 'rb')
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

if __name__ == '__main__':

    args = args_parser()
    # wandb.init(config=args,
    #            project="ont-shot FL")

    setup_seed(args.seed)
    # pdb.set_trace()
    train_dataset, test_dataset, user_groups, traindata_cls_counts = partition_data(
        args.dataset, args.partition, beta=args.beta, num_users=args.num_users)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256,
                                              shuffle=False, num_workers=4)
    # BUILD MODEL

    global_model = get_model(args)
    bst_acc = -1
    description = "inference acc={:.4f}% loss={:.2f}, best_acc = {:.2f}%"
    local_weights = []
    global_model.train()
    acc_list = []
    users = []
    if args.type == "pretrain":
        # ===============================================
        for idx in range(args.num_users):
            print("client {}".format(idx))
            users.append("client_{}".format(idx))
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx])
            w, local_acc = local_model.update_weights(copy.deepcopy(global_model), idx)

            acc_list.append(local_acc)
            local_weights.append(copy.deepcopy(w))

        # wandb 

        for i in range(args.local_ep):
            print({"client_{}_acc".format(users[0]):acc_list[0][i],
                "client_{}_acc".format(users[1]):acc_list[1][i],
                "client_{}_acc".format(users[2]):acc_list[2][i],
                "client_{}_acc".format(users[3]):acc_list[3][i],
                "client_{}_acc".format(users[4]):acc_list[4][i],
            })
        # np.save("client_{}_acc.npy".format(args.num_users), acc_list)
        print({"client_accuracy" : acc_list})
        # torch.save(local_weights, '{}_{}.pkl'.format(name, iid))
        torch.save(local_weights, '{}_{}clients_{}.pkl'.format(args.dataset, args.num_users, args.beta))
        # update global weights by FedAvg
        print(args)
        global_weights = average_weights(local_weights)
        global_model.load_state_dict(global_weights)
        print("avg acc:")
        test_acc, test_loss = test(global_model, test_loader)
        model_list = []
        for i in range(len(local_weights)):
            net = copy.deepcopy(global_model)
            net.load_state_dict(local_weights[i])
            model_list.append(net)
        ensemble_model = Ensemble(model_list)
        print("ensemble acc:")
        test(ensemble_model, test_loader)
        # ===============================================
    else:
        # ===============================================
        model_dir = "emnist_digits_SpinalNet_iid-diff-quantity_b201"
        dataset_name = "emnist_digits_iid-diff-quantity_400_b201.pkl"
        print(args, model_dir, dataset_name)
        party_num = dataset_name.split("_")[-2]
        party_num_length = -len(party_num)
        print("party_num_length", party_num_length, "party_num", party_num)
        model_list_dir = os.listdir(model_dir)
        # print(model_list_dir)
        for model in model_list_dir:
            if model.split(".")[-1] != "pkl" or model.split(".")[0][party_num_length:] != party_num or model.find("-1") >= 0:
                continue
            local_weights.append(torch.load(model_dir + "/{}".format(model), map_location=torch.device(args.device)))

        # local_weights = [torch.load(model_dir + "/party_-1_5.pkl", map_location=torch.device(args.device))]

        print(len(local_weights))
        # local_weights = torch.load('{}_{}clients_{}.pkl'.format(args.dataset, args.num_users, args.beta))
        global_weights = average_weights(local_weights)
        global_model.load_state_dict(global_weights)
        print("avg acc:")
        test_features, test_targets = get_whole_test_set(dataset_name)
        test_features, test_targets = convert_dataset(args, test_features, test_targets)
        test_accuracy = test_on_dataset(args, test_features, test_targets, global_model, args.device, batch_size_test=args.batch_size, name="test")
        # test_acc, test_loss = test(global_model, test_loader, args.device)
        model_list = []
        for i in range(len(local_weights)):
            net = copy.deepcopy(global_model)
            net.load_state_dict(local_weights[i])
            model_list.append(net)
        ensemble_model = Ensemble(model_list)

        # test(ensemble_model, test_loader, args.device)
        print("ensemble acc:")
        # test_accuracy = test_on_dataset(args, test_features, test_targets, ensemble_model, args.device, batch_size_test=args.batch_size, name="test", jd=True)

        # ===============================================
        global_model = get_model(args)
        # ===============================================

        # data generator
        nz = args.nz
        nc = 3 if "cifar" in args.dataset or args.dataset == "svhn" else 1
        img_size = 32 if "cifar" in args.dataset or args.dataset == "svhn" else 28
        generator = Generator(nz=nz, ngf=64, img_size=img_size, nc=nc).cuda(int(args.device[-1]))
        args.cur_ep = 0
        img_size2 = (3, 32, 32) if "cifar" in args.dataset or args.dataset == "svhn" else (1, 28, 28)
        num_class = 100 if args.dataset == "cifar100" else 10
        synthesizer = AdvSynthesizer(ensemble_model, model_list, global_model, generator,
                                     nz=nz, num_classes=num_class, img_size=img_size2,
                                     iterations=args.g_steps, lr_g=args.lr_g,
                                     synthesis_batch_size=args.synthesis_batch_size,
                                     sample_batch_size=args.batch_size,
                                     adv=args.adv, bn=args.bn, oh=args.oh,
                                     save_dir=args.save_dir, dataset=args.dataset, device=int(args.device[-1]))
        # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        criterion = KLDiv(T=args.T)
        optimizer = torch.optim.SGD(global_model.parameters(), lr=args.lr,
                                    momentum=0.9)
        global_model.train()
        distill_acc = []
        for epoch in tqdm(range(args.epochs)):
            # 1. Data synthesis
            synthesizer.gen_data(args.cur_ep, int(args.device[-1]))  # g_steps
            args.cur_ep += 1
            kd_train(synthesizer, [global_model, ensemble_model], criterion, optimizer, int(args.device[-1]))  # # kd_steps
            # acc, test_loss = test(global_model, test_loader, args.device)
            acc = test_on_dataset(args, test_features, test_targets, global_model, args.device, batch_size_test=args.batch_size, name="test")
            distill_acc.append(acc)
            is_best = acc > bst_acc
            bst_acc = max(acc, bst_acc)
            _best_ckpt = 'df_ckpt/{}.pth'.format(args.other)
            print("best acc:{}".format(bst_acc))
            save_checkpoint({
                'state_dict': global_model.state_dict(),
                'best_acc': float(bst_acc),
            }, is_best, _best_ckpt)
            print({'accuracy': acc})

        print({"global_accuracy": distill_acc})
        np.save("distill_acc_{}.npy".format(args.dataset), np.array(distill_acc))

        # ===============================================


