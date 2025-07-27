#
# Copyright 2022- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#
import os.path

import numpy as np
import torch
from code.data.sampler import CategoriesSampler
# import code.data.cifar100.cifar100 as Dataset

"""
replace arg part to ue config.py file as much as you can.
if it is hard, then args is ok. we would modify the base_train.py, 
incremental_run, train_biag according to this

"""
# def set_up_datasets(args):
#
#     if args.dataset == 'miniimagenet':
#         args.base_class = 60
#         args.num_classes=100
#         args.way = 5
#         args.shot = 5
#         args.sessions = 9
#     elif args.dataset == 'cifar100':
#         args.base_class = 60
#         args.num_classes=100
#         args.way = 5
#         args.shot = 5
#         args.sessions = 9
#
#     args.Dataset=Dataset
#     return args

def set_up_datasets(args):
    if args.dataset == 'miniimagenet':
        args.base_class = 60; args.num_classes = 100
        args.way = 5; args.shot = 5; args.sessions = 9

        import importlib
        args.Dataset = importlib.import_module("code.data.miniimagenet.miniimagenet")
    elif args.dataset == 'cifar100':
        args.base_class = 60; args.num_classes = 100
        args.way = 5; args.shot = 5; args.sessions = 9
        import importlib
        args.Dataset = importlib.import_module("code.data.cifar100.cifar100")
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    return args

def _index_dirname(args):
    return "mini_imagenet" if args.dataset == "miniimagenet" else args.dataset

def get_dataloader(args,session):
    if session == 0:
        trainset, trainloader, testloader = get_base_dataloader_meta(args, do_augment =False)
    else:
        trainset, trainloader, testloader = get_new_dataloader(args,session, do_augment =False)
    return trainset, trainloader, testloader

def get_base_dataloader(args):
    # txt_path = "code/data/index_list/" + args.dataset + "/session_" + str(0 + 1) + '.txt'
    class_index = np.arange(args.base_class)

    if args.dataset == 'cifar100':
        trainset = args.Dataset.CIFAR100(root=args.data_folder, train=True, download=True,
                                         index=class_index, base_sess=True)
        testset = args.Dataset.CIFAR100(root=args.data_folder, train=False, download=False,
                                        index=class_index, base_sess=True)
    elif args.dataset == 'miniimagenet':
        trainset = args.Dataset.MiniImageNet(root=args.data_folder, train=True,
                                             index=class_index, base_sess=True)
        testset = args.Dataset.MiniImageNet(root=args.data_folder, train=False, index=class_index)

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_training, shuffle=True,
                                              num_workers=8, pin_memory=True)
    testloader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=args.batch_size_inference, shuffle=False, num_workers=8, pin_memory=True)

    return trainset, trainloader, testloader



# def get_base_dataloader_meta(args,do_augment=True):
#     txt_path = "code/data/index_list/" + args.dataset + "/session_" + str(0 + 1) + '.txt'
def get_base_dataloader_meta(args, do_augment=True):
    txt_path = f"code/data/index_list/{_index_dirname(args)}/session_{0 + 1}.txt"
    class_index = np.arange(args.base_class)
    if args.dataset == 'cifar100':
        trainset = args.Dataset.CIFAR100(root=args.data_folder, train=True, download=True,
                                         index=class_index, base_sess=True) #, do_augment=do_augment)
        testset = args.Dataset.CIFAR100(root=args.data_folder, train=False, download=False,
                                        index=class_index, base_sess=True)

    if args.dataset == 'miniimagenet':
        trainset = args.Dataset.MiniImageNet(root=args.data_folder, train=True,
                                             index_path=txt_path, do_augment=do_augment)
        testset = args.Dataset.MiniImageNet(root=args.data_folder, train=False,
                                            index=class_index)


    sampler = CategoriesSampler(trainset.targets, args.max_train_iter, args.num_ways_training,
                                 args.num_shots_training + args.num_query_training)

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_sampler=sampler, num_workers=args.num_workers,
                                              pin_memory=True)

    testloader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=args.batch_size_inference, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    return trainset, trainloader, testloader

def get_new_dataloader(args, session, do_augment=True):
    txt_path = f"code/data/index_list/{_index_dirname(args)}/session_{session + 1}.txt"
# def get_new_dataloader(args, session, do_augment=True):
#     # Load support set (don't do data augmentation here )
#     txt_path = "code/data/index_list/" + args.dataset + "/session_" + str(session + 1) + '.txt'
    if args.dataset == 'cifar100':
        class_index = open(txt_path).read().splitlines()
        trainset = args.Dataset.CIFAR100(root=args.data_folder, train=True, download=False,
                                         index=class_index, base_sess=False) #, do_augment=do_augment)
    elif args.dataset == 'miniimagenet':
        trainset = args.Dataset.MiniImageNet(root=args.data_folder, train=True,
                                       index_path=txt_path, do_augment=do_augment)

    # always load entire dataset in one batch    
    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=trainset.__len__() , shuffle=False,
                                                  num_workers=args.num_workers, pin_memory=True)

    # test on all encountered classes
    class_new = get_session_classes(args, session)
    if args.dataset == 'cifar100':
        testset = args.Dataset.CIFAR100(root=args.data_folder, train=False, download=False,
                                        index=class_new, base_sess=False)
    elif args.dataset == 'miniimagenet':
        testset = args.Dataset.MiniImageNet(root=args.data_folder, train=False,
                                      index=class_new)

    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=args.batch_size_inference, shuffle=False,
                                             num_workers=args.num_workers, pin_memory=True)

    return trainset, trainloader, testloader

def get_session_classes(args,session):
    class_list=np.arange(args.base_class + session * args.way)
    return class_list

class CutMixCollate:   ### => this is my custom class for apply cutmix to base train session
    def __init__(self, alpha: float = 1.0, prob: float = 0.5):
        self.alpha, self.prob = alpha, prob
        self.rng = np.random.default_rng()

    def __call__(self, batch):
        imgs, labels = list(zip(*batch))
        imgs   = torch.stack(imgs, 0)
        labels = torch.tensor(labels)

        if self.rng.random() < self.prob:
            lam = self.rng.beta(self.alpha, self.alpha)
            B, _, H, W = imgs.size()
            rnd_index  = torch.randperm(B)

            cut_rat = np.sqrt(1. - lam)
            cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
            cx,  cy  = self.rng.integers(W), self.rng.integers(H)
            x1, y1   = np.clip(cx - cut_w // 2, 0, W), np.clip(cy - cut_h // 2, 0, H)
            x2, y2   = np.clip(cx + cut_w // 2, 0, W), np.clip(cy + cut_h // 2, 0, H)

            imgs[:, :, y1:y2, x1:x2] = imgs[rnd_index, :, y1:y2, x1:x2]
            targets = (labels, labels[rnd_index], lam)
        else:
            targets = (labels, labels, 1.0)

        return imgs, targets