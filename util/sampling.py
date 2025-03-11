#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
from argparse import Namespace

import numpy as np
from torch.utils.data import DataLoader


def split_iid_data(args: Namespace, dataset: DataLoader):
    """从数据集中分割IID数据给各客户端"""
    num_items = int(len(dataset)/args.num_users)
    dict_users, all_idxes = {}, [i for i in range(len(dataset))]
    for i in range(args.num_users):
        dict_users[i] = set(np.random.choice(all_idxes, num_items, replace=False))
        all_idxes = list(set(all_idxes) - dict_users[i])
    return dict_users
