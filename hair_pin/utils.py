#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import numpy as np
import copy
import torch
import pandas as pd
from sklearn import preprocessing
from options import args_parser

def get_dataset():
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """
    fp = 'Data/'
    data_a_in = pd.read_csv(fp + "hairpin_BPF_3th_gap_combined-2f.csv", encoding="gbk")
    data_b_in = pd.read_csv(fp + "hairpin_BPF_7th_gap_combined-2f.csv", encoding="gbk")

    args = args_parser()
  
    # 训练数据集 Training dataset sampling
    N = args.SampleRate
    data_a = data_a_in.sample(frac=min(N, 1.0), random_state=1)
    data_b = data_b_in.sample(frac=min(N, 1.0), random_state=1)

    
    # 测试数据集 Rest for testing
    test_a = data_a_in.append(data_a).drop_duplicates(keep=False) 
    test_b = data_b_in.append(data_b).drop_duplicates(keep=False)
    
    
    # 数据集标准化 Data normalization
    scale_a = preprocessing.StandardScaler()
    data_a_norm = scale_a.fit_transform(data_a_in)
    data_b_norm = scale_a.fit_transform(data_b_in)
    test_a_norm = scale_a.fit_transform(test_a)
    test_b_norm = scale_a.fit_transform(test_b)
    
    col_name = data_a_in.columns
    data_a = pd.DataFrame(data_a_norm, columns=col_name)
    data_b = pd.DataFrame(data_b_norm, columns=col_name)
    test_a = pd.DataFrame(test_a_norm, columns=col_name)
    test_b = pd.DataFrame(test_b_norm, columns=col_name)
    

    # 共享数据集 Ratio of shared data for encrypted FL algorithm 
    rho = args.ratio
    data_a_share = data_a.sample(frac=rho, random_state=1)
    data_b_share = data_b.sample(frac=rho, random_state=1)


    # 共享数据集加噪 Adding AWGN noise
    with_noise = True
    if with_noise:
        mu_a, sigma_a = 0, 0.001
        mu_b, sigma_b = 0, 0.001
        # creating a noise with the same dimension as the dataset
        noise_a = np.random.normal(mu_a, sigma_a, data_a_share.shape)
        noise_b = np.random.normal(mu_b, sigma_b, data_b_share.shape)
        data_a_share += noise_a
        data_b_share += noise_b
    
    
    # 打标签 Label the heterogeneous data
    data_a['NUM'] =3* np.ones(len(data_a))
    data_a_share['NUM'] =3* np.ones(len(data_a_share))   
    data_b['NUM'] = 7*np.ones(len(data_b))
    data_b_share['NUM'] = 7*np.ones(len(data_b_share))

    
    # 分享数据 Share the noised data
    with_share = True
    if with_share:
        data_a = pd.concat([data_a, data_b_share], axis=0).sample(frac=1.0, random_state=1)
        data_b = pd.concat([data_b, data_a_share], axis=0).sample(frac=1.0, random_state=1)
 

    # 选择测试数据 Test dataset selection
    if bool(args.test_unexampled):
        print("Test model with the unexampled 5th order filter data")
        # 测试数据使用5阶 "1" means unexampled data
        test = pd.read_csv(fp + "hairpin_BPF_5th_gap_combined-2f.csv", encoding="gbk")
        test['NUM'] = 5*np.ones(len(test))
        test = test.sample(frac=1, random_state=1)
        n = test.shape[0]
        test_a = test.iloc[:int(n / 2)]
        test_b = test.iloc[int(n / 2):]
    else:
        print("Test model with the exampled 3th and 7th order filter")
        # 测试数据使用3，7阶打乱版，数据打标签，混合, 分给A和B, 打乱
        # "0" mean exampled data
        test_a['NUM'] = 3*np.ones(len(test_a))
        test_b['NUM'] = 7*np.ones(len(test_b))
        test_a_2 = pd.concat([test_a, test_b], axis=0).sample(frac=0.5, random_state=1)
        test_b_2 = pd.concat([test_a, test_b, test_a_2], axis=0).drop_duplicates(keep=False)
        test_a = test_a_2.sample(frac=1)
        test_b = test_b_2.sample(frac=1)
    
    
    col_name = data_a_in.columns
    data_a = pd.DataFrame(data_a, columns=col_name)
    data_b = pd.DataFrame(data_b, columns=col_name)
    test_a = pd.DataFrame(test_a, columns=col_name)
    test_b = pd.DataFrame(test_b, columns=col_name)
    
    data_a = data_a.reset_index(drop=True)
    data_b = data_b.reset_index(drop=True)
    test_a = test_a.reset_index(drop=True)
    test_b = test_b.reset_index(drop=True)

    train = {'A': data_a, 'B': data_b}
    test = {'A': test_a, 'B': test_b}

    return train, test, scale_a


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')
    print('    Federated parameters:')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return
