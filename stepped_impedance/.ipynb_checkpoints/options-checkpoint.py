#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--epochs', type=int, default=30,
                        help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=100,
                        help="number of users: K")#100
    parser.add_argument('--frac', type=float, default=0.1,
                        help='the fraction of clients: C')#0.1
    parser.add_argument('--local_ep', type=int, default=20,
                        help="the number of local epochs: E")#20
    parser.add_argument('--local_bs', type=int, default=100,
                        help="local batch size: B")#100
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')#0.001
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.5)')#0.9
    parser.add_argument('--ratio', type=float, default=0,
                        help='Ratio of shared data for encrypted FL algorithm (default: 0)')
    parser.add_argument('--SampleRate', type=float, default=1,
                        help='ratio of the data to be trained (default: 1)')
    parser.add_argument('--test_unexampled', type=float, default=1,
                        help='Test dataset selection, "1" means unexampled data, "0" mean exampled data (default: 1)') 
    # model arguments
    parser.add_argument('--model', type=str, default='mlp', help='model name')

    # other arguments
    parser.add_argument('--optimizer', type=str, default='adam', help="type \
                        of optimizer")
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    args = parser.parse_args()
    return args
