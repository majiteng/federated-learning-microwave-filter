#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.hidden_1 = nn.Linear(dim_hidden, dim_out[0])
        self.hidden_2 = nn.Linear(dim_hidden, dim_out[1])
        self.hidden = nn.Linear(dim_hidden, dim_out)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        out_1 = self.hidden_1(x)
        out_2 = self.hidden_2(x)
        out_1 = self.sigmoid(out_1)
        # out = self.hidden(x)
        return out_1, out_2


class MLP1(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP1, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.hidden = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        out = self.hidden(x)
        return out

class MLP2(nn.Module):
  '''
    Multilayer Perceptron.
  '''
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Flatten(),
      nn.Linear(10, 1024),#10, 256
      nn.ReLU(),
      nn.Linear(1024, 1024),
      nn.ReLU(), 
      nn.Linear(1024, 2048),
      nn.ReLU(), 
      nn.Linear(2048, 2048),#1024 2048
      nn.ReLU(),
      nn.Linear(2048, 1024),#2048, 1024
      nn.ReLU(),
      nn.Linear(1024, 1024),
      nn.ReLU(),
      nn.Linear(1024, 82)#1024, 82
    )


  def forward(self, x):
    '''Forward pass'''
    return self.layers(x)