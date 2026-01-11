#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from sklearn import metrics
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import pandas as pd
from sklearn import preprocessing



class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]
        self.col_name = ["NUM", "W1", "W2", "W3", "L1", "L2", "L3", "L4", "L5", "S1"]


    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        feature = self.dataset.loc[self.idxs[item], self.col_name].values
        label = self.dataset.loc[
            self.idxs[item], self.dataset.columns[~self.dataset.columns.isin(self.col_name)]].values
        return torch.tensor(feature, dtype=torch.float), torch.tensor(label, dtype=torch.float)


class LocalUpdate(object):
    def __init__(self, args, dataset, logger):
        self.args = args
        self.logger = logger
        self.train_loader, self.val_loader = self.train_val_test(dataset)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mse = nn.MSELoss().to(self.device)

    def train_val_test(self, dataset):
        idxs = [i for i in range(len(dataset))]
        idxs_train = idxs[:int(0.8 * len(idxs))]
        idxs_val = idxs[int(0.8 * len(idxs)):]
        train_loader = DataLoader(DatasetSplit(dataset, idxs_train), batch_size=self.args.local_bs, shuffle=True)
        val_loader = DataLoader(DatasetSplit(dataset, idxs_val), batch_size=self.args.local_bs, shuffle=False)
        return train_loader, val_loader

    def update_weights(self, model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = []
        epoch_r2 = []
        r2_sum=[]

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        for iter in range(self.args.local_ep):
            batch_loss = []
            batch_r2 = []
            
            for batch_idx, (x, label) in enumerate(self.train_loader):
                x, label = x.to(self.device), label.to(self.device)

                model.zero_grad()
                pred = model(x)
                loss = self.mse(pred, label)
                loss.backward()
                optimizer.step()

                r2 = np.mean([metrics.r2_score(label.cpu().detach().numpy()[:, i], pred.cpu().detach().numpy()[:, i])
                              for i in range(pred.size(1))])

                # r2 = metrics.r2_score(label.detach().numpy().ravel(), pred.detach().numpy().ravel())

                if batch_idx % 100 == 0:
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({}%)]\tLoss: {:.6f}\tR2: {:.6f}'.format(
                        global_round + 1, iter + 1, str(batch_idx * len(x)).zfill(4),
                        len(self.train_loader.dataset),
                        str(10. * batch_idx / len(self.train_loader)).zfill(4), loss.item(), r2.item()))
                    r2_sum.append(r2)
                self.logger.add_scalar('loss', loss.item())
                self.logger.add_scalar('r2', r2.item())
                batch_loss.append(loss.item())
                batch_r2.append(r2.item())
                
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            epoch_r2.append(sum(batch_r2) / len(batch_r2))
            print(r2_sum)
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss), sum(epoch_r2) / len(epoch_r2), epoch_r2

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        list_r2 = []

        for batch_idx, (x, label) in enumerate(self.val_loader):
            x, label = x.to(self.device), label.to(self.device)

            # Inference
            pred = model(x)
            batch_loss = self.mse(pred, label)
            loss += batch_loss.item()

            # Prediction
            pred_type = calc_type(pred[:, 0])
            true_type = calc_type(label[:, 0])
            correct += torch.sum(torch.eq(pred_type, true_type)).item()
            total += len(label)

            # r2 += metrics.r2_score(label.detach().numpy().ravel(), out.detach().numpy().ravel())
            r2 = np.mean([metrics.r2_score(label.cpu().detach().numpy()[:, i], pred.cpu().detach().numpy()[:, i])
                          for i in range(pred.size(1))])
            list_r2.append(r2)

        accuracy = correct / total
        loss /= len(self.train_loader)
        return accuracy, loss, sum(list_r2) / len(list_r2)


def calc_type(pred):
    pred = torch.sigmoid(pred)
    pred[pred > 0.5] = 1
    pred[pred <= 0.5] = 0
    return pred.view(-1)


def test_inference(model, test):
    """ Returns the test accuracy and loss.
    """
   
    test_dataset = pd.concat([test['A'], test['B']], ignore_index=True)
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mse = nn.MSELoss().to(device)
    idxs = [i for i in range(len(test_dataset))]
    test_loader = DataLoader(DatasetSplit(test_dataset, idxs), batch_size=128, shuffle=False)

    preds, truth = [], []

    for batch_idx, (x, label) in enumerate(test_loader):
        x, label = x.to(device), label.to(device)

        # Inference
        pred = model(x)
    
        batch_loss = mse(pred, label)
        loss += batch_loss.item()

        preds.append(pred.detach().cpu())
        truth.append(label.detach().cpu())

        # Prediction
        pred_type = calc_type(pred[:, 0])
        correct += torch.sum(torch.eq(pred_type, label[:, 0])).item()
        total += len(label)

    accuracy = correct / total
    loss /= len(test_loader)
    final_preds = torch.cat(preds, dim=0)
    final_truth = torch.cat(truth, dim=0)


    testA1 = np.array(test['A'])
    testA2 = torch.tensor(testA1)
    indices=torch.tensor([0,1,2,3,4,5,6,7,8])
    testA3 = torch.index_select(testA2,1,indices)
    
    final_predsA=final_preds[:int(len(final_preds) / 2)]#.detach().numpy()
    final_truthA=final_truth[:int(len(final_truth) / 2)]#.detach().numpy()
    #x_A=torch.tensor(x[:int(len(x) / 2)].detach().cpu())
    fianl_PREall= torch.cat([testA3,final_predsA], dim=1)
    fianl_TRUall= torch.cat([testA3,final_truthA], dim=1)

    return accuracy, loss, final_preds, final_truth,fianl_PREall,fianl_TRUall,final_predsA,final_truthA,testA3
