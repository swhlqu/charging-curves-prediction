#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from torch import optim
import time
import warnings
from torch.utils.data import Dataset, DataLoader, TensorDataset
import math
from pathlib import Path
import os

warnings.filterwarnings('ignore')

class Dataset_generation(Dataset):
    def __init__(self, path, batch_size):
        self.path = path
        self.batch_size = batch_size
        self.x_train = None
        self.x_valid = None
        self.x_test = None
        self.y_train = None
        self.y_valid = None
        self.y_test = None

    def load_npy(self):

        self.x_train = np.load(self.path + "x_train.npy")
        self.x_valid = np.load(self.path + "x_valid.npy")
        self.x_test = np.load(self.path + "x_test.npy")
        self.y_train = np.load(self.path + "y_train.npy")
        self.y_valid = np.load(self.path + "y_valid.npy")
        self.y_test = np.load(self.path + "y_test.npy")


    def return_dataloader(self):
        self.load_npy()
        print(self.x_train.shape, self.y_train.shape,
              self.x_valid.shape, self.y_valid.shape,
              self.x_test.shape, self.y_test.shape)
        train = []
        valid = []
        test = []
        for i in range(51):
            x_train = self.x_train[:, :, i].reshape(-1, 10)
            y_train = self.y_train[:, :, i].reshape(-1, 10)

            x_valid = self.x_valid[:, :, i].reshape(-1, 10)
            y_valid = self.y_valid[:, :, i].reshape(-1, 10)

            x_test = self.x_test[:, :, i].reshape(-1, 10)
            y_test = self.y_test[:, :, i].reshape(-1, 10)
            train_set = TensorDataset(torch.from_numpy(x_train).to(torch.float32),
                                      torch.from_numpy(y_train).to(torch.float32))
            valid_set = TensorDataset(torch.from_numpy(x_valid).to(torch.float32),
                                      torch.from_numpy(y_valid).to(torch.float32))
            test_set = TensorDataset(torch.from_numpy(x_test).to(torch.float32),
                                     torch.from_numpy(y_test).to(torch.float32))

            train_loader = DataLoader(train_set, batch_size=self.batch_size)
            valid_loader = DataLoader(valid_set, batch_size=self.batch_size)
            test_loader = DataLoader(test_set, batch_size=self.batch_size)

            train.append(train_loader)
            valid.append(valid_loader)
            test.append(test_loader)


        return train, valid, test


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe


def adjust_learning_rate(optimizer, epoch, learning_rate, lradj):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if lradj == 'type1':
        lr_adjust = {epoch: learning_rate * (0.9 ** ((epoch - 1) // 50))}
    elif lradj == 'type2':
        lr_adjust = {
            1000: 5e-4, 2000: 1e-4, 3000: 5e-5, 4000: 1e-5,
            5000: 5e-6, 6000: 1e-6
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.dropout = nn.Dropout(0.1)
        self.linear_layer1 = nn.Linear(10, 512)
        self.linear_layer2 = nn.Linear(512, 512)
        self.final_layer = nn.Linear(512, 10)
        self.mu_activation = nn.ELU()

    def forward(self, inputs):

        x = self.mu_activation(self.linear_layer1(inputs))
        x = self.mu_activation(self.linear_layer2(x))
        out = self.final_layer(x)

        return out

class Exp():
    def __init__(self, n, patience):
        super(Exp, self).__init__()
        self.patience = patience
        self.n = n
        self.learning_rate = 0.001
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = MyModel().to(self.device)
        self.checkpoints_path = './checkpoint/MHDNN-' + str(self.n)

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        for i, (batch_enc, batch_output) in enumerate(vali_loader):
            batch_prediction, batch_output = self._process_one_batch(batch_enc, batch_output)
            loss = criterion(batch_prediction.detach().cpu(), batch_output.detach().cpu())
            total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, train_loader, vali_loader, test_loader, train_epochs):
        if not os.path.exists(self.checkpoints_path):
            os.makedirs(self.checkpoints_path)
        my_file = Path(self.checkpoints_path + '/' + 'checkpoint.pth')
        if my_file.exists():
            best_model_path = self.checkpoints_path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))
            print('have loader best model')

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(self.patience, verbose=True)
        model_optim = self._select_optimizer()  # adam,learning_rate
        criterion = self._select_criterion()  # mse
        for epoch in range(train_epochs):
            iter_count = 0
            train_loss = []
            epoch_time = time.time()
            self.model.train()
            for i, (batch_enc, batch_output) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_prediction, batch_output = self._process_one_batch(batch_enc, batch_output)
                loss = criterion(batch_prediction, batch_output)
                train_loss.append(loss.item())
                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_loader, criterion)
            test_loss = self.vali(test_loader, criterion)
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, self.checkpoints_path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch + 1, self.learning_rate, lradj='type2')

        best_model_path = self.checkpoints_path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, test_loader):
        my_file = Path(self.checkpoints_path + '/' + 'checkpoint.pth')
        if my_file.exists():
            best_model_path = self.checkpoints_path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))
            print('have loader best model')
        self.model.eval()
        preds = []
        trues = []
        for i, (batch_enc, batch_output) in enumerate(test_loader):
            batch_prediction, batch_output = self._process_one_batch(batch_enc, batch_output)
            preds.append(batch_prediction.detach().cpu().numpy())
            trues.append(batch_output.detach().cpu().numpy())

        preds_last = preds.pop(-1)
        trues_last = trues.pop(-1)

        preds = np.array(preds)
        trues = np.array(trues)
        preds = preds.reshape(-1, 10)
        trues = trues.reshape(-1, 10)

        preds_last = np.array(preds_last).reshape(-1, 10)
        trues_last = np.array(trues_last).reshape(-1, 10)

        preds = np.concatenate((preds, preds_last), axis=0)
        trues = np.concatenate((trues, trues_last), axis=0)

        print('test shape:', preds.shape, trues.shape)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        pd.DataFrame(preds).to_csv('DNN_preds.csv', header=False, index=False)
        pd.DataFrame(trues).to_csv('DNN_trues.csv', header=False, index=False)

        return None

    def predict(self, test_loader, n):
        my_file = Path(self.checkpoints_path + '/' + 'checkpoint.pth')
        if my_file.exists():
            best_model_path = self.checkpoints_path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))
            print('have loader best model')
        self.model.eval()
        preds = []
        trues = []
        for i, (batch_enc, batch_output) in enumerate(test_loader):
            batch_prediction, batch_output = self._process_one_batch(batch_enc, batch_output)
            preds.append(batch_prediction.detach().cpu().numpy())
            trues.append(batch_output.detach().cpu().numpy())

        preds_last = preds.pop(-1)
        trues_last = trues.pop(-1)

        preds = np.array(preds)
        trues = np.array(trues)
        preds = preds.reshape(-1, 10)
        trues = trues.reshape(-1, 10)

        preds_last = np.array(preds_last).reshape(-1, 10)
        trues_last = np.array(trues_last).reshape(-1, 10)

        preds = np.concatenate((preds, preds_last), axis=0)
        trues = np.concatenate((trues, trues_last), axis=0)

        print('test shape:', preds.shape, trues.shape)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        pd.DataFrame(preds).to_csv(str(n) + '_DNN_preds.csv', header=False, index=False)
        pd.DataFrame(trues).to_csv(str(n) + '_DNN_trues.csv', header=False, index=False)

        return None

    def _process_one_batch(self, batch_enc , batch_output):
        batch_enc = batch_enc.float().to(self.device)
        batch_output = batch_output.float().to(self.device)
        batch_prediction = self.model(batch_enc)

        return batch_prediction, batch_output


mse_list = pd.DataFrame(np.zeros(shape = (1,1)))
mae_list = pd.DataFrame(np.zeros(shape = (1,1)))
rmse_list = pd.DataFrame(np.zeros(shape = (1,1)))
data_generation = Dataset_generation(
    path = "",
    batch_size=1024)
train, valid, test = data_generation.return_dataloader()
assert (len(train) == 51)

for n in range(1, 52):
    exp = Exp(n=n, patience=1000)
    print('>>>>>>>start training>>>>>>>>>>>>>>>>>>>>>>>>>>')
    exp.train(train[n-1], valid[n-1], test[n-1], train_epochs=2000000)
    torch.cuda.empty_cache()
    print('>>>>>>>start testing>>>>>>>>>>>>>>>>>>>>>>>>>>')
    exp.predict(test_loader = test[n-1], n = n)
