# -*- coding: utf-8 -*-
"""
## The setup of Project
"""
import sys
import os
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import random
import argparse

import models as model_

"""Load Data"""
def DataInput(data_path):
  arr_df = np.load(data_path)
  X = arr_df[:,:-1]
  Y_onehot = arr_df[:,-1] - 1
  arr_X = X.reshape(X.shape[0],-1,240)
  arr_X = arr_X.swapaxes(1,2)
  train_index = random.sample(range(arr_X.shape[0]), int(arr_X.shape[0]*0.8))
  test_index = [i for i in range(arr_X.shape[0]) if i not in train_index]
  return arr_X, Y_onehot, train_index,test_index

"""## training"""
def train(dataloader, model, loss_fn, optimizer, epoch, final_epoch):
    size = len(dataloader.dataset)
    train_acc = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        train_acc += (pred.argmax(1) == y).type(torch.float).sum().item()
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if epoch == final_epoch:
      print(f"epoch: {epoch}--Train accuracy is: {(100*train_acc/size):>0.1f}%")
    return train_acc/size

def test(dataloader, model, loss_fn, epoch, final_epoch):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    if epoch == final_epoch:
      test_loss /= num_batches
      correct /= size
      print(f"Test Error: Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}")
    return correct

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
  loss_fn = nn.CrossEntropyLoss()
  parser = argparse.ArgumentParser()
  parser.add_argument('--batchsize', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
  parser.add_argument('--epochs', type=int, default=45, metavar='N', help='number of epochs to train (default: 45)')
  parser.add_argument('--path', type=str, help='where data is stored')
  args = parser.parse_args()
  arr_X, Y_onehot, train_index, test_index = DataInput(args.path)
  for i in range(4):
    if i == 0:
      # multi-channel (abs): num*1*240*120
      data_x = np.zeros([arr_X.shape[0],1,arr_X.shape[1],int(arr_X.shape[2]/2)])
      data_x[:,0,:,:] =  abs(arr_X[:,:,:int(arr_X.shape[2]/2)] + 1j*arr_X[:,:,int(arr_X.shape[2]/2):])
      print("----Tradtional CNN: abs only (real number)-----")
    if i == 1:
      # multi-channel (abs&phase): num*2*240*120
      data_x = np.zeros([arr_X.shape[0],2,arr_X.shape[1],int(arr_X.shape[2]/2)])
      data_x[:,0,:,:] =  np.abs(arr_X[:,:,:int(arr_X.shape[2]/2)] + 1j*arr_X[:,:,int(arr_X.shape[2]/2):])
      data_x[:,1,:,:] =  np.angle(arr_X[:,:,:int(arr_X.shape[2]/2)] + 1j*arr_X[:,:,int(arr_X.shape[2]/2):])
      print("----CVNN Multichannel: Abs and phase----")
    if i == 2:
      # multi-channel (real & imag): num*2*240*120
      data_x = np.zeros([arr_X.shape[0],2,arr_X.shape[1],int(arr_X.shape[2]/2)])
      data_x[:,0,:,:] =  arr_X[:,:,:int(arr_X.shape[2]/2)]
      data_x[:,1,:,:] =  arr_X[:,:,int(arr_X.shape[2]/2):]
      print("----CVNN Multichannel: Real and imaginary----")
    if i == 3:
      # DCN
      data_x = np.zeros([arr_X.shape[0],2,1, arr_X.shape[1], int(arr_X.shape[2]/2)])
      data_x[:,0,:,:,:] = arr_X[:,:,:int(arr_X.shape[2]/2)].reshape(data_x[:, 0,...].shape) #X_real
      data_x[:,1,:,:,:] = arr_X[:,:,int(arr_X.shape[2]/2):].reshape(data_x[:, 0,...].shape) #X_imag
      print("----CVNN DCN (Deep complex networks)----")
    Xtrain = data_x[train_index,...]
    Ytrain = Y_onehot[train_index,...]
    Xtest = data_x[test_index,...]
    Ytest = Y_onehot[test_index,...]
    data_train = torch.utils.data.TensorDataset(torch.from_numpy(Xtrain).type(torch.FloatTensor), torch.from_numpy (Ytrain).type(torch.LongTensor))
    data_test = torch.utils.data.TensorDataset(torch.from_numpy(Xtest).type(torch.FloatTensor), torch.from_numpy (Ytest).type(torch.LongTensor))
    # Create data loaders.
    train_dataloader = DataLoader(data_train, batch_size = args.batchsize)
    test_dataloader = DataLoader(data_test, batch_size = args.batchsize)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for k in range(3):
      if k == 0:
        print(f"model is plain shallow CNN (one Building Layer)")
        if i == 0:
          model = model_.shallow_Net().to(device)
        else:
          if i == 3:
            model = model_.shallow_Complex_Net().to(device)
          else:
            model = model_.shallow_ch2_Net().to(device)
      if k == 1:
        print(f"model is plain deep CNN (five Building Layers)")
        if i == 0:
          model = model_.deep_Net().to(device)
        else:
          if i == 3:
            model = model_.deep_Complex_Net().to(device)
          else:
            model = model_.deep_ch2_Net().to(device)
      if k == 2:
        print(f"model is ResNet18")
        if i == 0:
          model = model_.ResNet18(in_channels = 1).to(device)
        else:
          if i == 3:
            model = model_.ResNet_complex().to(device)
          else:
            model = model_.ResNet18(in_channels = 2).to(device)
      optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
      for t in range(args.epochs):
        lr = 1e-4
        if t > 30:
          if t < 40:
            lr = 1e-5
          else:
            lr = 1e-6*5
        adjust_learning_rate(optimizer, lr)
        train(train_dataloader, model, loss_fn, optimizer, t, args.epochs-1)
        test(test_dataloader, model, loss_fn, t, args.epochs-1)
      print("Done!")
      
if __name__ == '__main__':
    main()
