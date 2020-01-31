import numpy as np
import torch
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader
import pandas as pd


def load_springs_data(path, batch_size=1, dataset_name='', shuffle=True):

    """

    :param path: directory of data
    :param batch_size: size of the batch in dataloaders
    :param dataset_name: name of the dataset
    :param shuffle: shuffle the samples in dataloaders
    :return:
    """
    print(path, dataset_name)

    loc_train = np.load(path + 'loc_train' + dataset_name + '.npy')
    label_train = np.load(path + 'label_train' + dataset_name + '.npy')
    edges_train = np.load(path + 'edges_train' + dataset_name + '.npy')

    loc_valid = np.load(path + 'loc_valid' + dataset_name + '.npy')
    label_valid = np.load(path + 'label_valid' + dataset_name + '.npy')
    edges_valid = np.load(path + 'edges_valid' + dataset_name + '.npy')

    loc_test = np.load(path + 'loc_test' + dataset_name + '.npy')
    label_test = np.load(path + 'label_test' + dataset_name + '.npy')
    edges_test = np.load(path + 'edges_test' + dataset_name + '.npy')

    # [num_samples, num_time_steps, num_dims, num_atoms]

    loc_max = loc_train.max()
    loc_min = loc_train.min()

    # Normalize to [-1, 1]
    loc_train = (loc_train - loc_min) * 2 / (loc_max - loc_min) - 1
    loc_valid = (loc_valid - loc_min) * 2 / (loc_max - loc_min) - 1
    loc_test = (loc_test - loc_min) * 2 / (loc_max - loc_min) - 1

    # Reshape to: [num_sims, num_atoms, num_time_steps, num_dims]
    loc_train = np.transpose(loc_train, [0, 3, 1, 2])
    loc_valid = np.transpose(loc_valid, [0, 3, 1, 2])
    loc_test = np.transpose(loc_test, [0, 3, 1, 2])

    edges_train = torch.FloatTensor(edges_train)
    edges_valid = torch.FloatTensor(edges_valid)
    edges_test = torch.FloatTensor(edges_test)

    feat_train = torch.FloatTensor(loc_train)
    feat_valid = torch.FloatTensor(loc_valid)
    feat_test = torch.FloatTensor(loc_test)

    label_train = torch.Tensor(label_train)
    label_valid = torch.Tensor(label_valid)
    label_test = torch.Tensor(label_test)

    train_data = TensorDataset(feat_train, label_train, edges_train)
    valid_data = TensorDataset(feat_valid, label_valid, edges_valid)
    test_data = TensorDataset(feat_test, label_test, edges_test)

    train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
    valid_data_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
    test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_data_loader, valid_data_loader, test_data_loader


def create_loaders(X, y, batch_size, shuffle=True):
    """

    :param X: Data
    :param y: Label
    :param batch_size: size of the batch in dataloaders
    :param shuffle: shuffle the samples in dataloaders
    :return:
    """
    feat = torch.FloatTensor(X).unsqueeze(-1)
    label = torch.Tensor(y)

    data = TensorDataset(feat, label)
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle)

    return data_loader


def load_cmapss_data(path, batch_size=1, time_steps=25, shuffle=True):
    """

    :param path:
    :param batch_size: batch_size: size of the batch in dataloaders
    :param time_steps: length of the samples
    :param shuffle: shuffle the trainloader
    :return:
    """

    X_cmapss = pd.read_csv(path + 'train_FD001.txt', sep=' ', header=None, index_col=1)
    X_cmapss.dropna(axis=1, inplace=True)

    # Variables 2, 3, 4 are the exogenous conditions

    nonconstant_variables = [6, 7, 8, 11, 12, 13, 15, 16, 18, 19, 21, 24, 25]

    X_cmapss = X_cmapss[[0] + nonconstant_variables]
    M = X_cmapss[nonconstant_variables].max()
    m = X_cmapss[nonconstant_variables].min()

    X_cmapss[nonconstant_variables] = (X_cmapss[nonconstant_variables] - m) / (M - m)

    X_train, y_train = [], []

    for i, ind in enumerate(np.unique(X_cmapss[0])[:60]):
        x = np.stack([X_cmapss.loc[X_cmapss[0] == ind][nonconstant_variables].values[i:i + time_steps].T
                      for i in range(0, len(X_cmapss.loc[X_cmapss[0] == ind]) - time_steps, 5)])
        X_train.append(x)
        y_train.append(np.arange(len(x)))

    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)

    X_valid, y_valid = [], []

    for i, ind in enumerate(np.unique(X_cmapss[0])[60:75]):
        x = np.stack([X_cmapss.loc[X_cmapss[0] == ind][nonconstant_variables].values[i:i + time_steps].T
                      for i in range(0, len(X_cmapss.loc[X_cmapss[0] == ind]) - time_steps, 5)])
        X_valid.append(x)
        y_valid.append(np.arange(len(x)))

    X_valid = np.concatenate(X_valid)
    y_valid = np.concatenate(y_valid)

    X_test, y_test = [], []

    for i, ind in enumerate(np.unique(X_cmapss[0])[75:]):
        x = np.stack([X_cmapss.loc[X_cmapss[0] == ind][nonconstant_variables].values[i:i + time_steps].T
                      for i in range(0, len(X_cmapss.loc[X_cmapss[0] == ind]) - time_steps, 5)])
        X_test.append(x)
        y_test.append(np.arange(len(x)))

    X_test = np.concatenate(X_test)
    y_test = np.concatenate(y_test)

    train_loader = create_loaders(X_train, y_train, batch_size, shuffle)
    valid_loader = create_loaders(X_valid, y_valid, batch_size, False)
    test_loader = create_loaders(X_test, y_test, batch_size, False)

    return train_loader, valid_loader, test_loader
