import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import os


def group_lasso_regularization(weight):
    """
    Penalization function of input layer weight
    :param weight: tensor of neural network input layer weights
    :return: the penalization to be used in the loss
    """
    return torch.sum(torch.norm(torch.norm(weight, dim=2), dim=0))


def proximal_update(network, lmbda, lr):
    """
    Performs update of the input layer of network with proximal gradient descent
    :param network: neural network
    :param lmbda: penalization coefficient for the group Lasso
    :param lr: learning rate
    :return: in-place operation
    """
    
    W = network.layers[0].weight
    hidden, p, lag, _ = W.shape
    norm = torch.norm(torch.norm(W, dim=0, keepdim=True), dim=2, keepdim=True)
    W.data = ((W / torch.clamp(norm, min=(lr * lmbda * 0.1)))
              * torch.clamp(norm - (lr * lmbda), min=0.0))


def train_causality_epoch(epoch, model, optimizer, dataloader, args, validation=False):

    t = time.time()
    loss_train, ridge_train, GL_train = [], [], []
    for batch_idx, data in enumerate(dataloader):
        Step = 'Validation '
        model.zero_grad()

        data = data[0].to(args.device)

        x_hat = model(data[:, :, :-1])

        loss = ((x_hat - data[:, :, args.lag:]) ** 2).sum() / data.size(0)

        R_reg = 0
        for i, m in enumerate(model.model_list):
            R_reg += torch.sum(torch.stack([torch.norm(l.weight, p=2)
                                            for i, l in enumerate(m.layers)
                                            if type(l) != type(m.activation)]))
        GL_reg = torch.sum(torch.stack([group_lasso_regularization(model.model_list[i].layers[0].weight)
                                        for i in range(len(model.model_list))]))

        if not validation:
            (loss + args.lmbd_prox * GL_reg + args.lmbd_ridge * R_reg).backward()
            optimizer.step()
            Step = 'Training '

        loss_train.append(loss.item())
        ridge_train.append(R_reg.item())
        GL_train.append(GL_reg.item())

    print(Step + 'epoch: {:04d}'.format(epoch),
          'loss: {:.5f}'.format(np.mean(loss_train)),
          'Ridge: {:.5f}'.format(np.mean(ridge_train)),
          'GL: {:.5f}'.format(np.mean(GL_train)),
          'Time: {:.4f}s'.format(time.time() - t))


def train_proximal_epoch(model, data, lr, lmbda, lmbda_ridge, max_iter=1):
    """

    :param model:
    :param data:
    :param lr:
    :param lmbda:
    :param lmbda_ridge:
    :param max_iter:
    :return:
    """
    x_hat = model(data[:, :, :-1])
    mse = ((x_hat - data[:, :, model.lag:]) ** 2).sum() / data.size(0)

    R_reg = 0
    for i, m in enumerate(model.model_list):
        R_reg += torch.sum(torch.stack([torch.norm(l.weight, p=2)
                                        for i, l in enumerate(m.layers)
                                        if type(l) != type(m.activation)]))
    loss = mse + lmbda_ridge * R_reg
    loss.backward()

    for it in range(max_iter):
        for m in model.model_list:
            proximal_update(m, lmbda, lr)  # proximal_update(net_copy, lmbda, lr, penalty)
            m.zero_grad()

    GL_reg = torch.sum(torch.stack([group_lasso_regularization(model.model_list[i].layers[0].weight)
                                    for i in range(len(model.model_list))]))

    return mse.item(), R_reg.item(), GL_reg.item()


def group_lasso_training(model, optimizer, dataloader, args, plot=False):
    """
    This function does the training of GL-VAR with proximal gradient under group lasso penalization

    :param model: function g of the GL-VAR
    :param optimizer: optimization module from PyTorch for training the model with respect to a loss
    :param dataloader: loader of data samples
    :param args: set of arguments for initialization and training of GL-VAR
    :return: save weights
    """
    s = 0.95
    epoch = 0

    while s > args.sparsity_min:
        epoch += 1
        for batch_idx, data in enumerate(dataloader):
            data = data[0].to(args.device)

            optimizer.zero_grad()
            mse, ridge, reg = train_proximal_epoch(model, data, args.lr,
                                                   args.lmbd_prox, args.lmbd_ridge)

            W = torch.stack([m.layers[0].weight.abs().mean(0).mean(1).mean(-1).cpu().detach()
                             for m in model.model_list])
            W = (W > 0).float().numpy()

        c = W.mean()

        if c <= s:
            torch.save(model.state_dict(), os.getcwd() + '/weights' + args.suffix + '/'
                       + 'g' + args.suffix + '_' + str(np.round(s, 2)) + '.pt')
            s = np.round(s - 0.05, 2)

        if epoch % 25 == 0:

            print('Epoch: {:04d}'.format(epoch), 'mse: {:.5f}'.format(mse), 'ridge: {:.5f}'.format(ridge),
                  'reg: {:.5f}'.format(reg), 'sparsity: {:.2f}'.format(1-c))

            if plot:
                plt.imshow(W, cmap='binary')
                plt.show(block=False)

    # Fine tuning of the sparse parameters

    print('Fine tuning for saved weights')

    all_losses, loss_train = [], []
    model.load_state_dict(torch.load(os.getcwd() + '/weights' + args.suffix + '/' + 'g' + args.suffix + '.pt'))

    for batch_idx, data in enumerate(dataloader):

        data = data[0].to(args.device)
        x_hat = model(data[:, :, :-1])
        loss = ((x_hat - data[:, :, args.lag:]) ** 2).mean(0).mean()  # F.mse_loss(x_hat, data[:, :, args.lag:])
        loss_train.append(loss.item())

    print('Sparsity: {:.2f}, loss: {:.5f}'.format(0.0, np.mean(loss_train)))
    all_losses.append(np.mean(loss_train))

    for s in np.arange(0.95, args.sparsity_min, -0.05):
        model.load_state_dict(torch.load(os.getcwd() + '/weights' + args.suffix + '/'
                                         + 'g' + args.suffix + '_' + str(np.round(s, 2)) + '.pt'))

        W_prior = torch.stack([m.layers[0].weight.squeeze(-1) for m in model.model_list])
        adjacency = (W_prior.sum(-1).sum(1) != 0).float().unsqueeze(1).unsqueeze(-1)

        if model.bias:
            bias_prior = torch.stack([m.layers[0].bias.data for m in model.model_list]).unsqueeze(0).unsqueeze(-1)
        else:
            bias_prior = torch.zeros([1, args.num_atoms, model.model_list[0].layers[0].weight.shape[0], 1]).to(args.device)

        w_fine_tuning = W_fine_tuning(W_prior, bias_prior).to(args.device)
        optimizer = optim.Adam(w_fine_tuning.parameters())
        loss = w_fine_tuning.train_epochs(100, dataloader, model, optimizer, adjacency, args)

        print('Sparsity: {:.2f}, loss: {:.4f}'.format(1-s, loss))
        all_losses.append(loss)

        W_adjusted_prior = w_fine_tuning.W_ft.mul(adjacency) + W_prior

        for i, m in enumerate(model.model_list):
            m.layers[0].weight.data = W_adjusted_prior[i].unsqueeze(-1)

        torch.save(model.state_dict(), os.getcwd() + '/weights' + args.suffix + '/'
                   + 'g' + args.suffix + '_' + str(np.round(s, 2)) + '.pt')

    plt.figure(figsize=(8, 2))
    plt.plot(np.arange(0, len(all_losses)*5, 5), np.array(all_losses), 'o-')
    if args.suffix == '_cmapss_001':
        plt.vlines(75, min(all_losses)*0.99, max(all_losses)*1.01)
    elif args.suffix == '_springs10':
        plt.vlines(55, min(all_losses) * 0.99, max(all_losses) * 1.01)
    plt.xlabel('Sparsity (%)')
    plt.ylabel('Mean-squared error')
    plt.show(block=False)


class W_fine_tuning(nn.Module):
    def __init__(self, w_prior, bias):
        super(W_fine_tuning, self).__init__()

        self.W_ft = nn.Parameter(torch.zeros_like(w_prior))
        self.w_prior = w_prior
        self.bias = nn.Parameter(bias, requires_grad=False)
        nn.init.xavier_normal_(self.W_ft)

    def train_epoch(self, dataloader, model, optimizer, adjacency, args):
        loss_train, reg_train = [], []
        for batch_idx, data in enumerate(dataloader):
            optimizer.zero_grad()

            data = data[0].to(args.device)

            W = (self.W_ft.mul(adjacency) + self.w_prior).unsqueeze(0).repeat(data.size(0), 1, 1, 1, 1)

            preds = 0
            for t in range(args.lag):
                preds += torch.stack([torch.bmm(W[:, i, :, :, t],
                                                data[:, :, t:data.size(2) - args.lag + t].view(data.size(0),
                                                                                               data.size(1), -1))
                                      for i, m in enumerate(model.model_list)], dim=1)
            preds = preds + self.bias
            if len(args.hidden_GC) > 0:
                preds = preds.contiguous().view(data.size(0), data.size(1), args.first_hidden, -1, args.in_dim)
            else:
                preds = preds.contiguous().view(data.size(0), data.size(1), 1, -1, args.in_dim)
            preds = torch.cat([m(preds[:, i], 2) for i, m in enumerate(model.model_list)], dim=1)

            target = data[:, :, args.lag:]
            loss = ((preds - target) ** 2).mean(0).mean()

            loss.backward()
            optimizer.step()

            loss_train.append(loss.item())

        return np.mean(loss_train)

    def train_epochs(self, epochs, dataloader, model, optimizer, adjacency, args):

        for epoch in range(epochs):
            loss = self.train_epoch(dataloader, model, optimizer, adjacency, args)

        return loss


def train_seq2graph_epoch(epoch, encoder, decoder, dataloader, optimizer, scheduler,
                          eta, rel_rec, rel_send, args):
    w_prior = torch.stack([m.layers[0].weight.squeeze(-1) for m in decoder.model_list])
    adjacency = (w_prior.sum(-1).sum(1) != 0).float().unsqueeze(1).unsqueeze(-1)
    if args.self_loops is False:
        adjacency[range(args.num_atoms), :, range(args.num_atoms), :] = 0

    if decoder.bias:
        bias_prior = torch.stack([m.layers[0].bias.data for m in decoder.model_list]).unsqueeze(0).unsqueeze(-1)
    else:
        bias_prior = torch.zeros([1, args.num_atoms, decoder.model_list[0].layers[0].weight.shape[0], 1])

    loss_train, reg_train = [], []
    for batch_idx, data in enumerate(dataloader):
        optimizer.zero_grad()

        data = data[0].to(args.device)

        w = encoder(data, rel_rec, rel_send)
        w = w.contiguous().view(-1, args.num_atoms, args.num_atoms, args.first_hidden, args.lag)
        w = w.permute(0, 1, 3, 2, 4).mul(adjacency.unsqueeze(0))
        w_sparse = w + w_prior.unsqueeze(0)

        preds = 0
        for t in range(args.lag):
            preds += torch.stack([torch.bmm(w_sparse[:, i, :, :, t],
                                            data[:, :, t:data.size(2) - args.lag + t].view(data.size(0),
                                                                                           data.size(1), -1))
                                  for i, m in enumerate(decoder.model_list)], dim=1)
        preds = preds + bias_prior

        if len(args.hidden_GC) > 0:
            preds = preds.contiguous().view(data.size(0), data.size(1), args.first_hidden, -1, args.in_dim)
            preds = torch.cat([m(preds[:, i], 1) for i, m in enumerate(decoder.model_list)], dim=1)
            target = data[:, :, args.lag:]
        else:
            preds = preds.squeeze()
            target = data[:, :, args.lag:].view(data.size(0), data.size(1), -1)

        loss = ((preds - target) ** 2).mean(0).sum()
        reg = torch.stack([torch.norm(w - encoder.P_prior.unsqueeze(0)) for w in w]).sum() + torch.norm(encoder.P_prior)

        (loss + eta * reg).backward()

        optimizer.step()

        loss_train.append(loss.item())
        reg_train.append(reg.item())

    scheduler.step()

    if epoch % 25 == 0:
        print('Epoch: {:04d}'.format(epoch),
              'loss: {:.5f}'.format(np.mean(loss_train)),
              'reg: {:.5f}'.format(np.mean(reg_train)))
