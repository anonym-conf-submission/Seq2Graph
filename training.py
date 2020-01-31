import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import time


def group_lasso_regularization(weight, type='group_lasso'):
    return torch.sum(torch.norm(torch.norm(weight, dim=2), dim=0))


def proximal_update(network, lam, lr, penalty):
    '''Perform in place proximal update on first layer weight matrix.

    Args:
      network: MLP network.
      lam: regularization parameter.
      lr: learning rate.
      penalty: one of group_lasso, GSGL, hierarchical.
    '''
    W = network.layers[0].weight
    hidden, p, lag, _ = W.shape
    if penalty == 'group_lasso':
        norm = torch.norm(torch.norm(W, dim=0, keepdim=True), dim=2, keepdim=True)
        W.data = ((W / torch.clamp(norm, min=(lr * lam * 0.1)))
                  * torch.clamp(norm - (lr * lam), min=0.0))
    elif penalty == 'GSGL':
        norm = torch.norm(W, dim=0, keepdim=True)
        W.data = ((W / torch.clamp(norm, min=(lr * lam * 0.1)))
                  * torch.clamp(norm - (lr * lam), min=0.0))
        norm = torch.norm(W, dim=(0, 2), keepdim=True)
        W.data = ((W / torch.clamp(norm, min=(lr * lam * 0.1)))
                  * torch.clamp(norm - (lr * lam), min=0.0))
    elif penalty == 'hierarchical':
        # Lowest indices along third axis touch most lagged values.
        for i in range(lag):
            norm = torch.norm(torch.norm(W[:, :, :(i + 1)], dim=2, keepdim=True), dim=0, keepdim=True)
            W.data[:, :, :(i + 1)] = (
                    (W.data[:, :, :(i + 1)] / torch.clamp(norm, min=(lr * lam * 0.1)))
                    * torch.clamp(norm - (lr * lam), min=0.0))
    else:
        raise ValueError('unsupported penalty: %s' % penalty)


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


def train_proximal_epoch(model, data, lr, lam, lam_ridge, penalty='group_lasso', max_iter=1):

    x_hat = model(data[:, :, :-1])
    mse = ((x_hat - data[:, :, model.lag:]) ** 2).sum() / data.size(0)

    R_reg = 0
    for i, m in enumerate(model.model_list):
        R_reg += torch.sum(torch.stack([torch.norm(l.weight, p=2)
                                        for i, l in enumerate(m.layers)
                                        if type(l) != type(m.activation)]))
    loss = mse + lam_ridge * R_reg
    loss.backward()

    for it in range(max_iter):
        for m in model.model_list:
            proximal_update(m, lam, lr, penalty)  # proximal_update(net_copy, lam, lr, penalty)
            m.zero_grad()

    GL_reg = torch.sum(torch.stack([group_lasso_regularization(model.model_list[i].layers[0].weight)
                                    for i in range(len(model.model_list))]))

    return mse.item(), R_reg.item(), GL_reg.item()


class W_fine_tuning(nn.Module):
    def __init__(self, w_prior, bias):
        super(W_fine_tuning, self).__init__()

        self.W_ft = nn.Parameter(torch.zeros_like(w_prior))
        self.w_prior = w_prior
        self.bias = nn.Parameter(bias, requires_grad=False)
        nn.init.xavier_normal_(self.W_ft)

    def train_epoch(self, dataloader, model, optimizer, adjacency, args):
        loss_train, reg_train = [], []
        for batch_idx, (data, _) in enumerate(dataloader):
            optimizer.zero_grad()

            data = data.to(args.device)

            W = (self.W_ft.mul(adjacency) + self.w_prior).unsqueeze(0).repeat(data.size(0), 1, 1, 1, 1)

            preds = 0
            for t in range(args.lag):
                preds += torch.stack([torch.bmm(W[:, i, :, :, t],
                                                data[:, :, t:data.size(2) - args.lag + t].view(data.size(0),
                                                                                               data.size(1), -1))
                                      for i, m in enumerate(model.model_list)], dim=1)
            preds = preds + self.bias
            preds = preds.contiguous().view(data.size(0), data.size(1), args.hidden_GC[0], -1, args.in_dim)
            preds = torch.cat([m(preds[:, i], 2) for i, m in enumerate(model.model_list)], dim=1)

            target = data[:, :, args.lag:]
            loss = ((preds - target) ** 2).mean(0).mean()

            loss.backward()
            optimizer.step()

            loss_train.append(loss.item())

        return np.mean(loss_train)


def train_seq2graph_epoch(epoch, encoder, decoder, dataloader, optimizer, scheduler,
                          eta, rel_rec, rel_send, args):

    w_prior = torch.stack([m.layers[0].weight.squeeze(-1) for m in decoder.model_list])
    adjacency = (w_prior.sum(-1).sum(1) != 0).float().unsqueeze(1).unsqueeze(-1)
    adjacency[range(args.num_atoms), :, range(args.num_atoms), :] = 0

    if decoder.bias:
        bias_prior = torch.stack([m.layers[0].bias.data for m in decoder.model_list]).unsqueeze(0).unsqueeze(-1)
    else:
        bias_prior = torch.zeros([1, args.num_atoms, decoder.model_list[0].layers[0].weight.shape[0], 1])

    loss_train, reg_train = [], []
    for batch_idx, (data, _) in enumerate(dataloader):
        optimizer.zero_grad()

        if args.cuda:
            data = data.cuda()

        w = encoder(data, rel_rec, rel_send)
        w = w.contiguous().view(-1, args.num_atoms, args.num_atoms, args.hidden_GC[0], args.lag)
        w = w.permute(0, 1, 3, 2, 4).mul(adjacency.unsqueeze(0))
        w_sparse = w + w_prior.unsqueeze(0)

        preds = 0
        for t in range(args.lag):
            preds += torch.stack([torch.bmm(w_sparse[:, i, :, :, t],
                                            data[:, :, t:data.size(2) - args.lag + t].view(data.size(0),
                                                                                           data.size(1), -1))
                                  for i, m in enumerate(decoder.model_list)], dim=1)
        preds = preds + bias_prior
        preds = preds.contiguous().view(data.size(0), data.size(1), args.hidden_GC[0], -1, args.in_dim)
        preds = torch.cat([m(preds[:, i], 1) for i, m in enumerate(decoder.model_list)], dim=1)

        target = data[:, :, args.lag:]
        loss = ((preds - target) ** 2).mean(0).sum()
        reg = torch.stack([torch.norm(w - encoder.P_prior.unsqueeze(0)) for w in w]).sum() + torch.norm(encoder.P_prior)

        (loss + eta * reg).backward()

        optimizer.step()

        loss_train.append(loss.item())
        reg_train.append(reg.item())

    scheduler.step()

    print('Epoch: {:04d}'.format(epoch),
          'loss: {:.5f}'.format(np.mean(loss_train)),
          'reg: {:.5f}'.format(np.mean(reg_train)))
