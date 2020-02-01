from args_experiments import *
from models import *
from load_data import *
from training import *

import matplotlib.pyplot as plt


args = args_cmapss()
print(args)

# Dataloader creation

if args.suffix == '_springs10':
    train_loader, valid_loader, test_loader = load_springs_data(path=args.path, batch_size=args.batch_size,
                                                                dataset_name=args.suffix, shuffle=True)
elif args.suffix == '_cmapss_001':
    train_loader, valid_loader, test_loader = load_cmapss_data(args.path, args.batch_size,
                                                               args.time_steps, shuffle=True)

args.num_atoms, args.time_steps, args.in_dim = train_loader.dataset[0][0].shape
torch.manual_seed(args.seed)

# Model initialization and training

model = GC(G_i, args.num_atoms, args.hidden_GC, 1, lag=args.lag, activation=nn.ReLU()).to(args.device)
optimizer = optim.Adam(model.parameters())

for epoch in range(args.epochs_adam):
    train_causality_epoch(epoch, model, optimizer, train_loader, args, validation=False)

torch.save(model.state_dict(), args.suffix + '.pt')

# Model sparsification with group lasso fine tuning until sparsity 15%
# This step may be long for low regularization importance coefficients

model.load_state_dict(torch.load(args.suffix + '.pt'))

s = 0.95
epoch = 0

while s >= 0.04:
    epoch += 1
    for batch_idx, data in enumerate(train_loader):
        data = data[0].to(args.device)

        optimizer.zero_grad()
        mse, ridge, reg = train_proximal_epoch(model, data, args.lr,
                                               args.lmbd_prox, args.lmbd_ridge, penalty='group_lasso')

        W = torch.stack([m.layers[0].weight.abs().mean(0).mean(1).mean(-1).cpu().detach()
                         for m in model.model_list])
        W = (W > 0).float().numpy()

    c = W.mean()

    if c <= s:
        torch.save(model.state_dict(), args.suffix + '_' + str(np.round(s, 2)) + '.pt')
        s = s - 0.05

    if epoch % 5 == 0:

        print('Epoch: {:04d}'.format(epoch), 'mse: {:.5f}'.format(mse), 'ridge: {:.5f}'.format(ridge),
              'reg: {:.5f}'.format(reg), 'sparsity: {:.2f}'.format(c))

# Fine tuning of the sparse parameters

all_losses, loss_train = [], []
model.load_state_dict(torch.load(args.suffix + '.pt'))

for batch_idx, (data, _) in enumerate(train_loader):

    if args.cuda:
        data = data.cuda()
    x_hat = model(data[:, :, :-1])
    loss = ((x_hat - data[:, :, args.lag:]) ** 2).mean(0).mean()  # F.mse_loss(x_hat, data[:, :, args.lag:])
    loss_train.append(loss.item())

print('Sparsity: {:.2f}, loss: {:.5f}'.format(1, np.mean(loss_train)))
all_losses.append(np.mean(loss_train))

for s in np.arange(0.95, 0.04, -0.05):
    model.load_state_dict(torch.load(args.suffix + '_' + str(np.round(s, 2)) + '.pt'))

    W_prior = torch.stack([m.layers[0].weight.squeeze(-1) for m in model.model_list])
    adjacency = (W_prior.sum(-1).sum(1) != 0).float().unsqueeze(1).unsqueeze(-1)

    if model.bias:
        bias_prior = torch.stack([m.layers[0].bias.data for m in model.model_list]).unsqueeze(0).unsqueeze(-1)
    else:
        bias_prior = torch.zeros([1, args.num_atoms, model.model_list[0].layers[0].weight.shape[0], 1]).to(args.device)

    w_fine_tuning = W_fine_tuning(W_prior, bias_prior).to(args.device)

    optimizer = optim.Adam(w_fine_tuning.parameters())

    for epoch in range(100):
        loss = w_fine_tuning.train_epoch(train_loader, model, optimizer, adjacency, args)

    print('Sparsity: {:.2f}, loss: {:.4f}'.format(s, loss))
    all_losses.append(loss)

    W_adjusted_prior = w_fine_tuning.W_ft.mul(adjacency) + W_prior

    for i, m in enumerate(model.model_list):
        m.layers[0].weight.data = W_adjusted_prior[i].unsqueeze(-1)

    torch.save(model.state_dict(), args.suffix + '_adjusted' + str(np.round(s, 2)) + '.pt')

plt.figure(figsize=(8, 2))
plt.plot(np.arange(0, len(all_losses)*5, 5), np.array(all_losses), 'o-')
plt.vlines(70, min(all_losses)*0.99, max(all_losses)*1.01)
plt.xlabel('Sparsity (%)')
plt.ylabel('Mean-squared error')
plt.show(block=False)

# Now the graphs is created, we choose the highest sparsity that does not decrease the performance
