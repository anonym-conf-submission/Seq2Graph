import argparse
import torch
import os


def args_springs():
    """ Arguments used for the definition and the training of the graph inference for dataset springs """

    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true', default=True, help='Disables CUDA training.')
    parser.add_argument('--batch-size', type=int, default=64, help='Number of samples per batch.')
    parser.add_argument('--epochs_adam', type=int, default=1000, help='Number of epochs to train.')
    parser.add_argument('--hidden-GC', type=int, default=[], help='Number of hidden neurons in g_j.')
    parser.add_argument('--lag', action='store_true', default=2, help='Lag in the Granger causality model.')
    parser.add_argument('--lr', action='store_true', default=5e-4, help='Learning rate.')
    parser.add_argument('--lmbd-prox', action='store_true', default=1e-3, help='Proximal lambda parameter.')
    parser.add_argument('--lmbd-ridge', action='store_true', default=1e-3, help='Proximal lambda parameter.')
    parser.add_argument('--num-atoms', type=int, default=10, help='Number of atoms in simulation.')
    parser.add_argument('--prediction-steps', type=int, default=1, help='Random seed.')
    parser.add_argument('--proximal-gradient', type=int, default=True, help='Use proximal gradient descent.')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed.')
    parser.add_argument('--stationary', action='store_true', default=True, help='Each time series is stationary.')
    parser.add_argument('--suffix', type=str, default='_springs10', help='Suffix for training data (e.g. "_charged".')

    args, unknown = parser.parse_known_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    args.path = os.getcwd() + '/data/data_cmapss'

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    return args


def args_cmapss():
    """ Arguments used for the definition and the training of the graph inference for dataset CMAPSS """

    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true', default=True, help='Disables CUDA training.')
    parser.add_argument('--batch-size', type=int, default=128, help='Number of samples per batch.')
    parser.add_argument('--hidden-GC', type=int, default=[4, 4], help='Number of hidden neurons in g_j.')
    parser.add_argument('--epochs_adam', type=int, default=1000, help='Number of epochs to train.')
    parser.add_argument('--lag', action='store_true', default=2, help='Lag in the Granger causality model.')
    parser.add_argument('--lr', action='store_true', default=5e-4, help='Learning rate.')
    parser.add_argument('--lmbd-prox', action='store_true', default=5e-3, help='Proximal lambda parameter.')
    parser.add_argument('--lmbd-ridge', action='store_true', default=5e-3, help='Proximal lambda parameter.')
    parser.add_argument('--num-atoms', type=int, default=10, help='Number of atoms in simulation.')
    parser.add_argument('--prediction-steps', type=int, default=1, help='Random seed.')
    parser.add_argument('--proximal-gradient', type=int, default=True, help='Use proximal gradient descent.')
    parser.add_argument('--time_steps', type=int, default=25, help='Number of time steps per sample.')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed.')
    parser.add_argument('--stationary', action='store_true', default=True, help='Each time series is stationary.')
    parser.add_argument('--suffix', type=str, default='_cmapss_001', help='Suffix for training data (e.g. "_charged".')

    args, unknown = parser.parse_known_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    args.path = os.getcwd() + '/data/data_cmapss'

    args.device = torch.device('cuda' if args.cuda else 'cpu')

    return args


def args_seq2graph():
    """ Arguments used for the definition and the training of the graph inference for dataset springs """

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64, help='Number of samples per batch.')
    parser.add_argument('--encoder-hidden', type=int, default=64, help='Number of hidden units.')
    parser.add_argument('--epochs', type=int, default=600, help='Number of epochs to train.')
    parser.add_argument('--lr', action='store_true', default=5e-4, help='Learning rate.')
    parser.add_argument('--lmbda', action='store_true', default=1e-3, help='Learning rate.')
    parser.add_argument('--tau', type=int, default=0.5, help='Gumbel softmax temperature.')

    args, unknown = parser.parse_known_args()

    return args
