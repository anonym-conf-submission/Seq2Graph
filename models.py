import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Two-layer fully-connected ELU net with batch norm."""

    def __init__(self, n_in, n_hid, n_out, do_prob=0., gain=1.0):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_out)
        self.bn = nn.BatchNorm1d(n_out)
        self.dropout_prob = do_prob

        self.init_weights(gain)

    def init_weights(self, gain=1.0):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data, gain=gain)
                m.bias.data.fill_(0.0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):
        x = inputs.view(inputs.size(0) * inputs.size(1), -1)
        x = self.bn(x)
        return x.view(inputs.size(0), inputs.size(1), -1)

    def forward(self, inputs):
        # Input shape: [num_sims, num_things, num_features]
        x = F.elu(self.fc1(inputs))
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = F.elu(self.fc2(x))
        return self.batch_norm(x)


class RelationalEncoder(nn.Module):
    """ A two-layer relational neural network for MTS emedding """

    def __init__(self, n_in, n_hid, n_out, gain=1.0):
        super(RelationalEncoder, self).__init__()

        self.mlp1 = MLP(n_in, n_hid, n_hid)
        self.mlp2 = MLP(n_hid * 2, n_hid, n_hid)
        self.mlp3 = MLP(n_hid, n_hid, n_hid)
        self.mlp4 = MLP(n_hid * 3, n_hid, n_hid)
        self.fc_out = MLP(n_hid, n_hid, n_out)
        self.init_weights(gain)

    def init_weights(self, gain=1.0):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data, gain)
                m.bias.data.fill_(0.0)

    def edge2node(self, x, rel_rec):
        incoming = torch.matmul(rel_rec.t(), x)
        return incoming / incoming.size(1)

    def node2edge(self, x, rel_rec, rel_send):
        receivers = torch.matmul(rel_rec, x)
        senders = torch.matmul(rel_send, x)
        edges = torch.cat([receivers, senders], dim=2)
        return edges

    def forward(self, inputs, rel_rec, rel_send):
        x = inputs.contiguous().view(inputs.size(0), inputs.size(1), -1)
        x = self.mlp1(x)
        x = self.node2edge(x, rel_rec, rel_send)
        x = self.mlp2(x)
        x_skip = x
        x = self.edge2node(x, rel_rec)
        x = self.mlp3(x)
        x = self.node2edge(x, rel_rec, rel_send)
        x = torch.cat((x, x_skip), dim=2)  # Skip connection
        x = self.mlp4(x)

        return self.fc_out(x)


class G_i(nn.Module):

    def __init__(self, in_dim, h_dims, out_dim, lag, activation=nn.ReLU(), bias=True):
        super(G_i, self).__init__()
        self.in_dim = in_dim
        self.h_dims = h_dims
        self.out_dim = out_dim
        self.lag = lag
        self.activation = activation
        self.bias = bias
        if len(h_dims) > 0:
            self.layers = nn.ModuleList()
            self.layers.append(nn.Conv2d(in_dim, h_dims[0], (lag, 1), bias=bias))
            for i in range(0, len(h_dims) - 1):
                self.layers.append(activation)
                self.layers.append(nn.Conv2d(h_dims[i], h_dims[i + 1], (1, 1), bias=bias))
            self.layers.append(activation)
            self.layers.append(nn.Conv2d(h_dims[-1], out_dim, (1, 1), bias=bias))
        else:
            self.layers = nn.ModuleList([nn.Conv2d(in_dim, out_dim, (lag, 1), bias=bias)])

        self.init_weights()

    def init_weights(self):
        for l in self.layers:
            if type(l) != type(self.activation):
                nn.init.xavier_normal_(l.weight.data, 0.1)
                if self.bias:
                    l.bias.data.fill_(0.0)

    def forward(self, X, first_layer_index=0):
        x = X
        for i, layer in enumerate(self.layers[first_layer_index:]):
            x = layer(x)
        return x


class GC(nn.Module):
    def __init__(self, g_i, in_dim, h_dims, out_dim, lag, activation=nn.ReLU(), bias=True):
        super(GC, self).__init__()
        self.in_dim = in_dim
        self.h_dims = h_dims
        self.out_dim = out_dim
        self.lag = lag
        self.bias = bias
        self.activation = activation
        self.model_list = nn.ModuleList([g_i(in_dim, h_dims, out_dim, lag, activation, bias)
                                         for _ in range(in_dim)])

    def forward(self, input_data, i=None, first_layer_index=0):
        if i is None:
            return torch.cat([m(input_data, first_layer_index) for m in self.model_list], dim=1)
        else:
            return self.model_list[i](input_data, first_layer_index)
