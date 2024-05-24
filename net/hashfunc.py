from net import *
from torch.optim.lr_scheduler import CosineAnnealingLR
from helper import *
import torch.nn as nn 
import torch

class Hash_func(nn.Module):
    def __init__(self, arch, fc_dim, bit, nb_cls):
        super(Hash_func, self).__init__()
        self.arch = arch
        self.Hash = nn.Sequential(
            nn.Linear(fc_dim, fc_dim),
            nn.ReLU(),
            nn.Linear(fc_dim, bit)
        )
        self.head = nn.Sequential(
            nn.Linear(nb_cls, fc_dim),
            #nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(fc_dim, bit)
        )
        self.hax = nn.Sequential(
            nn.Linear(fc_dim, fc_dim),
            #nn.Dropout(0.2),
            nn.ReLU(inplace=False),
            nn.Linear(fc_dim, bit),
            nn.Tanh(),
        )
        self.P = nn.Parameter(torch.FloatTensor(nb_cls, bit), requires_grad=True)
        nn.init.xavier_uniform_(self.P, gain=nn.init.calculate_gain('tanh'))

        self.relu = nn.LeakyReLU(0.2)
        self.tanh = nn.Tanh()
        self.gnn = GraphConvolution
        self.lrn = [self.gnn(bit, 1024), self.gnn(1024, 1024), self.gnn(1024, 1024)]
        for i, layer in enumerate(self.lrn):
            self.add_module('lrn_{}'.format(i), layer)
        _adj = torch.FloatTensor(gen_A(nb_cls, 0.4, '/data2/knif/dataset/nuswide10k/adj.npz'))
        self.adj = Parameter(gen_adj(_adj), requires_grad=False)
        self.inp = Parameter(torch.rand(nb_cls, bit))
        self.hypo = nn.Linear(1024, bit)
        #self.cls = CosSim(bit, nb_cls, learn_cent=False)
    
    def forward(self, x):
        feat = self.arch(x)
        code = self.hax(feat)
        x = self.inp
        for i in range(len(self.lrn)):
            x = self.lrn[i](x, self.adj)
            x = self.relu(x)
        x = self.hypo(x)
        norm_img = torch.norm(code, dim=1)[:,None]*torch.norm(x, dim=1)[None,:]+1e-6
        x = x.transpose(0, 1)
        pred = torch.matmul(code, x)/norm_img
        code_h = 2 * (torch.sigmoid(self.head(pred)) -0.5)
        return feat, code, code_h, pred

class CosSim(nn.Module):
    def __init__(self, nfeat, nclass, learn_cent=True):
        super(CosSim, self).__init__()
        self.nfeat = nfeat
        self.nclass = nclass
        self.learn_cent = learn_cent

         # if no centroids, by default just usual weight
        codebook = torch.randn(nclass, nfeat)

        self.centroids = nn.Parameter(codebook.clone())
        if not learn_cent:
            self.centroids.requires_grad_(True)

    def forward(self, x):
        norms = torch.norm(x, p=2, dim=-1, keepdim=True)
        nfeat = torch.div(x, norms)

        norms_c = torch.norm(self.centroids, p=2, dim=-1, keepdim=True)
        ncenters = torch.div(self.centroids, norms_c)
        logits = torch.matmul(nfeat, torch.transpose(ncenters, 0, 1))

        return logits

class HashDec(nn.Module):
    def __init__(self, bit):
        super(HashDec, self).__init__()
        self.bn = nn.BatchNorm1d(bit)
        self.dec = nn.Linear(bit, 2)
    
    def forward(self, x):
        x = ReverseLayerF.apply(x, 1.0)
        x = self.bn(x)
        out = F.relu(self.dec(x))
        return out

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

from torch.nn import Parameter
import math
class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

def gen_A(num_classes, t, adj_file):
    result = np.load(adj_file)
    _adj = result['adj']
    _nums = result['nums']
    _nums = _nums[:, None]
    _adj = _adj / _nums
    _adj[_adj < t] = 0
    _adj[_adj >= t] = 1
    _adj = _adj.squeeze()
    _adj = _adj * 0.5
    _adj = _adj + np.identity(num_classes, np.float32)
    return _adj


def gen_adj(A):
    D = torch.pow(A.sum(axis=1).float(), -0.5)
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(A, D).t(), D)
    return adj