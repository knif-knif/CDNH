import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HashProxy(nn.Module):
    def __init__(self, temp, bce=False):
        super(HashProxy, self).__init__()
        self.temp = temp
        self.bce = bce
        self.Cy_Loss = nn.BCELoss()

    def forward(self, X, P, L, dim=1):

        X = F.normalize(X, p = 2, dim = -1)
        P = F.normalize(P, p = 2, dim = -1)
        
        D = F.linear(X, P) / self.temp
        if self.bce:
            xent_loss = self.Cy_Loss(torch.sigmoid(D), L)
        else:
            L /= torch.sum(L, dim=dim, keepdim=True).expand_as(L)
            xent_loss = torch.mean(torch.sum(-L * F.log_softmax(D, -1), -1))
        return xent_loss

def classifier(X, P, temp=0.1):
    X = F.normalize(X, p = 2, dim = -1)
    P = F.normalize(P, p = 2, dim = -1)
    D = F.linear(X, P) / temp

    return D

class HashDistill(nn.Module):
    def __init__(self):
        super(HashDistill, self).__init__()
        
    def forward(self, xS, xT):
        HKDloss = (1 - F.cosine_similarity(xS, xT.detach())).mean()
        return HKDloss

class BCEQuantization(nn.Module):
    def __init__(self, std):
        super(BCEQuantization, self).__init__()
        self.BCE = nn.BCELoss()
        self.std=std
    def normal_dist(self, x, mean, std):
        prob = torch.exp(-0.5*((x-mean)/std)**2)
        return prob
    def forward(self, x):
        x_a = self.normal_dist(x, mean=1.0, std=self.std)
        x_b = self.normal_dist(x, mean=-1.0, std=self.std)
        y = (x.sign().detach() + 1.0) / 2.0
        l_a = self.BCE(x_a, y)
        l_b = self.BCE(x_b, 1-y)
        return (l_a + l_b)

def negative_log_likelihood_similarity_loss(u, v, s):
    u = u.double()
    v = v.double()
    omega = torch.mm(u, v.T) / 2
    loss = -((s > 0).float() * omega - torch.log(1 + torch.exp(omega)))
    loss = torch.mean(loss)
    return loss


def calculate_similarity(labels1, labels2, tau=0):
    s = torch.mm(labels1, labels2.T)
    s = (s > tau).float()
    return s

def calc_label_sim(label_1, label_2):
    Sim = label_1.float().mm(label_2.float().t())
    return Sim


def cla_loss(view1_predict, view2_predict, labels_1, labels_2):
    cla_loss1 = ((view1_predict - labels_1.float()) ** 2).sum(1).sqrt().mean()
    cla_loss2 = ((view2_predict - labels_2.float()) ** 2).sum(1).sqrt().mean()

    return cla_loss1 + cla_loss2

def D(p, z):
    return - F.cosine_similarity(p, z.detach(), dim=-1).mean()

def mdl_loss(view1_feature, view2_feature, labels_1, labels_2):
    cos = lambda x, y: x.mm(y.t()) / (
        (x ** 2).sum(1, keepdim=True).sqrt().mm((y ** 2).sum(1, keepdim=True).sqrt().t())).clamp(min=1e-6) / 2.
    # theta11 = cos(view1_feature, view1_feature)
    theta12 = cos(view1_feature, view2_feature)
    # theta22 = cos(view2_feature, view2_feature)
    # Sim11 = calc_label_sim(labels_1, labels_1).float()
    Sim12 = calc_label_sim(labels_1, labels_2).float()
    # Sim22 = calc_label_sim(labels_2, labels_2).float()
    # term11 = ((1 + torch.exp(theta11)).log() - Sim11 * theta11).mean()
    term12 = ((1 + torch.exp(theta12)).log() - Sim12 * theta12).mean()
    # term22 = ((1 + torch.exp(theta22)).log() - Sim22 * theta22).mean()
    mdl_loss = term12

    return mdl_loss

from sklearn.metrics.pairwise import cosine_similarity


class HashLoss(nn.Module):
    def __init__(self, num_classes, hash_code_length, temp=0.2):
        super().__init__()
        self.num_classes = num_classes
        self.hash_code_length = hash_code_length
        self.classify_loss_fun = nn.BCELoss()
        self.temp = temp

    def calculate_similarity(self, label1):
        temp = torch.einsum('ij,jk->ik', label1, label1.t())
        L2_norm = torch.norm(label1, dim=1, keepdim=True)
        fenmu = torch.einsum('ij,jk->ik', L2_norm, L2_norm.t())
        sim = temp / fenmu
        return sim

    def hash_NLL_my(self, out, s_matrix):
        hash_bit = out.shape[1]
        cos = torch.tensor(cosine_similarity(out.detach().cpu(), out.detach().cpu())).cuda()
        w = torch.abs(s_matrix - (1 + cos) / 2)
        inner_product = torch.einsum('ij,jk->ik', out, out.t())

        L = w * ((inner_product + hash_bit) / 2 - s_matrix * hash_bit) ** 2

        diag_matrix = torch.tensor(np.diag(torch.diag(L.detach()).cpu())).cuda()
        loss = L - diag_matrix
        count = (out.shape[0] * (out.shape[0] - 1) / 2)

        return loss.sum() / 2 / count

    def quanti_loss(self, out):
        b_matrix = torch.sign(out)
        temp = torch.einsum('ij,jk->ik', out, out.t())
        temp1 = torch.einsum('ij,jk->ik', b_matrix, b_matrix.t())
        q_loss = temp - temp1
        q_loss = torch.abs(q_loss)
        loss = torch.exp(q_loss / out.shape[1])

        return loss.sum() / out.shape[0] / out.shape[0]

    def forward(self, X, P, L):
        X = F.normalize(X, p = 2, dim = -1)
        P = F.normalize(P, p = 2, dim = -1)
        D = F.linear(X, P) / self.temp
        classify_loss = self.classify_loss_fun(torch.sigmoid(D), L)
        sim_matrix = self.calculate_similarity(L)
        hash_loss = self.hash_NLL_my(X, sim_matrix)
        quanti_loss = self.quanti_loss(X)
        return classify_loss, hash_loss, quanti_loss
    
class NtXentLoss(nn.Module):
    def __init__(self, temperature=0.3):
        super(NtXentLoss, self).__init__()
        #self.batch_size = batch_size
        self.temperature = temperature
        #self.device = device

        #self.mask = self.mask_correlated_samples(batch_size)
        self.similarityF = nn.CosineSimilarity(dim = 2)
        self.criterion = nn.CrossEntropyLoss(reduction = 'sum')
    

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size 
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask
    

    def forward(self, z_i, z_j, device):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        batch_size = z_i.shape[0]
        N = 2 * batch_size

        z = torch.cat((z_i, z_j), dim=0)

        sim = self.similarityF(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        #sim = 0.5 * (z_i.shape[1] - torch.tensordot(z.unsqueeze(1), z.T.unsqueeze(0), dims = 2)) / z_i.shape[1] / self.temperature

        sim_i_j = torch.diag(sim, batch_size )
        sim_j_i = torch.diag(sim, -batch_size )
        
        mask = self.mask_correlated_samples(batch_size)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).view(N, 1)
        negative_samples = sim[mask].view(N, -1)

        labels = torch.zeros(N).to(device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss
from torch.autograd import Function
class hash(Function):
    @staticmethod
    def forward(ctx, input):
        # ctx.save_for_backward(input)
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        # input,  = ctx.saved_tensors
        # grad_output = grad_output.data

        return grad_output

def hash_layer(input):
    return hash.apply(input)

def compute_kl(prob, prob_v):
    prob_v = prob_v.detach()
    # prob = prob.detach()

    kl = prob * (torch.log(prob + 1e-8) - torch.log(prob_v + 1e-8)) + (1 - prob) * (torch.log(1 - prob + 1e-8 ) - torch.log(1 - prob_v + 1e-8))
    kl = torch.mean(torch.sum(kl, axis = 1))
    return kl

class AGCELoss(nn.Module):
    def __init__(self, num_classes=10, a=1, q=2, eps=1e-8, scale=1.):
        super(AGCELoss, self).__init__()
        self.a = a
        self.q = q
        self.num_classes = num_classes
        self.eps = eps
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        # label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        loss = ((self.a+1)**self.q - torch.pow(self.a + torch.sum(labels * pred, dim=1), self.q)) / self.q
        return loss.mean() * self.scale

class AUELoss(nn.Module):
    def __init__(self, num_classes=10, a=1.5, q=0.9, eps=1e-8, scale=1.0):
        super(AUELoss, self).__init__()
        self.num_classes = num_classes
        self.a = a
        self.q = q
        self.eps = eps
        self.scale = scale

    def forward(self, pred, L):
        pred = F.softmax(pred/0.2, dim=1)
        pL = L/torch.sum(L, dim=1, keepdim=True).expand_as(L)
        loss = (torch.pow(self.a - torch.sum(pL * pred, dim=1), self.q) - (self.a-1)**self.q)/ self.q
        return loss.mean() * self.scale

class pNorm(nn.Module):
    def __init__(self, p=0.5):
        super(pNorm, self).__init__()
        self.p = p

    def forward(self, pred, p=None):
        if p:
            self.p = p
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1)

        # 一个很简单的正则
        norm = torch.sum(pred ** self.p, dim=1)
        return norm.mean()