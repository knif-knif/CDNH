import torch
import torch.nn as nn
from loguru import logger
import importlib
import numpy as np
import random
from sklearn.mixture import GaussianMixture
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

from data.data_loader import load_data
from helper.settings import parse_arguments
from helper.losses import *
from helper.utils import *
from net.nets import ImgNet

def update_lr(lr):
    for param_group in opta.param_groups:
        param_group['lr'] = lr
    for param_group in optb.param_groups:
        param_group['lr'] = lr

def step(opt, loss):
    opt.zero_grad()
    loss.backward()
    opt.step()

def log(epoch, batch_idx, num_iter, loss):
    sys.stdout.write('\r')
    sys.stdout.write('| %s | Epoch [%2d/%2d] Iter[%3d/%3d] loss: %.4f'
                %(args.dataset, epoch, args.Epoch, batch_idx+1, num_iter, loss))
    sys.stdout.flush()

def warmup(epoch, net, opt):
    net.train()
    num_iter = len(train_loader)
    for batch_idx, (img, _, label, _) in enumerate(train_loader):
        img, label = img.cuda(), label.cuda()
        _, pred, _ = net(img)
        loss = CyLoss(torch.sigmoid(pred), label)
        log(epoch+1, batch_idx, num_iter, loss.item())
        step(opt, loss)

def eval_train(net, all_loss, eval_loader):
    net.eval()
    losses = torch.zeros(args.num_train)
    with torch.no_grad():
        for (img, _, label, index) in eval_loader:
            img, label = img.cuda(), label.cuda()
            _, pred, _ = net(img)
            for b in range(img.size(0)):
                losses[index[b]]=CyLoss(torch.sigmoid(pred[b]), label[b]) 
    losses = (losses-losses.min()) / (losses.max()-losses.min())
    all_loss.append(losses)

    if args.noiseLevel == 0.9:
        history = torch.stack(all_loss)
        img_loss = history[-5:].mean(0)
        img_loss = img_loss.reshape(-1, 1)
    else:
        img_loss = losses.reshape(-1, 1)
    
    gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
    gmm.fit(img_loss)
    prob = gmm.predict_proba(img_loss)
    prob = prob[:, gmm.means_.argmin()]
    return prob, all_loss

def train(epoch, net, opt, label_loader):
    net.train()
    num_iter = len(label_loader)
    for batch_idx, (img, label, _) in enumerate(label_loader):
        img, label = img.cuda().float(), label.cuda().float()
        code, _, pred = net(img)
        #cyloss, hxloss, qloss = HxLoss(code, pred, label_x)
        sim = calculate_similarity(label, label)
        loss = negative_log_sim(code, code, sim) + MsLoss(torch.sigmoid(pred), label)
        #loss = cyloss + 0.01 * hxloss + 0.0001 * qloss #+ CyLoss(torch.sigmoid(pred_feat), label_x)
        step(opt, loss)
        log(epoch, batch_idx, num_iter, loss.item())

def create_net(bit, num_classes):
    net = ImgNet(bit, num_classes).cuda()
    return net

def create_optim(type, net, lr):
    if type == 'Adam':
        return torch.optim.Adam(net.parameters(), lr=lr, eps=1e-8)
    elif type == 'SGD':
        return torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=args.wdecay)

if __name__ == '__main__':
    args = parse_arguments()
    logger.add(os.path.join('log', f'{args.dataset}_{args.bit}_{args.noiseRate}.log'), rotation='500 MB', level='INFO')
    logger.info(f"Use neg_log Loss + mse Loss")
    logger.info(args)
    
    query_loader, train_loader, retrieval_loader = load_data( args )
    neta = create_net(args.bit, args.num_classes)
    netb = create_net(args.bit, args.num_classes)
    opta = create_optim(args.optim, neta, args.lr)
    optb = create_optim(args.optim, netb, args.lr)

    CyLoss = nn.BCELoss()
    MsLoss = nn.MSELoss()
    HxLoss = HashLoss(args.num_classes, args.bit)
    best_map = 0
    all_loss = [[], []]
    
    for epoch in range(args.Epoch):
        train(epoch, neta, opta, train_loader)
        train(epoch, netb, optb, train_loader)
        if (epoch%5==4):
            #args.lr*=0.8
            update_lr(args.lr)
            epoch_map_all, epoch_map_2100 = evaluate_multi(query_loader, retrieval_loader, neta, netb)
            best_map = max(best_map, epoch_map_all)
            logger.info(f"[mAP]: @All: {epoch_map_all:.4f}, @2100: {epoch_map_2100} [{best_map:.4f}]")