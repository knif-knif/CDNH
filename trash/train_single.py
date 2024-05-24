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

def log(epoch, batch_idx, num_iter, loss, lossB=0):
    sys.stdout.write('\r')
    sys.stdout.write('| %s:%.1f-%s | Epoch [%2d/%2d] Iter[%3d/%3d]\t lossA: %.4f lossB: %.4f'
                %(args.dataset, args.noiseLevel, args.noiseType, epoch, args.Epoch, batch_idx+1, num_iter, loss, lossB))
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

def test(epoch):
    global best_acc
    neta.eval()
    netb.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for (img, label) in query_loader:
            img, label = img.cuda(), label.cuda()
            _, preda, _ = neta(img)
            _, predb, _ = netb(img)
            pred = torch.sigmoid(preda) + torch.sigmoid(predb)
            _, pred = torch.max(pred, 1)
            _, target = torch.max(label, 1)
            total += label.size(0)
            correct += pred.eq(target).cpu().sum().item()
    acc = 100. * correct / total
    if best_acc < acc:
        best_acc = acc
    print("\n| Test Epoch #%d\n  Accuracy: %.2f%%\n  Best Acc %.2f%%" % (epoch, acc, best_acc))

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
    for batch_idx, (img_x, img_aug_x, label_x, w_x) in enumerate(label_loader):
        w_x = w_x.view(-1, 1).type(torch.FloatTensor)

        img_x, img_aug_x, label_x, w_x = img_x.cuda(), img_aug_x.cuda(), label_x.cuda(), w_x.cuda()
        code, pred_feat, pred = net(img_x)
        #loss = CyLoss(torch.sigmoid(pred), label_x)
        cyloss, hxloss, qloss = HxLoss(code, pred, label_x)
        loss = cyloss + 0.01 * hxloss + 0.0001 * qloss + CyLoss(torch.sigmoid(pred_feat), label_x)
        #loss = loss*0.1
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
    logger.add(os.path.join('log', f'{args.dataset}_{args.bit}_{args.noiseType}_{args.noiseLevel}_{args.log_id}.log'), rotation='500 MB', level='INFO')
    logger.info(args)
    
    query_loader, train_loader, retrieval_loader, eval_loader = load_data( args )
    neta = create_net(args.bit, args.num_classes)
    netb = create_net(args.bit, args.num_classes)
    opta = create_optim(args.optim, neta, args.lr)
    optb = create_optim(args.optim, netb, args.lr)

    CyLoss = nn.BCELoss()
    SeLoss = SemiLoss()
    HxLoss = HashLoss(args.num_classes, args.bit)
    best_acc = 0.
    best_map = 0
    all_loss = [[], []]
    for epoch in range(args.Epoch):
        neta.set_alpha(epoch)
        netb.set_alpha(epoch)
        if epoch < args.warm_up:
            print('Warm Net A')
            warmup(epoch, neta, opta)
            print('\nWarm Net B')
            warmup(epoch, netb, optb) 
            test(epoch)
            #torch.save(neta, './ckpt/neta_feat.pt')
            #torch.save(netb, './ckpt/netb_feat.pt')
        else:
            neta.vgg19.eval()
            neta.vgg19.requires_grad_(False)
            netb.vgg19.eval()
            netb.vgg19.requires_grad_(False)

            proba, all_loss[0] = eval_train(neta, all_loss[0], eval_loader)
            probb, all_loss[1] = eval_train(netb, all_loss[1], eval_loader)

            preda = (proba > args.p_threshold)
            predb = (probb > args.p_threshold)

            print('Train Net A')
            labeled_loader, _ = load_data(args, True, predb, probb)
            train(epoch, neta, opta, labeled_loader)

            print('\nTrain Net B')
            label_loader, _ = load_data(args, True, preda, proba)
            train(epoch, netb, optb, labeled_loader)
        if epoch % 20 == 19: args.lr = args.lr * 0.8
        if epoch % 10 == 9:
            #update_lr(args.lr)
            start_time = time.time()
            map_all = evaluate_single(query_loader, retrieval_loader, neta, netb)
            if best_map < map_all:best_map = map_all
            logger.info(f"\n[mAP]:{map_all:.4f} [==] {best_map:.4f} (Time:{time.time() - start_time:.0f})")
            test(epoch)