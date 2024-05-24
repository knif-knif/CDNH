import torch
import kornia.augmentation as Kg
import numpy as np
import scipy
from scipy.io import savemat
from helper.losses import *

def mean_average_precision(query_code,
                           retrieval_code,
                           query_targets,
                           retrieval_targets,
                           device,
                           topk=None
                           ):
    """
    Calculate mean average precision(map).

    Args:
        query_code (torch.Tensor): Query data hash code.
        retrieval_code (torch.Tensor): Retrieval data hash code.
        query_targets (torch.Tensor): Query data targets, one-hot
        retrieval_targets (torch.Tensor): retrieval data targets, one-hot
        device (torch.device): Using CPU or GPU.
        topk: int

    Returns:
        meanAP (float): Mean Average Precision.
    """
    # query_code = query_code.to(device)
    # retrieval_code = retrieval_code.to(device)
    # query_targets = query_targets.to(device)
    # retrieval_targets = retrieval_targets.to(device)
    num_query = query_targets.shape[0]
    if topk == None:
        topk = retrieval_targets.shape[0] 
    mean_AP = 0.0

    for i in range(num_query):
        # Retrieve images from database
        retrieval = (query_targets[i, :] @ retrieval_targets.t() > 0).float()

        # Calculate hamming distance
        hamming_dist = 0.5 * (retrieval_code.shape[1] - query_code[i, :] @ retrieval_code.t())

        # Arrange position according to hamming distance
        retrieval = retrieval[torch.argsort(hamming_dist)][:topk]

        # Retrieval count
        retrieval_cnt = retrieval.sum().int().item()

        # Can not retrieve images
        if retrieval_cnt == 0:
            continue

        # Generate score for every position
        score = torch.linspace(1, retrieval_cnt, retrieval_cnt).to(device)  # 返回一个1维张量，包含在区间start和end上均匀间隔的step个点。
        # Acquire index
        index = (torch.nonzero(retrieval).squeeze() + 1.0).float().to(device)

        mean_AP += (score / index).mean()

    mean_AP = mean_AP / num_query
    return mean_AP.item()


def evaluate_hx(query_loader, retrieval_loader, net):
    net.eval()
    with torch.no_grad():
        for i, (img, label) in enumerate(query_loader):
            img, label = img.cuda(), label.cuda().float()

            # code = hash_layer(torch.sigmoid(net(img)) - 0.5)
            _, code, code_h,  _ = net(img)
            
            if i==0: 
                query_code = code
                query_code_h = code_h
                query_label = label
            else:
                query_code = torch.cat([query_code, code], 0)
                query_code_h = torch.cat([query_code_h, code_h], 0)
                query_label = torch.cat([query_label, label], 0)
    
        for i, (img, label) in enumerate(retrieval_loader):
            img, label = img.cuda(), label.cuda().float()
            # code = hash_layer(torch.sigmoid(net(img)) - 0.5)
            _, code, code_h, _ = net(img)

            if i==0:
                retrieval_code = code
                retrieval_label = label
                retrieval_code_h = code_h
            else:
                retrieval_code = torch.cat([retrieval_code, code], 0)
                retrieval_code_h = torch.cat([retrieval_code_h, code_h], 0)
                retrieval_label = torch.cat([retrieval_label, label], 0)

    query_code = torch.sign(query_code)
    query_code_h = torch.sign(query_code_h)
    retrieval_code = torch.sign(retrieval_code)
    retrieval_code_h = torch.sign(retrieval_code_h)
    map_all = mean_average_precision(
        query_code,
        retrieval_code,
        query_label,
        retrieval_label,
        torch.device('cuda')
    )

    map_5000 = mean_average_precision(
        query_code,
        retrieval_code,
        query_label,
        retrieval_label,
        torch.device('cuda'),
        5000
    )
    return map_all, map_5000