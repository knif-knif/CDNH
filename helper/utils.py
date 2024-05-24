import torch
import numpy as np

def pseudo_s(p, tau, label):
    new_p = torch.zeros_like(p)
    ind = torch.argsort(p, dim=1, descending=True)
    bs = p.shape[0]
    n = p.shape[1]
    conf_id = []
    for i in range(bs):
        sum_p = 0
        for j in range(n//2):
            pj = p[i][ind[i][j]]
            sum_p += pj
            if sum_p>tau and pj*(j+1)>tau:
                new_p[i][ind[i][:j+1]] = 1
                conf_id.append(i)
                break
        if len(conf_id)==0 or conf_id[-1]!=i: new_p[i] = label[i]
    return new_p, np.array(conf_id)

def pseudo(p, tau, gm, label):
    new_p = torch.zeros_like(p)
    conf_id = []
    for i in range(p.shape[0]):
        sp = True
        for j in range(p.shape[1]):
            if p[i][j]>=tau:new_p[i][j]=1
            elif p[i][j]>gm:
                sp=False
                break
        if sp: conf_id.append(i) 
        else: new_p[i] = label[i]
    return new_p, np.array(conf_id)

def knn(cur_feature, feature, label, num_classes, knn_k=100, chunks=1, norm='global'):
    num = len(cur_feature)
    num_class = torch.tensor([torch.sum(label[:,i]==1).item() for i in range(num_classes)]).to(feature.device) + 1e-10
    pi = num_class / num_class.sum()
    split = torch.tensor(np.linspace(0, num, chunks+1, dtype=int), dtype=torch.long).to(feature.device)
    score = torch.tensor([]).to(feature.device)
    pred = torch.tensor([], dtype=torch.long).to(feature.device)
    feature = torch.nn.functional.normalize(feature, dim=1)
    with torch.no_grad():
        for i in range(chunks):
            torch.cuda.empty_cache()
            part_feature = cur_feature[split[i]: split[i+1]]
            part_score, part_pred = knn_predict(part_feature, feature.T, label, num_classes, knn_k)
            score = torch.cat([score, part_score], dim=0)
            pred = torch.cat([pred, part_pred], dim=0)

        if norm!='global':
            score = score / pi
        
        score = score/score.sum(1, keepdim=True)

    return score

def knn_predict(feature, feature_bank, feature_labels, classes, knn_k):
    sim_matrix = torch.mm(feature, feature_bank) # n * n
    
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1) # n * k
    score = torch.zeros(sim_weight.shape[0], classes).cuda() # n * cl
    for b in range(score.shape[0]):
        score[b] = feature_labels[sim_indices[b]].sum(dim=0)
    
    pred = score

    return score, pred


    