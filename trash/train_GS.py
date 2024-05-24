import torch
from sklearn.mixture import GaussianMixture
from loguru import logger
logger.remove(handler_id=None)
import warnings
warnings.filterwarnings('ignore')

from helper.settings import *
from helper.losses import *
from helper.utils import *
from helper.bar_show import progress_bar
from net.backbones import *
from net.hashfunc import *
from data.DataLoader import *

def set_train(train):
    if train:
        Arch.train()
        Hash_S.train()
        Hash_T.train()
    else:
        Arch.eval()
        Hash_S.eval()
        Hash_T.eval()

def create_arch(arch):
    if arch == 'Vgg':
        Arch = AlexNet()
        args.fc_dim = 4096
    elif arch == 'ResNet':
        Arch = ResNet()
        args.fc_dim = 2048
    elif arch == 'ViT':
        Arch = ViT('vit_base_patch16_224')
        args.fc_dim = 768
    elif arch == 'DeiT':
        Arch = DeiT('deit_base_distilled_patch16_224')
        args.fc_dim = 768
    elif arch == 'SwinT':
        Arch = SwinT('swin_base_patch4_window7_224')
        args.fc_dim = 1024
    else:
        raise ("Wrong dataset name.")
    
    return Arch

# KL divergence
def kl_divergence(p, q):
    return (p * ((p+1e-10) / (q+1e-10)).log()).sum(dim=1)

## Jensen-Shannon Divergence 
class Jensen_Shannon(nn.Module):
    def __init__(self):
        super(Jensen_Shannon,self).__init__()
        pass
    def forward(self, p,q):
        m = (p+q)/2
        return 0.5*kl_divergence(p, m) + 0.5*kl_divergence(q, m)

## Calculate JSD
def Calculate_JSD():  
    JS_dist = Jensen_Shannon()
    JSD   = torch.zeros(args.num_train)    

    for batch_idx, (img, label, index) in enumerate(eval_loader):
        img, label = img.cuda(), label.cuda()
        img_T = Norm(Crop(AugT(img)))
        img_S = Norm(Crop(AugS(img)))

        ## Get outputs of both network
        with torch.no_grad():
            code_T = Hash_T(Arch(img_T))
            code_S = Hash_S(Arch(img_S))
            code_T = F.normalize(code_T, p = 2, dim = -1)
            code_S = F.normalize(code_S, p = 2, dim = -1)
            PT = F.normalize(Hash_T.P, p = 2, dim = -1)
            PS = F.normalize(Hash_S.P, p = 2, dim = -1)
            DT = F.linear(code_T, PT) / 0.2
            DS = F.linear(code_S, PS) / 0.2
            label /= torch.sum(label, dim=1, keepdim=True).expand_as(label)
            ## Get the Prediction
            D = (DT + DS)/2 
            D = F.softmax(D, -1)
  
        dist = JS_dist(D, label)

        for b in range(img.size(0)):
            JSD[index[b]] = dist[b]

    return JSD

def eval_train(tp):
    Arch.eval()
    Hash_S.eval()
    Hash_T.eval()
    losses = torch.zeros(args.num_train)
    with torch.no_grad():
        for (img, label, index) in eval_loader:
            img, label = img.cuda(), label.cuda().float()
            img = Norm(Crop(AugT(img))) if tp else Norm(Crop(AugS(img)))
            feat = Arch(img)
            code = Hash_T(feat) if tp else Hash_S(feat)
            for b in range(img.size(0)):
                losses[index[b]] = HP_Loss(code[b], Hash_T.P if tp else Hash_S.P, label[b], 0)
    losses = (losses-losses.min()) / (losses.max()-losses.min())
    img_loss = losses.reshape(-1, 1)

    gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
    gmm.fit(img_loss)
    prob = gmm.predict_proba(img_loss)
    prob = prob[:, gmm.means_.argmin()]
    return prob


def warm(args, epoch, tp):
    set_train(True)
    global MAX_mAP, MAX_EPOCH
    C_loss = 0.0
    S_loss = 0.0
    R_loss = 0.0
    for i, (img, label, _) in enumerate(train_loader):
        img, label = img.cuda(), label.cuda().float()

        optim.zero_grad()

        l1 = torch.tensor(0., device=args.device)
        l2 = torch.tensor(0., device=args.device)
        l3 = torch.tensor(0., device=args.device)

        imgT = Norm(Crop(AugT(img)))
        imgS = Norm(Crop(AugS(img)))
        feat = Arch(imgT) if tp else Arch(imgS)
        code = Hash_T(feat) if tp else Hash_S(feat)

        l1 = HP_Loss(code, Hash_T.P if tp else Hash_S.P, label)
        #l2 = HD_Loss(imgS, imgT) * args.lambda_d
        #sim = calculate_similarity(label, label)
        #l2 = negative_log_likelihood_similarity_loss(code, code, sim)
        l3 = REG_Loss(code) * args.lambda_q

        loss = l1 + l3
        loss.backward()
        optim.step()

        C_loss += l1.item()
        S_loss += l2.item()
        R_loss += l3.item()

        if (i+1) % 10 == 0 or (i+1)==len(train_loader):    # print every 10 mini-batches
            progress_bar(i, len(train_loader), f"Loss: (C: {C_loss:.3f}, S: {S_loss:.3f}, R: {R_loss:.3f}) | Net: ({'Teacher' if tp else 'Student'})")
        
        C_loss = 0.0
        S_loss = 0.0
        R_loss = 0.0

    if epoch == args.warm_up-1 or (epoch+1)%args.eval_epoch==0:
        set_train(False)
        net = nn.Sequential(Arch, Hash_T if tp else Hash_S)
        map_all, map_5000 = evaluate(query_loader, retrieval_loader, net.eval())
        logger.info(f"\t[mAP]: map@All: {map_all:.4f}, map@5000: {map_5000:.4f} (Warm {'Teacher' if tp else 'Student'})")
        if map_all > MAX_mAP:
            MAX_mAP = map_all
            MAX_EPOCH = epoch+1
        
        set_train(True)

def train(args, epoch, tp):
    set_train(True)
    global MAX_mAP, MAX_EPOCH
    
    C_loss = 0.0
    S_loss = 0.0
    R_loss = 0.0
    for i, (img, label, _) in enumerate(conf_train):
        img, label = img.cuda(), label.cuda().float()
        optim.zero_grad()

        l1 = torch.tensor(0., device=args.device)
        l2 = torch.tensor(0., device=args.device)
        l3 = torch.tensor(0., device=args.device)

        imgT = Norm(Crop(AugT(img))) 
        imgS = Norm(Crop(AugS(img)))
        feat = Arch(imgT) if tp else Arch(imgS)
        code = Hash_T(feat) if tp else Hash_S(feat)

        l1 = HP_Loss(code, Hash_T.P if tp else Hash_S.P, label)
        #l2 = HD_Loss(imgS, imgT) * args.lambda_d
        #sim = calculate_similarity(label, label)
        #l2 = negative_log_likelihood_similarity_loss(code, code, sim)       
        l3 = REG_Loss(code) * args.lambda_q

        loss = l1 + l3
        loss.backward()
        optim.step()

        C_loss += l1.item()
        S_loss += l2.item()
        R_loss += l3.item()

        if (i+1) % 10 == 0 or (i+1)==len(conf_train):    # print every 10 mini-batches
            progress_bar(i, len(conf_train), f"Loss: (C: {C_loss:.3f}, S: {S_loss:.3f}, R: {R_loss:.3f}) | Net: ({'Teacher' if tp else 'Student'})")
        
        C_loss = 0.0
        S_loss = 0.0
        R_loss = 0.0

    if (epoch+1)%args.eval_epoch==0:
        set_train(False)
        net = nn.Sequential(Arch, Hash_T if tp else Hash_S)
        map_all, map_5000 = evaluate(query_loader, retrieval_loader, net.eval())
        logger.info(f"\t[mAP]: map@All: {map_all:.4f}, map@5000: {map_5000:.4f} (Train {'Teacher' if tp else 'Student'})")
        if map_all > MAX_mAP:
            MAX_mAP = map_all
            MAX_EPOCH = epoch+1
        
        set_train(True)


if __name__ == '__main__':
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    args = parse_arguments()

    os.makedirs('./log/' + args.log_id, exist_ok=True)
    logger.add(os.path.join('log/' + args.log_id, f'{args.dataset}_{args.bit}_{args.noiseRate}.log'), 
               format="<green>{time:YYYY-MM-DD HH:mm}</green> | <level>{level: <4}</level> | - <level>{message}</level>",
               rotation='500MB', level='INFO')
    logger.info('Start Train')
    logger.info(args)

    query_loader, train_loader, eval_loader, retrieval_loader, Clean_Rate = load_data(args)
    logger.info(f"[Clean Rate]: {Clean_Rate:.4f} | ({1-Clean_Rate:.2f})")

    AugS = Augmentation(args.img_size, 1.0)
    AugT = Augmentation(args.img_size, args.transformation_scale)

    Crop = nn.Sequential(Kg.CenterCrop(args.img_size))
    Norm = nn.Sequential(Kg.Normalize(mean=torch.as_tensor(args.mean), std=torch.as_tensor(args.std)))

    Arch = create_arch(args.arch).cuda()
    Hash_T = Hash_func(args.fc_dim, args.bit, args.num_classes).cuda()
    Hash_S = Hash_func(args.fc_dim, args.bit, args.num_classes).cuda()

    HP_Loss = HashProxy(args.temp)
    HD_Loss = HashDistill()
    REG_Loss = BCEQuantization(0.5)

    params = [{'params': Arch.parameters(), 'lr': 0.05*args.lr}, {'params': Hash_T.parameters(), 'lr': args.lr}, {'params': Hash_S.parameters()}]
    optim = torch.optim.Adam(params, lr=args.lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optim, args.Epoch, eta_min=0, last_epoch=-1)
        
    MAX_mAP = 0.0
    MAX_EPOCH = 0

    for epoch in range(args.Epoch):
        logger.info(f"[Epoch]: {epoch+1:<3}")
        print('Epoch:', epoch+1, 'LR:', '%.6f,'%optim.param_groups[0]['lr'], '%.6f'%optim.param_groups[1]['lr'], '%.6f'%optim.param_groups[2]['lr'])
        if epoch < args.warm_up:
            warm(args, epoch, True)
            warm(args, epoch, False)
        else:
            # prob_T = Calculate_JSD()
            # threshold = torch.mean(prob_T)
            # if threshold.item()>0.7:
            #     threshold = threshold - (threshold-torch.min(prob_T)) / 5
            # pred_T = (prob_T < threshold)
            prob_T = eval_train(False)
            pred_T = (prob_T > args.p_threshold)
            conf_train, Label_Size, Auc, Acc = load_conf(args, pred_T, prob_T)
            logger.info(f"\t[Teacher]: Label_Size: {Label_Size}, AUC: {Auc:.4f}, Acc: {Acc:.4f}")
            train(args, epoch, True)
            prob_S = eval_train(True)
            pred_S = (prob_S > args.p_threshold)
            # prob_S = Calculate_JSD()
            # threshold = torch.mean(prob_S)
            # if threshold.item()>0.7:
            #     threshold = threshold - (threshold-torch.min(prob_S)) / 5
            # pred_S = (prob_S < threshold)
            conf_train, Label_Size, Auc, Acc = load_conf(args, pred_S, prob_S)
            logger.info(f"\t[Student]: Label_Size: {Label_Size}, AUC: {Auc:.4f}, Acc: {Acc:.4f}")
            train(args, epoch, False)
        scheduler.step()
    logger.info(f"[MAX_mAP]: {MAX_mAP:.4f} ({MAX_EPOCH})")