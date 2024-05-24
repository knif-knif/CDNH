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

def eval_train(tp):
    Arch.eval()
    Hash_S.eval()
    Hash_T.eval()
    losses = torch.zeros(args.num_train, args.num_classes)
    with torch.no_grad():
        for (img, label, index) in eval_loader:
            img, label = img.cuda(), label.cuda().float()
            img = Norm(Crop(AugT(img))) if tp else Norm(Crop(AugS(img)))
            feat = Arch(img)
            code = Hash_T(feat) if tp else Hash_S(feat)
            for b in range(img.size(0)):
                losses[index[b]] = HMP_Loss(code[b], Hash_T.P if tp else Hash_S.P, label[b], 0)
    losses = (losses-losses.min()) / (losses.max()-losses.min())
    img_loss = losses.reshape(-1, args.num_classes)

    gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
    gmm.fit(img_loss)
    prob = gmm.predict_proba(img_loss)
    prob = prob[:, gmm.means_.argmin()]
    return prob


def warm(args, epoch, tp):
    set_train(True)
    global MAX_mAP
    print('Epoch:', epoch+1, 'LR:', '%.6f,'%optim.param_groups[0]['lr'], '%.6f'%optim.param_groups[1]['lr'], '%.6f'%optim.param_groups[2]['lr'])
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
        l2 = HD_Loss(imgS, imgT) * args.lambda_d
        l3 = REG_Loss(code) * args.lambda_q

        loss = l1 + l2 + l3
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
        
        set_train(True)

def train(args, epoch, tp):
    set_train(True)
    global MAX_mAP
    
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
        l2 = HD_Loss(imgS, imgT) * args.lambda_d
        l3 = REG_Loss(code) * args.lambda_q

        loss = l1 + l2 + l3
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
        
        set_train(True)


if __name__ == '__main__':
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    print("Init...")
    args = parse_arguments()

    logger.add(os.path.join('log', f'{args.log_id}{args.dataset}_{args.bit}_{args.noiseRate}.log'), 
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

    HP_Loss = HashProxy(args.temp, bce=True)
    HMP_Loss = HashProxy(args.temp, bce=True, reduction=True)
    HD_Loss = HashDistill()
    REG_Loss = BCEQuantization(0.5)

    params = [{'params': Arch.parameters(), 'lr': 0.05*args.lr}, {'params': Hash_T.parameters(), 'lr': args.lr}, {'params': Hash_S.parameters()}]
    optim = torch.optim.Adam(params, lr=args.lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optim, args.Epoch, eta_min=0, last_epoch=-1)
        
    MAX_mAP = 0.0

    for epoch in range(args.Epoch):
        logger.info(f"[Epoch]: {epoch+1:<3}")
        print('Epoch:', epoch, 'LR:', '%.6f,'%optim.param_groups[0]['lr'], '%.6f'%optim.param_groups[1]['lr'], '%.6f'%optim.param_groups[2]['lr'])
        if epoch < args.warm_up:
            warm(args, epoch, True)
            warm(args, epoch, False)
        else:
            prob_T = eval_train(False)
            prob_S = eval_train(True)

            pred_T = (prob_T > args.p_threshold)
            pred_S = (prob_S > args.p_threshold)

            conf_train, Label_Size, Auc = load_conf(args, pred_T, prob_T)
            logger.info(f"\t[Teacher]: Label_Size: {Label_Size}, AUC: {Auc:.4f}")
            train(args, epoch, True)
            conf_train, Label_Size, Auc = load_conf(args, pred_S, prob_S)
            logger.info(f"\t[Student]: Label_Size: {Label_Size}, AUC: {Auc:.4f}")
            train(args, epoch, False)
        scheduler.step()
    logger.info(f"[MAX_mAP]: {MAX_mAP:.4f}")