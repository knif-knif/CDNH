import torch
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

from helper.settings import *
from helper.losses import *
from helper.evaluate import *
from helper.bar_show import progress_bar
from net.backbones import *
from net.hashfunc import *
from data.DataLoader import *

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

def train(args):
    global MAX_mAP
    for epoch in range(args.Epoch):
        print('Epoch:', epoch, 'LR:', '%.6f,'%optim.param_groups[0]['lr'], '%.6f'%optim.param_groups[1]['lr'])
        C_loss = 0.0
        S_loss = 0.0
        R_loss = 0.0
        for i, (img, label) in enumerate(train_loader):
            img, label = img.cuda(), label.cuda().float()

            optim.zero_grad()

            l1 = torch.tensor(0., device=args.device)
            l2 = torch.tensor(0., device=args.device)
            l3 = torch.tensor(0., device=args.device)

            Is = Norm(Crop(AugS(img)))
            It = Norm(Crop(AugT(img)))

            Xt = net(It)
            l1 = HP_Loss(Xt, Hash.P, label)

            Xs = net(Is)
            l2 = HD_Loss(Xs, Xt) * args.lambda_d
            l3 = REG_Loss(Xt) * args.lambda_q

            loss = l1 + l2 + l3
            loss.backward()
            optim.step()

            C_loss += l1.item()
            S_loss += l2.item()
            R_loss += l3.item()

            if (i+1) % 10 == 0 or (i+1)==len(train_loader):    # print every 10 mini-batches
                progress_bar(i, len(train_loader), f'Loss: (C: {C_loss:.3f}, S: {S_loss:.3f}, R: {R_loss:.3f}) | Map: (Max: {MAX_mAP:.4f})')
            
            
            C_loss = 0.0
            S_loss = 0.0
            R_loss = 0.0

             
        if epoch >= args.warm_up:
            scheduler.step()

        if (epoch+1) % args.eval_epoch == 0 and (epoch+1) >= args.eval_init:
            map_all, map_5000 = evaluate(query_loader, retrieval_loader, net.eval())
            logger.info(f"[Epoch {epoch+1}]: map@All: {map_all:.4f}, map@5000: {map_5000:.4f}")
            #print(f"map@All: {map_all:.4f}, map@5000: {map_5000:.4f}")
            if map_all > MAX_mAP:
                MAX_mAP = map_all
            
            net.train()

if __name__ == '__main__':
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    print("Init...")
    args = parse_arguments()

    logger.add(os.path.join('log', f'{args.log_id}{args.dataset}_{args.bit}_{args.noiseRate}.log'), rotation='500MB', level='INFO')
    logger.info('Start Train')
    logger.info(args)

    query_loader, train_loader, retrieval_loader = load_data(args)
    
    AugS = Augmentation(args.img_size, 1.0)
    AugT = Augmentation(args.img_size, args.transformation_scale)

    Crop = nn.Sequential(Kg.CenterCrop(args.img_size))
    Norm = nn.Sequential(Kg.Normalize(mean=torch.as_tensor(args.mean), std=torch.as_tensor(args.std)))

    Arch = create_arch(args.arch)
    Hash = Hash_func(args.fc_dim, args.bit, args.num_classes)
    net = nn.Sequential(Arch, Hash).cuda()

    HP_Loss = HashProxy(args.temp)
    HD_Loss = HashDistill()
    REG_Loss = BCEQuantization(0.5)

    params = [{'params': Arch.parameters(), 'lr': 0.05*args.lr}, {'params': Hash.parameters()}]
    optim = torch.optim.Adam(params, lr=args.lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optim, T_max=len(train_loader), eta_min=0, last_epoch=-1)

    MAX_mAP = 0.0
    train(args)
