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

def train(args):
    global MAX_mAP
    for epoch in range(args.Epoch):
        print('Epoch:', epoch+1, 'LR:', scheduler.get_last_lr())
        C_loss = 0.0
        for i, (img, label, _) in enumerate(train_loader):
            img, label = img.cuda(), label.cuda().float()

            optim.zero_grad()

            #Is = Norm(Crop(AugS(img)))
            #It = Norm(Crop(img))
            feat = backbone(img)
            code = hash_lay(feat)

            loss = hlss(code, hash_lay.P, label)
            loss.backward()
            optim.step()

            C_loss += loss.item()

            if (i+1) % 10 == 0 or (i+1)==len(train_loader):    # print every 10 mini-batches
                progress_bar(i, len(train_loader), f'Loss: (C: {C_loss:.3f}) | Map: (Max: {MAX_mAP:.4f})')
                C_loss = 0
            
             
        if epoch >= 10:
            scheduler.step()

        if (epoch+1) % args.eval_epoch == 0 and (epoch+1) >= args.eval_init:
            map_all, map_5000 = evaluate(query_loader, retrieval_loader, model.eval())
            logger.info(f"[Epoch {epoch+1}]: map@All: {map_all:.4f}, map@5000: {map_5000:.4f}")
            if map_all > MAX_mAP:
                MAX_mAP = map_all
            
            model.train()

if __name__ == '__main__':
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    print("Init...")
    args = parse_arguments()

    logger.add(os.path.join('log', f'{args.log_id}{args.dataset}_{args.bit}_{args.noiseRate}.log'), rotation='500MB', level='INFO')
    logger.info('Start Train')
    logger.info(args)

    query_loader, train_loader, _, retrieval_loader, cr = load_data(args)
    print(cr)
    
    AugS = Augmentation(args.img_size, 1.0)
    AugT = Augmentation(args.img_size, args.transformation_scale)

    Crop = nn.Sequential(Kg.CenterCrop(args.img_size))
    Norm = nn.Sequential(Kg.Normalize(mean=torch.as_tensor(args.mean), std=torch.as_tensor(args.std)))

    backbone, args = create_arch('ViT', args)
    backbone = backbone.cuda()
    hash_lay = Hash_func(args.fc_dim, args.bit, args.num_classes).cuda()
    criterion = SDCLoss(rec=1, rec_type='l1', quan=1, quan_type='cs', beta_ab=5, ortho_constraint=False).cuda() #True, cont=1, contrastive=SimCLRLoss(0.3
    hlss = HashProxy(args.temp)
    model = nn.Sequential(backbone, hash_lay)
    optim = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=50, gamma=0.1)

    #params = [{'params': Arch.parameters(), 'lr': args.lr}, {'params': Hash.parameters()}]
    #optim = torch.optim.Adam(params, lr=args.lr, weight_decay=1e-5)
    #scheduler = CosineAnnealingLR(optim, T_max=len(train_loader), eta_min=0, last_epoch=-1)

    MAX_mAP = 0.0
    train(args)