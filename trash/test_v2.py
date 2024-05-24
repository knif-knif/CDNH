import torch
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

from helper.settings import *
from helper.losses import *
from helper.evaluate import evaluate_hx
from helper.bar_show import progress_bar
from net.backbones import *
from net.hashfunc import *
from data.DataLoader import *

def train(args):
    global MAX_mAP
    for epoch in range(args.Epoch):
        print('Epoch:', epoch+1, 'LR:', scheduler.get_last_lr())
        C_loss = 0.0
        L_loss = 0.0
        for i, (img1, img2, label, _) in enumerate(train_loader):
            img1, img2 = img1.cuda(), img2.cuda()
            label = label.cuda().float()
            optim.zero_grad()

            feat1 = backbone(img1)
            code1 = hash_encode(feat1)
            feat2 = backbone(img2)
            code2 = hash_encode(feat2)
            prob1 = torch.sigmoid(code1)
            prob2 = torch.sigmoid(code2)
            z1 = hash_layer(prob1 - 0.5)
            z2 = hash_layer(prob2 - 0.5)
            co_loss = criterion(z1, z2, args.device)
            hl_loss = hlss(prob1, hash_encode.P, label)
            loss = co_loss #*0.03 + hl_loss #+ kl_loss * 0.001

            loss.backward()
            optim.step()

            C_loss += co_loss.item()
            L_loss += hl_loss.item()

            if (i+1) % 10 == 0 or (i+1)==len(train_loader):    # print every 10 mini-batches
                progress_bar(i, len(train_loader), f'Loss: (C: {C_loss:.3f}, L: {L_loss:.3f}) | Map: (Max: {MAX_mAP:.4f})')
                C_loss = 0
                L_loss = 0
            
             
        if epoch >= 10:
            scheduler.step()

        if (epoch+1) % args.eval_epoch == 0 and (epoch+1) >= args.eval_init:
            map_all, map_5000 = evaluate_hx(query_loader, retrieval_loader, model.eval())
            logger.info(f"[Epoch {epoch+1}]: map@All: {map_all:.4f}, map@5000: {map_5000:.4f}")
            if map_all > MAX_mAP:
                MAX_mAP = map_all
            
            model.train()

if __name__ == '__main__':
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    print("Init...")
    args = parse_arguments()

    logger.add(os.path.join('log', f'{args.log_id}{args.dataset}_{args.bit}_{args.nr}.log'), rotation='500MB', level='INFO')
    logger.info('Start Train')
    logger.info(args)

    _,query_loader, train_loader, _, retrieval_loader, cr = load_data(args)
    print(cr)

    backbone, args = create_arch('ViT', args)
    backbone = backbone.cuda()
    hash_encode = Hash_func(args.fc_dim, args.bit, args.num_classes).cuda()
    criterion = NtXentLoss()
    hlss = HashProxy(args.temp)
    model = nn.Sequential(backbone, hash_encode)
    optim = torch.optim.Adam([{'params': hash_encode.parameters()}], lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=80, gamma=0.1)

    MAX_mAP = 0.0
    train(args)