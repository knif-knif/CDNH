import torch
from loguru import logger
from sklearn.mixture import GaussianMixture
from torch.utils.data import Subset
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

from net.backbones import *
from net.hashfunc import *
from helper import *
import torch.nn.functional as F
from data.DataLoader import load_data, Sub_Dataset

def warm(args, epoch):
    global MAX_mAP, cos_metrix, train_loader, eval_loader, net1, net2, opt1, opt2, sl1, sl2
    net1.train()
    net2.train()
    print('Epoch:', epoch+1, 'LR:', sl1.get_last_lr(), sl2.get_last_lr())
    codes = []
    for i, (img, img_aug, label, index) in enumerate(train_loader):
        img, img_aug, label = img.cuda(), img_aug.cuda(), label.cuda().float()
        opt1.zero_grad()
        _, code, code_h, prob = net1(img)
        _, code_aug, code_ah, prob_aug = net1(img_aug)
        code_m = F.normalize(code, dim=1)
        u_loss = D(code_ah, code)/2 + D(code_h, code_aug)/2
        loss = criterion_n(prob, label)/2 + criterion_n(prob_aug, label)/2 + u_loss + pnorm(prob)
        loss.backward()
        opt1.step()
        if (i+1) % 10 == 0 or (i+1)==len(train_loader):
            progress_bar(i, len(train_loader), f'{loss.item()} >> {u_loss.item()}(Warm)')

    with torch.no_grad():
        for i, (img, _, _, _) in enumerate(eval_loader):
            img = img.cuda()
            _, code, _, _ = net1(img)
            code_m = F.normalize(code, dim=1)
            if i==0: codes = code_m
            else: codes = torch.cat([codes, code_m], 0)

    cos_metrix = torch.mm(codes, codes.T).detach()

    for i, (img, img_aug, label, index) in enumerate(train_loader):
        img, img_aug, label = img.cuda(), img_aug.cuda(), 1-label.cuda().float()
        opt2.zero_grad()
        _, code, code_h, prob = net2(img)
        _, code_aug, code_ah, prob_aug = net2(img_aug)
        code_m = F.normalize(code, dim=1)
        cos_m = torch.mm(code_m, code_m.T)
        u_loss = D(code_ah, code)/2 + D(code_h, code_aug)/2
        d_loss = D(cos_m, cos_metrix[index][:,index]) 
        loss = criterion_n(prob, label)/2 + criterion_n(prob_aug, label)/2 + d_loss + u_loss + pnorm(prob)
        loss.backward()
        opt2.step()
        if (i+1) % 10 == 0 or (i+1)==len(train_loader):
            progress_bar(i, len(train_loader), f'{loss.item()} >> {u_loss.item()} >> {d_loss.item()}(Warm)')
    sl1.step()
    sl2.step()

def train(args, epoch):
    global MAX_mAP, cos_metrix, train_loader, vague_loader, modify_label, net1, net2, opt1, opt2, sl1, sl2
    print('Epoch:', epoch+1, 'LR:', sl1.get_last_lr(), sl2.get_last_lr())
    C_loss = 0.0
    L_loss = 0.0
    D_loss = 0.0
    net1.train()
    net2.train()
    codes = []
    for i, (img, _, _, index) in enumerate(train_loader):
        
        label = modify_label[index]
        img = img.cuda()
        label = label.cuda().float()
        opt2.zero_grad()
        _, code, code_h, prob = net2(img)
        code_m = F.normalize(code, dim=1)
        cos_metrix[index][:,index] = torch.mm(code_m, code_m.T)
        co_loss = D(code_h, code)
        hl_loss = criterion_n(prob, 1-label) + pnorm(prob)
        loss = hl_loss + co_loss
        loss.backward()
        torch.nn.utils.clip_grad_value_(net2.parameters(), clip_value=0.5)
        opt2.step()

        C_loss += co_loss.item()
        L_loss += hl_loss.item()

        if (i+1) % 10 == 0 or (i+1)==len(train_loader):
            progress_bar(i, len(train_loader), f'Loss: (C: {C_loss/(i+1):.3f}, L: {L_loss/(i+1):.3f}) | Map: (Max: {MAX_mAP:.4f})')

    with torch.no_grad():
        for i, (img, _, _, _) in enumerate(eval_loader):
            img = img.cuda()
            _, code, _, _ = net2(img)
            code_m = F.normalize(code, dim=1)
            if i==0: codes = code_m
            else: codes = torch.cat([codes, code_m], 0)

    cos_metrix = torch.mm(codes, codes.T).detach()

    for i, (img, _, _, index) in enumerate(train_loader):
        img = img.cuda()
        label = modify_label[index]
        label = label.cuda().float()
        opt1.zero_grad()
        _, code, code_h, prob = net1(img)
        code_m = F.normalize(code, dim=1)
        cos_m = torch.mm(code_m, code_m.T)
        co_loss = D(code_h, code)
        hl_loss = criterion_n(prob, label) + pnorm(prob)
        ds_loss = D(cos_m, cos_metrix[index][:,index])
        loss = hl_loss + co_loss + ds_loss
        loss.backward()
        torch.nn.utils.clip_grad_value_(net1.parameters(), clip_value=0.5)
        opt1.step()

        C_loss += co_loss.item()
        L_loss += hl_loss.item()
        D_loss += ds_loss.item()

        if (i+1) % 10 == 0 or (i+1)==len(train_loader):
            progress_bar(i, len(train_loader), f'Loss: (C: {C_loss/(i+1):.3f}, L: {L_loss/(i+1):.3f}, D: {D_loss/(i+1):.3f}) | Map: (Max: {MAX_mAP:.4f})')

    sl1.step()
    sl2.step()

def modify(args, epoch):
    global modify_label, net1, net2, eval_loader, clean_label
    net1.eval()
    net2.eval()
    prediction = []
    org_label = []
    losses = torch.zeros(args.num_train)
    with torch.no_grad():
        for i, (img, _, _, ind) in enumerate(eval_loader):
            img = img.cuda()
            label = modify_label[ind].cuda().float()
            org_label.append(label)
            _, _, _, pred = net1(img)
            prediction.append(pred)
            if (i+1) % 10 == 0 or (i+1)==len(eval_loader):    # print every 10 mini-batches
                progress_bar(i, len(eval_loader), f"(Evaluate)")
        
        for i, (img, _, _, ind) in enumerate(eval_loader):
            img = img.cuda()
            label = 1-modify_label[ind].cuda().float()
            _, _, _, pred = net2(img)
            for b in range(img.size(0)):
                losses[ind[b]] = criterion_n(pred[b].reshape(1, args.num_classes), label[b].reshape(1, args.num_classes))
            if (i+1) % 10 ==0 or (i+1)==len(eval_loader):
                progress_bar(i, len(eval_loader), f"(Evaluate)")

        org_label = torch.cat(org_label, dim=0)
        # relabeling
        prediction = torch.cat(prediction, dim=0)
        prediction = torch.softmax(prediction/args.temp, dim=1)
        pseudo_label, conf_id = pseudo_s(prediction, args.tau, org_label)
        pseudo_label = pseudo_label.cpu()
        losses = ((losses-losses.min()) / (losses.max() - losses.min())).reshape(-1, 1)
        gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
        gmm.fit(losses)
        prob = gmm.predict_proba(losses)
        argi = gmm.means_[:,0].argsort()
        clean_id = conf_id
        vague_id = torch.where(torch.tensor(prob[:, argi[1]])>=args.theta_s)[0]
        noise_id = torch.where(torch.tensor(prob[:, argi[0]])>=args.theta_s)[0]

        TP = torch.sum(torch.all(pseudo_label[clean_id] == clean_label[clean_id], dim=1)).item()
        FP = torch.sum(torch.any(pseudo_label[clean_id] != clean_label[clean_id], dim=1)).item()
        TN = torch.sum(torch.any(pseudo_label[noise_id] != clean_label[noise_id], dim=1)).item()
        FN = torch.sum(torch.all(pseudo_label[noise_id] == clean_label[noise_id], dim=1)).item()

        VN = torch.sum(torch.all(pseudo_label[vague_id] == clean_label[vague_id], dim=1)).item()
        
        logger.info(f"TP:{TP}  FP:{FP}  TN:{TN}  FN:{FN}  VN:{VN}")

        correct = torch.sum(torch.all(pseudo_label == clean_label, dim=1)).item()
        conf_acc = torch.sum(torch.all(pseudo_label[conf_id] == clean_label[conf_id], dim=1)).item()
        logger.info(f"Correct:{correct}  Clean:{len(clean_id)}  Conf:{len(conf_id)}  Conf_Acc:{conf_acc}  Vague:{len(vague_id)}")
    net1.train()
    net2.train()
    return clean_id, vague_id, noise_id, pseudo_label.detach()
            
if __name__ == '__main__':
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    args = parse_arguments()
    os.makedirs('./log/' + args.log_id, exist_ok=True)
    logger.add(os.path.join('log/' + args.log_id, f'{args.dataset}_{args.bit}_{args.nr}.log'), 
               format="<green>{time:YYYY-MM-DD HH:mm}</green> | <level>{level: <4}</level> | - <level>{message}</level>",
               rotation='500MB', level='INFO')
    logger.info(args)

    train_data, query_loader, train_loader, eval_loader, retrieval_loader, clean_label, train_img = load_data(args)
    clean_label = clean_label.cpu()
    arch1, args = create_arch('ViT', args)
    arch2, args = create_arch('ViT', args)

    net1 = Hash_func(arch1, args.fc_dim, args.bit, args.num_classes).cuda()#torch.load('warm1.pt')# # 
    net2 = Hash_func(arch2, args.fc_dim, args.bit, args.num_classes).cuda()#torch.load('warm2.pt')# # 
    criterion_r = AGCELoss()
    criterion_n = nn.CrossEntropyLoss()
    criterion_m = nn.MSELoss()
    pnorm = pNorm()
    opt1 = torch.optim.SGD(net1.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-5)
    opt2 = torch.optim.SGD(net2.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-5)
    sl1 = torch.optim.lr_scheduler.StepLR(opt1, step_size=5, gamma=0.1)
    sl2 = torch.optim.lr_scheduler.StepLR(opt2, step_size=5, gamma=0.1)

    MAX_mAP = 0.0
    args.temp = 0.2

    modify_label = torch.tensor(eval_loader.dataset.Noise_Label)
    cos_metrix = torch.zeros(args.num_train, args.num_train).cuda()
    for epoch in range(0, args.Epoch):
        logger.info(f"[Epoch {epoch+1}]:")
        if epoch%2==0:
            warm(args, epoch)
            #_, _, _, modify_label = modify(args, epoch)
            # torch.save(net1, 'warm1.pt')
            # torch.save(net2, 'warm2.pt')
        else:
            # conf_subset = Sub_Dataset(train_img[clean_id], modify_label[clean_id].cpu(), clean_id)
            # conf_loader = torch.utils.data.DataLoader(conf_subset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
            #vague_subset = Sub_Dataset(train_img[vague_id], modify_label[vague_id].cpu(), vague_id)
            #vague_loader = torch.utils.data.DataLoader(vague_subset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
            # noise_subset = Sub_Dataset(train_img[noise_id], modify_label[noise_id].cpu(), noise_id)
            # noise_loader = torch.utils.data.DataLoader(noise_subset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
            train(args, epoch)
            #_, _, _, modify_label = modify(args, epoch)
        
        if (epoch+1) % args.eval_epoch == 0:
            map_all, map_5000 = evaluate_hx(query_loader, retrieval_loader, net1.eval())
            logger.info(f"A map@All: {map_all:.4f}, map@5000: {map_5000:.4f}\n\n")
            if map_all > MAX_mAP:
                MAX_mAP = map_all
            map_all, map_5000 = evaluate_hx(query_loader, retrieval_loader, net2.eval())
            logger.info(f"B map@All: {map_all:.4f}, map@5000: {map_5000:.4f}\n\n")
        
        net1.train()
        net2.train()