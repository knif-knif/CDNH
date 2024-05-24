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
    global MAX_mAP
    print('Epoch:', epoch+1, 'LR:', sl1.get_last_lr())

    for i, (img, img_aug, label, _) in enumerate(train_loader):
        img, img_aug, label = img.cuda(), img_aug.cuda(), label.cuda().float()
        opt1.zero_grad()
        _, code, _, prob = net1(img)
        _, code_aug, _, prob_aug = net1(img_aug)
        #label = label/torch.sum(label, dim=1, keepdim=True).expand_as(label)
        u_loss = 0#criterion_d(code)+criterion_d(code_aug)#D(code_ah, code)/2 + D(code_h, code_aug)/2
        loss = criterion_n(prob, label)/2 + criterion_n(prob_aug, label)/2 #+ u_loss
        loss.backward()
        opt1.step()
        if (i+1) % 10 == 0 or (i+1)==len(train_loader):    # print every 10 mini-batches
            progress_bar(i, len(train_loader), f'{loss.item()} >> {u_loss.item()}(Warm)')

def train(args, epoch, model, optim, sl):
    global MAX_mAP
    print('Epoch:', epoch+1, 'LR:', sl1.get_last_lr())
    C_loss = 0.0
    L_loss = 0.0
    cnt = 0
    conf_iter = iter(conf_loader)
    noise_iter = iter(noise_loader)
    for i, (img_u, _, _, ind_u) in enumerate(vague_loader):
        try:
            img_x, _, _, index = next(conf_iter)
            img_n, _, _, _ = next(noise_iter)
        except:
            conf_iter = iter(conf_loader)
            noise_iter = iter(noise_loader)
            img_x, _, _, index = next(conf_iter)
            img_n, _, _, _ = next(noise_iter)
        
        label_x = modify_label[index]
        label_u = modify_label[ind_u]
        #label_n = modify_label[ind_n]

        img_u = img_u.cuda()
        img_x = img_x.cuda()
        img_n = img_n.cuda()
        label_x = label_x.cuda().float()
        label_u = label_u.cuda().float()
        label = torch.cat([label_x, label_u], 0)
        img = torch.cat([img_x, img_u], 0)
        optim.zero_grad()
        _, code, code_h, _ = model(img_n)
        _, code_x, _, prob_x = model(img)
        sim = calculate_similarity(label, label)
        co_loss = negative_log_likelihood_similarity_loss(code_x, code_x, sim) + D(code_h, code)
        hl_loss = criterion_r(prob_x, label)
        loss = hl_loss + co_loss
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
        optim.step()

        C_loss += co_loss.item()
        L_loss += hl_loss.item()

        if (i+1) % 10 == 0 or (i+1)==len(vague_loader):    # print every 10 mini-batches
            progress_bar(i, len(vague_loader), f'Loss: (C: {C_loss/(i+1):.3f}, L: {L_loss/(i+1):.3f}) | Map: (Max: {MAX_mAP:.4f})')

        
    sl.step()

def modify(args, epoch, model):
    model.eval()
    prediction = []
    org_label = []
    losses = torch.zeros(args.num_train)
    with torch.no_grad():
        for i, (img, _, _, ind) in enumerate(eval_loader):
            img = img.cuda()
            label = modify_label[ind].cuda().float()
            org_label.append(label)
            _, _, _, pred = model(img)
            for b in range(img.size(0)):
                losses[ind[b]] = criterion_n(pred[b].reshape(1, args.num_classes), label[b].reshape(1, args.num_classes)) #if tp else D(code[b].reshape(1, args.bit), code_h[b].reshape(1, args.bit))
            prediction.append(pred)
            if (i+1) % 10 == 0 or (i+1)==len(eval_loader):    # print every 10 mini-batches
                progress_bar(i, len(eval_loader), f"(Evaluate)")
        
        org_label = torch.cat(org_label, dim=0)
        # relabeling
        prediction = torch.cat(prediction, dim=0)

        prediction = torch.softmax(prediction/args.temp, dim=1)
        pseudo_label, conf_id = pseudo_s(prediction, args.tau, org_label)

        losses = ((losses-losses.min()) / (losses.max() - losses.min())).reshape(-1, 1)

        gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
        gmm.fit(losses)
        prob = gmm.predict_proba(losses)
        prob = prob[:, gmm.means_.argmin()]
        prob = torch.tensor(prob)
        vague_id = torch.where(prob>=args.theta_s)[0]
        while len(vague_id)==0: 
            args.theta_s -= 0.1
            vague_id = torch.where(prob>=args.theta_s)[0]
        noise_id = torch.where(prob<args.theta_s)[0]
        clean_id = conf_id
        # gmm = GaussianMixture(n_components=3, max_iter=10, tol=1e-2, reg_covar=5e-4)
        # gmm.fit(losses)
        # prob = gmm.predict_proba(losses)
        # prob = prob[:, gmm.means_.argmin()]
        # prob = torch.tensor(prob)
        # clean_id = torch.where(prob<=prob.mean())[0]
        
        TP = torch.sum(torch.all(pseudo_label[clean_id] == clean_label[clean_id], dim=1)).item()
        FP = torch.sum(torch.any(pseudo_label[clean_id] != clean_label[clean_id], dim=1)).item()
        TN = torch.sum(torch.any(pseudo_label[noise_id] != clean_label[noise_id], dim=1)).item()
        FN = torch.sum(torch.all(pseudo_label[noise_id] == clean_label[noise_id], dim=1)).item()
        
        logger.info(f"TP:{TP}  FP:{FP}  TN:{TN}  FN:{FN}")

        correct = torch.sum(torch.all(pseudo_label == clean_label, dim=1)).item()
        conf_acc = torch.sum(torch.all(pseudo_label[conf_id] == clean_label[conf_id], dim=1)).item()
        logger.info(f"Correct:{correct}  Clean:{len(clean_id)}  Conf:{len(conf_id)}  Conf_Acc:{conf_acc}  Vague:{len(vague_id)}")

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

    #arch1, args = create_arch('ViT', args)
    #arch2, args = create_arch('ViT', args)
    #model = torch.load('warm.pt').cuda()

    net1 = torch.load('warm1.pt')#Hash_func(arch1, args.fc_dim, args.bit, args.num_classes).cuda()
    #net2 = torch.load('warm2.pt')#Hash_func(arch2, args.fc_dim, args.bit, args.num_classes).cuda()
    criterion_r = AGCELoss()
    criterion_n = nn.CrossEntropyLoss()
    opt1 = torch.optim.SGD(net1.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-5)
    #opt2 = torch.optim.SGD(net2.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-5)
    sl1 = torch.optim.lr_scheduler.StepLR(opt1, step_size=3, gamma=0.3)
    #sl2 = torch.optim.lr_scheduler.StepLR(opt2, step_size=10, gamma=0.3)

    MAX_mAP = 0.0
    args.temp = 0.1

    modify_label = torch.tensor(eval_loader.dataset.Noise_Label)
    for epoch in range(5, args.Epoch):
        logger.info("\n\n")
        logger.info(f"[Epoch {epoch+1}]:")
        if epoch<args.warm_up:
            warm(args, epoch)
            torch.save(net1, 'warm1.pt')
            #torch.save(net2, 'warm2.pt')
        else:
            clean_id, vague_id, noise_id, modify_label = modify(args, epoch, net1)
            conf_subset = Sub_Dataset(train_img[clean_id], modify_label[clean_id].cpu(), clean_id)
            vague_subset = Sub_Dataset(train_img[vague_id], modify_label[vague_id].cpu(), vague_id)
            noise_subset = Sub_Dataset(train_img[noise_id], modify_label[noise_id].cpu(), noise_id)
            conf_loader = torch.utils.data.DataLoader(conf_subset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
            vague_loader = torch.utils.data.DataLoader(vague_subset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
            noise_loader = torch.utils.data.DataLoader(noise_subset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
            train(args, epoch, net1, opt1, sl1)
        
        if (epoch+1) % args.eval_epoch == 0:
            map_all, map_5000 = evaluate_hx(query_loader, retrieval_loader, net1.eval())
            logger.info(f"map@All: {map_all:.4f}, map@5000: {map_5000:.4f}")
            if map_all > MAX_mAP:
                MAX_mAP = map_all
        
        net1.train()