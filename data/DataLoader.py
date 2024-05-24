from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torchnet.meter import AUCMeter
from PIL import Image
from helper.noise import *
from data.trans_img import *

clean_label = []
train_img = []

class TransformWeak(object):
    def __init__(self, mean, std):
        self.trans = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=224, 
                                  padding=int(224*0.125),
                                  padding_mode='reflect')
        ])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    
    def __call__(self, x):
        return self.normalize(self.trans(x))

def load_data(args):
    root = args.root
    dataset = args.dataset
    noiseRate = args.nr
    noiseType = args.noiseType
    batch_size = args.batch_size
    num_workers = args.num_workers
    query_dataset = Multi_Dataset(root, dataset, train='test', transform=val_img_trans)
    train_dataset = Multi_Dataset(root, dataset, args.noise, noiseRate, noiseType, train='train', transform=train_transform(), transform_aug=train_aug_transform())
    retrieval_dataset = Multi_Dataset(root, dataset, train='database', transform=val_img_trans)

    query_loader = DataLoader(
        query_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    eval_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    retrieval_loader = DataLoader(
        retrieval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers 
    )

    return train_dataset, query_loader, train_loader, eval_loader, retrieval_loader, torch.tensor(clean_label).cuda().float(), train_img

class Multi_Dataset(Dataset):
    def __init__(self, root, dataset, noise=True, noiseRate=0, noiseType='symmetric', random_state=1, transform=None, transform_aug=None, train='train'):
        global clean_label, train_img
        self.root = root
        self.transform = transform
        self.transform_aug = transform_aug
        multi_datas = np.load(root+dataset+'/'+train+'.npz')
        self.img = multi_datas['img']
        self.label = multi_datas['label']
        self.train = train

        if train=='train':
            Multi_Dataset.Ground_Truth = self.label
            train_img = self.img
            clean_label = Multi_Dataset.Ground_Truth
            if noise : self.label, actual_noise_rate = noisify_dataset(nb_classes=self.label.shape[1], train_labels=self.label, noise_type=noiseType,
                    closeset_noise_ratio=noiseRate, random_state=random_state, verbose=True)
            #self.label, actual_noise_rate = generate_noisy_labels(self.label, noiseRate, random_state)
            Multi_Dataset.Noise_Label = self.label
            clean = np.all(Multi_Dataset.Ground_Truth == Multi_Dataset.Noise_Label, axis=1)
            print('Clean Rate : ', clean.sum()/len(self.label))
            # print('Noise Rate : ', actual_noise_rate)

        self.len = len(self.label)

    def __getitem__(self, index):
        img = self.img[index]
        img = Image.fromarray(img)
        img1 = self.transform(img)
        if self.train=='train': img2 = self.transform_aug(img)
        if self.train=='train': return img1, img2, self.label[index], index
        else: return img1, self.label[index]

    def __len__(self):
        return self.len

class Sub_Dataset(Dataset):
    def __init__(self, img, label, ind, transform=train_transform(), transform_aug=train_aug_transform()):
        self.img = img
        self.label = label
        self.ind = ind
        self.trans = transform
        self.trans_aug = transform_aug
        self.len = len(self.label)
    
    def __getitem__(self, index):
        img = self.img[index]
        img = Image.fromarray(img)
        ind = self.ind[index]
        img1 = self.trans(img)
        img2 = self.trans_aug(img)
        return img1, img2, self.label[index], ind 

    def __len__(self):
        return self.len