import argparse
import torch
import random
import os
import numpy as np

multi_labels_dataset = [
    'nuswide',
    'nuswide10k',
    'mscoco',
    'mirflickr',
    'iapr'
]

dataset_class_num = {
    'cifar10': 10,
    'nuswide': 21,
    'mscoco': 80,
    'mirflickr': 24,
    'iapr': 255,
    'nuswide10k': 10,
}

dataset_query_num = {
    'cifar10': 10000,
    'nuswide': 2100,
    'mscoco': 5000,
    'mirflickr': 2000,
    'nuswide10k': 2000,
    'iapr': 2000
} 

dataset_train_num = {
    'cifar10': 5000,
    'nuswide': 10500,
    'mscoco': 10000,
    'mirflickr': 10000,
    'nuswide10k': 8000,
    'iapr': 10000
}

def seed_torch(seed=2023):
    random.seed(seed)
    os.environ['PYTHONHASHSEES'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
def parse_arguments():
    parser = argparse.ArgumentParser(description='Label Noise Hashing with PyTorch')
    parser.add_argument('--dataset',        type=str,       default='nuswide10k',                   help='Dataset name.')
    parser.add_argument('--root',           type=str,       default='/data2/knif/dataset/',         help='Path of dataset.')
    parser.add_argument('--bit',            type=int,       default=32,                             help='Code length.')
    parser.add_argument('--num-workers',    type=int,       default=12,                             help='Threads number.')
    parser.add_argument('--Epoch',          type=int,       default=10,                             help='Epoch number.')
    parser.add_argument('--batch-size',     type=int,       default=64,                             help='Batch size.')
    parser.add_argument('--arch',           type=str,       default='ViT',                          help='backbone.')
    parser.add_argument('--noise',          action='store_true',                                    help='add noise to label.')
    parser.add_argument('--lr',             type=float,     default=0.0001,                         help='Learning rate.')
    parser.add_argument('--gpu',            type=str,       default='0',                            help='GPU ID.')
    parser.add_argument('--seed',           type=int,       default=2021,                           help='Seed.')
    parser.add_argument('--eval-epoch',     type=int,       default=2,                              help='Pre eval-epoch to eval')  
    parser.add_argument('--log-id',         type=str,       default='',                             help='log_identify.')
    parser.add_argument('--warm-up',        type=int,       default=5,                              help='Warm up the Net.')
    parser.add_argument('--tau',            type=float,     default=0.9,                            help='Threshold of the Pseudo label.')
    parser.add_argument('--theta_s',        type=float,     default=0.5,                            help='Threshold of the clean label.')
    parser.add_argument('--temp',           type=float,     default=0.1,                            help="hp_loss parameter")
    parser.add_argument('--nr',             type=float,     default=0.6,                            help="Noise Rate for label")
    parser.add_argument('--noiseType',      type=str,       default='symmetric',                    help='Noise Type')
  
    args = parser.parse_args()

    args.num_classes = dataset_class_num[args.dataset]
    args.num_query = dataset_query_num[args.dataset]
    args.num_train = dataset_train_num[args.dataset]
    args.multi_labels = args.dataset in multi_labels_dataset
    args.mean = [0.485, 0.456, 0.406]
    args.std = [0.229, 0.224, 0.225]
    args.img_size = 224
    args.K = 100
    args.theta_c = 0.8

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.device = torch.device('cuda')

    seed_torch(args.seed)

    return args