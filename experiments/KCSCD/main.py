import torch
import numpy as np
import random
import pandas as pd
import argparse
from train import train
from sklearn.model_selection import train_test_split

def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=1,help='The id of gpu, e.g. 0.')
parser.add_argument('--epoch_n', type=int, default=200, help='The epoch number of training')
# parser.add_argument('--dataset', type=str, default="ASSIST17")
parser.add_argument('--dataset', type=str, default="junyi")
# parser.add_argument('--dataset', type=str, default="ASSIST09")
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--d_type', type=str, default="validation")
parser.add_argument('--isAffect', type=bool, default=True)
parser.add_argument('--weight_decay',type=float, default=1e-5, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--alpha', type=int, default=1.0, help='loss weight')


if __name__ == '__main__':
    args = parser.parse_args()
    dataset = args.dataset
    # split_dataset(dataset, 0.1, 0.1, 42)
    if dataset in {"ASSIST17"}:
        args.batch_size = 256
        args.data_dir = 'data/' + dataset + '/seed42/'
        args.train_set = 'train_data.json'
        args.val_set = 'val_data.json'
        args.test_set = 'test_data.json'
        args.n_skill = 102
        args.n_exercise = 3162
        args.n_student = 1709

    if dataset in {"junyi"}:
        args.batch_size = 256
        args.data_dir = 'data/' + dataset + '/seed42/'
        args.train_set = 'train_data.json'
        args.val_set = 'val_data.json'
        args.test_set = 'test_data.json'
        args.n_skill = 835
        args.n_exercise = 835
        args.n_student = 10000

    if dataset in {"ASSIST09"}:
        args.batch_size = 256
        args.data_dir = 'data/' + dataset + '/seed42/'
        args.train_set = 'train_data.json'
        args.val_set = 'val_data.json'
        args.test_set = 'test_data.json'
        args.n_skill = 123
        args.n_exercise = 13016
        args.n_student = 4163

    seed_everything(25)
    train(args)
