import random
import math
import os.path as osp
import argparse

import torch
import numpy as np
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import torch_geometric.transforms as T
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader

import utils
from models import Classifier, DownClassifier

from tensorboardX import SummaryWriter

print('[INFO] Using torch', torch.__version__)

def set_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1,
                        help='seed')
    parser.add_argument('--lr_decay_steps', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=20,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0008,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--nn_hid', type=int, default=48,
                        help='hidden size')
    parser.add_argument('--nn_out', type=int, default=97,
                        help='hidden size')
    parser.add_argument('--hidden', type=int, default=128,
                        help='hidden size')
    parser.add_argument('--final_pool', type=float, default=0.6,
                        help='pooling ratio')
    
    parser.add_argument('--datadir', type=str, default='/home/zwq/data/')
    parser.add_argument('--dataset', type=str, default='DD',
                        help='DD/PROTEINS/NCI1/NCI109/Mutagenicity')
    parser.add_argument('--epochs', type=int, default=200,
                        help='maximum number of epochs')
    
    parser.add_argument('--gpu', type=int, default=1,
                        help='gpu device id')
    args = parser.parse_args()
    return args

args = set_parser()
writer = SummaryWriter(args.dataset)
##################################### Setting: Dataset ######################################
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

path = osp.join(args.datadir, args.dataset)
dataset = TUDataset(path, name=args.dataset, use_node_attr=True, transform=utils.Indegree())

num_feat = dataset.num_features
num_classes = dataset.num_classes

num_node_list = sorted([data.num_nodes for data in dataset])
s_k = num_node_list[int(math.ceil(args.final_pool * len(num_node_list))) - 1]
sk = max(10, s_k)

print("##################################################")
print("[INFO] Dataset name:", args.dataset)
print("[INFO] K use in dataset is", sk)
print("[INFO] All data exists ", len(num_node_list), "Graphs")
print("[INFO] Max node:", max(num_node_list))
print("[INFO] Min node:", min(num_node_list))
print("[INFO] num_feat| num_classes:", num_feat, num_classes)

torch.cuda.set_device(args.gpu)
device = torch.device('cuda:'+str(args.gpu))

############################### train & val ####################################
def train(model, train_loader, epoch, optimizer):
    model.train()
    total_iters = len(train_loader)
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()


def evalute(model, loader):
    model.eval()
    total_correct=0.
    total_loss=0.
    total = len(loader.dataset)
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        pred = out.max(dim=1)[1]
        total_correct += pred.eq(data.y).sum().item()
        loss = F.nll_loss(out, data.y)
        # crossentropy(reduce=True) for default
        total_loss += loss.item() * len(data.y)
    loss, acc = 1.0*total_loss / total, 1.0*total_correct / total
    model.train()
    return loss, acc


def main():
    for fold in range(10):
        print("="*50)
        print("[INFO] Fold:", fold)
        print("="*50)
        train_idx, val_idx = utils.sep_data(dataset, seed=args.seed, fold_idx=fold)
        train_idx = torch.LongTensor(train_idx)
        val_idx = torch.LongTensor(val_idx)
        train_dataset = dataset[train_idx]
        val_dataset = dataset[val_idx]
        print("[INFO] Train:", len(train_idx), "| Val:", len(val_idx))
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        
        model = Classifier(
            node_feat=num_feat,
            nn_hid=args.nn_hid,
            nn_out=args.nn_out,
            k=sk,
            hidden=args.hidden,
            num_class=num_classes
        ).to(device)

        print(model)

        optimizer = torch.optim.SGD([
            {'params': model.feature_extractor.parameters()},
            {'params': model.mlp.parameters(), 'lr':0.001},
            {'params': model.readout.parameters(), 'lr': 0.05}
        ], lr=args.lr, momentum=0.9)
        scheduler = lr_scheduler.MultiStepLR(optimizer, [50,100,150], gamma=0.1)
        best = 0.
        ######################### training #####################
        for epoch in range(args.epochs):
            scheduler.step()
            train(model, train_loader, epoch, optimizer)
            train_loss, train_acc = evalute(model, train_loader)
            val_loss, val_acc = evalute(model, val_loader)
            writer.add_scalars('%s/loss' % args.dataset, {'train_loss':train_loss, 'val_loss': val_loss}, epoch)
            writer.add_scalars('%s/acc' % args.dataset, {'train_acc': train_acc, 'val_acc':  val_acc}, epoch)
            print("Epoch:{:05d} | TrainLoss:{:.4f} | TrainAcc:{:.4f} |"
              " ValLoss:{:.4f} ValAcc:{:.4f}".
              format(epoch, train_loss, train_acc,
                     val_loss, val_acc))
            best = best if best > val_acc else val_acc
            print("[INFO] Best Acc:", best)
        with open('results/ACC_0904_%s.txt' % args.dataset, 'a+') as f:
            f.write(str(best) + '\n')


if __name__ == "__main__":
    main()











