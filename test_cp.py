from __future__ import print_function
from cmath import inf
import os
import argparse
import logging

import time
import numpy as np
import xlwt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import csv

from models.resnet import *
from models.wideresnet import *

import metrics

# define the parser for the input arguments
parser = argparse.ArgumentParser(description='PyTorch CIFAR Adversarial Training')
model_options = ['resnet18', 'wrn-28-10', 'wrn-40-2']
dataset_options = ['cifar10', 'cifar100', 'tiny-imagenet']
parser.add_argument('--batch-size', type=int, default=128, metavar='N', help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N', help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float, metavar='W')
parser.add_argument('--lr_max', type=float, default=0.1, metavar='LR', help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')

parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--model-dir', default='checkpoint/res18-2', help='directory of model for saving checkpoint')
parser.add_argument('--data-dir', default='data/res18-2', help='directory of dataset')
parser.add_argument('--save-freq', '-s', default=10, type=int, metavar='N', help='save frequency')

parser.add_argument('--dataset', '-d', default='cifar10', choices=dataset_options)
parser.add_argument('--model', '-a', default='resnet18', choices=model_options)

args = parser.parse_args()
# print(args)


def append_to_csv_file(dict_data, file_path, fieldnames):
            write_header = not os.path.exists(file_path) or os.path.getsize(file_path) == 0
            with open(file_path, 'a', newline='') as file:
                
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader() if write_header else None
                # No need to write the header each time; just append rows
                writer.writerow(dict_data)
                print("Metrics data saved to", file_path)

# settings
model_dir = args.model_dir
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")

# 用在dataloader的时候加快进程 only!
kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}

# setup data loader
# transforms.RandomCrop(32, padding=4)：随机裁剪为32*32，填充为4
# transforms.RandomHorizontalFlip()：随机水平翻转
transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),])
transform_test = transforms.Compose([transforms.ToTensor(),])

if args.dataset == 'cifar10':
    trainset = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
    testset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    num_classes = 10
elif args.dataset == 'cifar100':
    trainset = torchvision.datasets.CIFAR100(root=args.data_dir, train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
    testset = torchvision.datasets.CIFAR100(root=args.data_dir, train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    num_classes = 100
    

test_labels = testset.targets
criterion = nn.CrossEntropyLoss()

print('==> Building model..')
for data, target in train_loader:
    print(data.shape)
    print(target.shape)
    break
# 测试集验证
# def eval_test(model, device, test_loader):
#     model.eval()
#     test_acc = 0
#     test_n = 0
#     test_loss = 0
#     correct = 0
#     with torch.no_grad():
#         for data, target in test_loader:
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             test_loss += F.cross_entropy(output, target, size_average=False).item()
#             pred = output.max(1, keepdim=True)[1]
#             correct += pred.eq(target.view_as(pred)).sum().item()
#             test_n += target.size(0)
#     test_time = time.time()
#     test_accuracy = correct
#     return test_loss, test_accuracy, test_n, test_time


# # learning rate decayed by epoches
# def adjust_learning_rate(optimizer, epoch):
#     """decrease the learning rate"""
#     lr = args.lr_max
#     if epoch >= 100:
#         lr = args.lr_max * 0.1
#     if epoch >= 150:
#         lr = args.lr_max * 0.01
#     if epoch >= 200:
#         lr = args.lr_max * 0.001
        
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
#     return lr

# # uselessness
# def lr_schedule(t):
#     if t / args.epochs < 0.5:
#         return args.lr_max
#     elif t / args.epochs < 0.75:
#         return args.lr_max / 10.
#     else:
#         return args.lr_max / 100.


# def main():
#     # define the logger
#     logger = logging.getLogger(__name__)
    
#     logging.basicConfig(
#         format='[%(asctime)s] - %(message)s',
#         datefmt='%Y/%m/%d %H:%M:%S',
#         level=logging.DEBUG,
#         # Sets the logging level to DEBUG. 
#         # This means that all messages at this level 
#         # and above (DEBUG, INFO, WARNING, ERROR, CRITICAL) will be captured.
        
#         # a list of the handlers to attach to the root logger.
#         handlers=[
#             logging.FileHandler(os.path.join(args.model_dir, 'output.log')),
#             logging.StreamHandler()
#         ])
#     # logger记录args
#     logger.info(args)
    
#     # define the model and optimizer
#     if args.model == 'wrn-28-10':
#         model = WideResNet(depth=28, num_classes=num_classes, widen_factor=args.width, dropRate=0.0).to(device)
#     elif args.model == 'wrn-40-2':
#         model = WideResNet(depth=40, num_classes=num_classes, widen_factor=2, dropRate=0.0).to(device)
#     elif args.model == 'resnet18':
#         model = ResNet18(num_classes=num_classes).to(device)
#     optimizer = optim.SGD(model.parameters(), lr=args.lr_max, momentum=args.momentum, weight_decay=args.weight_decay)

#     # create a excel
#     # f = xlwt.Workbook()  # init 
#     # worksheet1 = f.add_sheet('metrics')
#     # worksheet1.write(1, 1,'acc')
#     # worksheet1.write(1, 2, 'auroc')
#     # worksheet1.write(1, 3, 'aupr-s')
#     # worksheet1.write(1, 4, 'aupr-e')
#     # worksheet1.write(1, 5, 'fpr')
#     # worksheet1.write(1, 6, 'aurc')
#     # worksheet1.write(1, 7, 'e-aurc')
#     # worksheet1.write(1, 8, 'ece')
#     # worksheet1.write(1, 9, 'nll')
#     # worksheet1.write(1, 10, 'new_fpr')
    
    

#     logger.info('Epoch \t Train Time \t Test Time \t LR \t Train Loss \t Train Reg \t Train Acc \t  Test Loss \t Test Acc')
    
    
#     for epoch in range(1, args.epochs + 1):
#         temp_lr = adjust_learning_rate(optimizer, epoch)
#         model.train()
#         start_time = time.time()
#         train_loss = 0
#         train_acc = 0
#         train_reg_loss = 0
#         train_n = 0  
#         for batch_idx, (data, target) in enumerate(train_loader):
#             data, target = data.to(device), target.to(device)
#             batch_size = len(data)
#             output_clean = model(data)
#             loss = criterion(output_clean, target)
            

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             train_loss += loss.item() * target.size(0)
#             train_reg_loss = 0
#             train_acc += (output_clean.max(1)[1] == target).sum().item()
#             train_n += target.size(0)

#         train_time = time.time()
        
#         # just eval on the test set
#         test_loss, test_accuracy, test_n, test_time = eval_test(model, device, test_loader)

#         acc, auroc, aupr_success, aupr, fpr, aurc, eaurc, ece, nll, new_fpr = metrics.calc_metrics(test_loader,
#                                                                                                  model)
#         # print(acc, auroc, aupr_success, aupr, fpr, aurc, eaurc, ece, nll, new_fpr)

#         # worksheet1.write(epoch + 1, 1, str(acc))
#         # worksheet1.write(epoch + 1, 2, str(auroc))
#         # worksheet1.write(epoch + 1, 3, str(aupr_success))
#         # worksheet1.write(epoch + 1, 4, str(aupr))
#         # worksheet1.write(epoch + 1, 5, str(fpr))
#         # worksheet1.write(epoch + 1, 6, str(aurc))
#         # worksheet1.write(epoch + 1, 7, str(eaurc))
#         # worksheet1.write(epoch + 1, 8, str(ece))
#         # worksheet1.write(epoch + 1, 9, str(nll))
#         # worksheet1.write(epoch + 1, 10, str(new_fpr))
#         # f.save('result.xls')        
#         metrics_data = {
#                 "epoch": epoch,
#                 "model": args.model,
#                 "loss": loss.item(),  # Placeholder for actual loss value
#                 "tesst-acc": acc,  # Placeholder for actual accuracy value
#                 "auroc": auroc,  # Placeholder for actual AUROC value
#                 "aupr_success": aupr_success,  # Placeholder for actual AUPR for successful predictions
#                 "aupr": aupr,  # Placeholder for actual AUPR value
#                 "fpr": fpr,  # Placeholder for actual FPR value
#                 "aurc": aurc,  # Placeholder for actual AURC value
#                 "eaurc": eaurc,  # Placeholder for actual EAURC value
#                 "ece": ece,  # Placeholder for actual ECE value
#                 "nll": nll,  # Placeholder for actual NLL value
#                 "new_fpr": new_fpr  # Placeholder for actual new FPR value
#                 }  
#         file_path = model_dir + '/{}-{}-baseline.csv'.format(args.model, args.dataset)
#         append_to_csv_file(metrics_data, file_path, metrics_data.keys()) 
#         # logging the metrics
#         logger.info('%d \t \t \t %.1f \t \t %.1f \t \t %.4f \t %.4f \t %.4f \t %.4f \t \t %.4f \t \t %.4f',
#                 epoch, train_time - start_time, test_time - train_time, temp_lr,
#                 train_loss/train_n, train_reg_loss/train_n, train_acc/train_n,
#                 test_loss/test_n, test_accuracy/test_n)

#         # save checkpoint
#         if epoch % args.save_freq == 0:
#             torch.save(model.state_dict(), os.path.join(model_dir, f'model_{epoch}.pth'))
#             # torch.save(optimizer.state_dict(), os.path.join(model_dir, f'opt_{epoch}.pth'))


# if __name__ == '__main__':
#     main()
