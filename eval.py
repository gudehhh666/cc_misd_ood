import torch
import metrics
import torch.nn as nn
import models.model as model
import torchvision
import argparse
import torchvision.transforms as transforms
import csv
import os
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
print(args)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



transform_test = transforms.Compose([transforms.ToTensor(),])

# trainset = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform_train)
# train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
testset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False)
num_classes = 10
transform_test = transforms.Compose([transforms.ToTensor(),])

model = model.ResNet18(num_classes).to(device)

# dict = torch.load('checkpoint/res18-/model_200.pth')
model_dict = torch.load('checkpoint/_epoch_200.pt', map_location=device)
model.load_state_dict(model_dict)
print('load model successfully')

acc, auroc, aupr_success, aupr, fpr, aurc, eaurc, ece, nll, new_fpr = metrics.calc_metrics(test_loader,
                                                                                                 model=model,)
    

    
    
# Define the metrics to be saved
metrics_data = {
                "model": args.model,
                "dataset": args.dataset,
                "ratio": 0,
                "acc": acc,  # Placeholder for actual accuracy value
                "auroc": auroc,  # Placeholder for actual AUROC value
                "aupr_success": aupr_success,  # Placeholder for actual AUPR for successful predictions
                "aupr": aupr,  # Placeholder for actual AUPR value
                "fpr": fpr,  # Placeholder for actual FPR value
                "aurc": aurc,  # Placeholder for actual AURC value
                "eaurc": eaurc,  # Placeholder for actual EAURC value
                "ece": ece,  # Placeholder for actual ECE value
                "nll": nll,  # Placeholder for actual NLL value
                "new_fpr": new_fpr  # Placeholder for actual new FPR value
                }       



    # Specify the file path where you want to store the data
file_path = file_path = './checkpoint/' + args.model + args.dataset + '.csv'
    
def append_to_csv_file(dict_data, file_path, fieldnames):
    write_header = not os.path.exists(file_path) or os.path.getsize(file_path) == 0
    with open(file_path, 'a', newline='') as file:
        
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader() if write_header else None
        # No need to write the header each time; just append rows
        writer.writerow(dict_data)
        print("Metrics data saved to", file_path)
    
append_to_csv_file(metrics_data, file_path, metrics_data.keys())            
