import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import pickle
import torchvision
import torchvision.transforms as transforms
from models import resnet
from data.dataset import CIFAR10_loss


# load_list = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']

# data = []
# targets = []
# # for load_name in load_list:
# #     path = os.path.join('data/res18-2/cifar-10-batches-py', load_name)
# #     with open(path, "rb") as f:
# #         entry = pickle.load(f, encoding="latin1")
# #         data.append(entry["data"])
# #         if "labels" in entry:
# #             targets.extend(entry["labels"])
# #         else:
# #             targets.extend(entry["fine_labels"])
                    
# with open('data/res18-2/cifar-10-batches-py/batches.meta', "rb") as f:
#     entry = pickle.load(f, encoding="latin1")
                    
# print(entry.keys())
# print(entry['label_names'])
# print(entry['num_cases_per_batch'])
# print(entry['num_vis'])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),])
transform_test = transforms.Compose([transforms.ToTensor(),])
# transform_test = transforms.Compose([])
data_dir = 'data/res18-2/'
batch_size = 64
test_batch_size = 64
kwargs = {'num_workers': 0, 'pin_memory': True}
trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, **kwargs)
testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False, **kwargs)
num_classes = 10

model = resnet.ResNet18(num_classes=num_classes).to(device)
model.load_state_dict(torch.load(r'D:\CASIA\code_cc_misd\checkpoint\res18-2\model_200.pth', map_location=device))

cretirion = nn.CrossEntropyLoss(reduction='none')

loss_list = []
for idx, (input, target) in enumerate(train_loader):
    input = input.to(device)
    target = target.to(device)
    output = model(input)
    with open('output.pkl', 'wb') as f:
        pickle.dump(output, f)
        break
    loss = cretirion(output, target)
    print(loss.data)
    # loss_list.extend(loss.data.cpu().numpy())
    break
# # save the loss_list in a file
# with open('loss_trained_new.pkl', 'wb') as f:
#     pickle.dump(loss_list, f)
# print('loss_list saved')


