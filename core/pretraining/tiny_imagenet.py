'''
Pre-training with PyTorch.
code source: https://github.com/kuangliu/pytorch-cifar
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader

import os
import argparse
from tqdm import tqdm
import sys
sys.path.append('../')
import pretraining_config
import utils
import time

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

transform_test = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

trainset = datasets.ImageFolder('../../../pretrain_dataset/tiny-imagenet-200/train', transform=transform_train)
trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=32)

testset = datasets.ImageFolder('../../../pretrain_dataset/tiny-imagenet-200/val', transform=transform_test)
testloader = DataLoader(testset, batch_size=128, shuffle=True, num_workers=32)

# Model
print('==> Building model..')
net = utils.get_model('Cloud', pretraining_config.model_name, 0, device, pretraining_config.model_cfg)
print(net)
net = net.to(device)
#if 'cuda' in device:
    #net = torch.nn.DataParallel(net)
    #cudnn.benchmark = True

'''
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
'''

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4, nesterov=True)
epochs = 100
milestones = [50, 80]  # Adjust these milestones as needed
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

# Training
def train(epoch):
    global best_acc
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader)):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        #torch.save(net.state_dict(), '../../../pretrained/'+ str(pretraining_config.model_name).lower() +'_tiny-imagenet_32.pth')
        best_acc = acc


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(testloader)):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            #print(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         #% (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    acc = 100.*correct/total
    print('Val Accuracy: {}'.format(acc))

start_time = time.time()
for epoch in range(start_epoch, start_epoch+100):
    train(epoch)
    #test(epoch)
    scheduler.step()
end_time = time.time()
training_time = end_time - start_time
test(epoch)
print(f"Best accuracy {best_acc}")
print(f"Training took {training_time} seconds")