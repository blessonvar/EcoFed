'''Some helper functions and classes for ActionFed, including:
	- get_local_dataloader: split dataset and get respective dataloader.
	- get_model: build the model according to location and split layer.
	- get_compressor: build autoencoder-based compressor.
	- extract_ae_weights: extract encoder weights or decoder weights from holistic weights.
	- split_weights_client: split client's weights from holistic weights.
	- transfer_weights_client: initialize client's weights from pretrained weights.
	- split_weights_server: split server's weights from holistic weights.
	- concat_weights: concatenate server's weights and client's weights.
	- zero_init: zero initialization.
	- fed_avg: FedAvg aggregation.
	- minmax_quant: 8 bits minmax quantization.
	- dequant: Dequantization.
	- KL_Loss: KL loss class.
	- CE_Loss: CE loss class.
'''
import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F


import pickle, struct, socket
from cnn import *
from config import *
import collections
import numpy as np

import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_local_dataloader(CLIENT_IDEX, cpu_count, dataset):
	indices = list(range(N))
	part_tr = indices[int((N/(K*C)) * CLIENT_IDEX) : int((N/(K*C)) * (CLIENT_IDEX+1))]

	if dataset == 'CIFAR10':
		transform_train = transforms.Compose([
		transforms.RandomCrop(32, padding=4),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
		trainset = torchvision.datasets.CIFAR10(
			root=dataset_path, train=True, download=True, transform=transform_train)
	if dataset == 'MNIST':
		transform_train=transforms.Compose([
		transforms.RandomCrop(28, padding=4),
		transforms.Grayscale(3),
        transforms.ToTensor(),
        transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081))])
		trainset = torchvision.datasets.MNIST(root=dataset_path, train=True, download=True,
                             transform=transform_train)
	if dataset == 'FMNIST':
		transform_train=transforms.Compose([
		transforms.RandomCrop(28, padding=4),
		transforms.RandomHorizontalFlip(),
		transforms.Grayscale(3),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
		trainset = torchvision.datasets.FashionMNIST(root=dataset_path, train=True, download=True,
                             transform=transform_train)
	if dataset == 'CIFAR100':
		transform_train=transforms.Compose([
		transforms.RandomCrop(32, padding=4),
		transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276))])
		trainset = torchvision.datasets.CIFAR100(root=dataset_path, train=True, download=True,
                             transform=transform_train)

	subset = Subset(trainset, part_tr)

	## If config.fedgkt is True, shuffle = False
	if train_mode == 'FedGKT':
		trainloader = DataLoader(
		subset, batch_size=B, shuffle=False, num_workers=cpu_count)
	else:
		trainloader = DataLoader(
		subset, batch_size=B, shuffle=True, num_workers=cpu_count)
	return trainloader

def get_model(location, model_name, op, device, cfg):
	cfg = cfg.copy()
	net = CNN(model_name, cfg, op, location)
	net = net.to(device)
	return net

def get_compressor(device, loc):
	aenet = autoencoder(loc)
	aenet = aenet.to(device)
	return aenet

def extract_ae_weights(pweights,ae_weights,loc):
	for key in ae_weights:
		if loc == 'Device':
			if 'encoder' in key:
				assert ae_weights[key].size() == pweights[key].size()
				pweights[key] = ae_weights[key]
		if loc == 'Server':
			if 'decoder' in key:
				assert ae_weights[key].size() == pweights[key].size()
				pweights[key] = ae_weights[key]
	return pweights

def split_weights_client(weights,cweights):
	for key in cweights:
		assert cweights[key].size() == weights[key].size()
		cweights[key] = weights[key]
	return cweights

def transfer_weights_client(weights_file,cweights):
    pretrained_weights = torch.load(weights_file)
    for key in cweights:
        if 'num_batches_tracked' not in key:
            assert cweights[key].size() == pretrained_weights[key].size()
            cweights[key] = pretrained_weights[key]
    return cweights

def transfer_weights_holistic(weights_file,weights):
    pretrained_weights = torch.load(weights_file)
    for key in weights:
        if 'num_batches_tracked' not in key and 'classifier' not in key:
            assert weights[key].size() == pretrained_weights[key].size()
            weights[key] = pretrained_weights[key]
    return weights

def transfer_weights_partial(weights_file,partial_weights,weights):
    pretrained_weights = torch.load(weights_file)
    for key in partial_weights:
        if 'num_batches_tracked' not in key and 'classifier' not in key:
            assert weights[key].size() == pretrained_weights[key].size()
            weights[key] = pretrained_weights[key]
    return weights

def split_weights_server(weights,cweights,sweights):
	ckeys = list(cweights)
	skeys = list(sweights)
	keys = list(weights)

	for i in range(len(skeys)):
		assert sweights[skeys[i]].size() == weights[keys[i + len(ckeys)]].size()
		sweights[skeys[i]] = weights[keys[i + len(ckeys)]]

	return sweights

def concat_weights(weights,cweights,sweights):
	concat_dict = collections.OrderedDict()

	ckeys = list(cweights)
	skeys = list(sweights)
	keys = list(weights)

	for i in range(len(ckeys)):
		concat_dict[keys[i]] = cweights[ckeys[i]]

	for i in range(len(skeys)):
		concat_dict[keys[i + len(ckeys)]] = sweights[skeys[i]]

	return concat_dict

def zero_init(net):
	for m in net.modules():
		if isinstance(m, nn.Conv2d):
			init.zeros_(m.weight)
			if m.bias is not None:
				init.zeros_(m.bias)
		if isinstance(m, nn.ConvTranspose2d):
			init.zeros_(m.weight)
			if m.bias is not None:
				init.zeros_(m.bias)
		elif isinstance(m, nn.BatchNorm2d):
			init.zeros_(m.weight)
			init.zeros_(m.bias)
			#init.zeros_(m.running_mean)
			#init.zeros_(m.running_var)
		elif isinstance(m, nn.Linear):
			init.zeros_(m.weight)
			if m.bias is not None:
				init.zeros_(m.bias)
	return net

def fed_avg(zero_model, w_local_list):
	keys = w_local_list[0][0].keys()
	for k in keys:
		for w in w_local_list:
			beta = w[1]
			if 'num_batches_tracked' in k:
				zero_model[k] = w[0][k]
			else:	
				zero_model[k] += (w[0][k] * beta)
	return zero_model

def fed_avg_simulation(aggregrated_model, w_local_list):
	keys = w_local_list[0][0].keys()
	for k in keys:
		for w in w_local_list:
			beta = w[1]
			if 'num_batches_tracked' in k:
				aggregrated_model[k] = w[0][k]
			else:	
				aggregrated_model[k] += (w[0][k] * beta)
	return aggregrated_model

def minmax_quant(tensor):
	scale, zero_point = tensor.max().item() / 127, 0
	dtype = torch.qint8
	q_tensor = torch.quantize_per_tensor(tensor, scale, zero_point, dtype)
	return q_tensor

def dequant(q_tensor):
	tensor = torch.dequantize(q_tensor)
	tensor.requires_grad = True
	return tensor

def dict_mean(dict):
    sum = 0
    for k in dict:
        sum += dict[k]
    return sum / len(dict)

def move_state_dict_to_device(state_dict, device):
    # Create an empty state dictionary on the GPU
    state_dict_on_device = {}
    
    # Move each weight to the GPU and add it to the new state dictionary
    for key in state_dict:
        state_dict_on_device[key] = state_dict[key].to(device)
    
    return state_dict_on_device


class KL_Loss(nn.Module):
    def __init__(self, temperature=1):
        super(KL_Loss, self).__init__()
        self.T = temperature

    def forward(self, output_batch, teacher_outputs):
        # output_batch  -> B X num_classes
        # teacher_outputs -> B X num_classes

        # loss_2 = -torch.sum(torch.sum(torch.mul(F.log_softmax(teacher_outputs,dim=1), F.softmax(teacher_outputs,dim=1)+10**(-7))))/teacher_outputs.size(0)
        # print('loss H:',loss_2)

        output_batch = F.log_softmax(output_batch / self.T, dim=1)
        teacher_outputs = F.softmax(teacher_outputs / self.T, dim=1) + 10 ** (-7)

        loss = self.T * self.T * nn.KLDivLoss(reduction='batchmean')(output_batch, teacher_outputs)

        # Same result KL-loss implementation
        # loss = T * T * torch.sum(torch.sum(torch.mul(teacher_outputs, torch.log(teacher_outputs) - output_batch)))/teacher_outputs.size(0)
        return loss

class CE_Loss(nn.Module):
    def __init__(self, temperature=1):
        super(CE_Loss, self).__init__()
        self.T = temperature

    def forward(self, output_batch, teacher_outputs):
        # output_batch      -> B X num_classes
        # teacher_outputs   -> B X num_classes

        output_batch = F.log_softmax(output_batch / self.T, dim=1)
        teacher_outputs = F.softmax(teacher_outputs / self.T, dim=1)

        # Same result CE-loss implementation torch.sum -> sum of all element
        loss = -self.T * self.T * torch.sum(torch.mul(output_batch, teacher_outputs)) / teacher_outputs.size(0)

        return loss