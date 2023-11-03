# Random_Sampler Object
import sys
import numpy as np
import random
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from Data_Generator import Data_Generator
sys.path.append('../')
import config

import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MyDataset(Dataset):
	def __init__(self, data, targets, transform=None):
		self.data = data
		self.targets = targets
		self.transform = transform

	def __getitem__(self, index):
		img, target = self.data[index], self.targets[index]

		# convert the numpy array to a PIL Image
		img = Image.fromarray(img)
  
		if self.transform:
			img = self.transform(img)
		return img, target

	def __len__(self):
		return len(self.data)

class Non_IID_Generator(Data_Generator):
	def __init__(self, cpu_count, dataset, num_clients, num_shards, shard_indices):
		self.cpu_count = cpu_count
		self.dataset = dataset
		self.shard_indices = shard_indices

		random.seed(config.SEED)
  
		if dataset == 'CIFAR10':
			self.transform_train = transforms.Compose([
			transforms.RandomCrop(32, padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
			trainset = datasets.CIFAR10(
				root=config.dataset_path, train=True, download=True)
		if dataset == 'MNIST':
			self.transform_train=transforms.Compose([
			transforms.RandomCrop(28, padding=4),
			transforms.Grayscale(3),
			transforms.ToTensor(),
			transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081))])
			trainset = datasets.MNIST(root=config.dataset_path, train=True, download=True)
		if dataset == 'FMNIST':
			self.transform_train=transforms.Compose([
			transforms.RandomCrop(28, padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.Grayscale(3),
			transforms.ToTensor(),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
			trainset = datasets.FashionMNIST(root=config.dataset_path, train=True, download=True)
		if dataset == 'CIFAR100':
			self.transform_train=transforms.Compose([
			transforms.RandomCrop(32, padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276))])
			trainset = datasets.CIFAR100(root=config.dataset_path, train=True, download=True)

		# For Non-iid case, we sort dataset by label first
		trainset.data, trainset.targets = zip(*sorted(zip(trainset.data, trainset.targets), key=lambda x: x[1]))
		
		# Divide the dataset into shards
		shard_size = int(len(trainset) / num_shards)
		shards = []
		for i in range(num_shards):
			data_indices = range(i*shard_size, (i+1)*shard_size)
			shards.append(data_indices)

		# Divide shards into clients
		if len(self.shard_indices) != num_shards:
			raise 'num_shards is not equal to the length of shard_indices!'

		if num_shards%num_clients != 0:
			raise 'num_shards is not a valid value!'
		client_size = int(num_shards / num_clients)
		client_data_size = shard_size * client_size
		self.clients_data = []
		for i in range(num_clients):
			shuffle_indices = random.sample(range(client_data_size), k = client_data_size) # FedGKT needs to shuffle before dataloader shuffle, without replacement
			if len(set(shuffle_indices)) != len(shuffle_indices):
				raise 'Shuffle with replacement!'
			concat_data_indices = []
			for j in range(client_size):
				index = i * client_size + j
				concat_data_indices.extend(shards[self.shard_indices[index]])
				
			shards_data = [trainset.data[k] for k in concat_data_indices]
			shards_targets = [trainset.targets[k] for k in concat_data_indices]

			shuffled_shards_data = [shards_data[index] for index in shuffle_indices]
			shuffled_shards_targets = [shards_targets[index] for index in shuffle_indices]
			self.clients_data.append(MyDataset(shuffled_shards_data, shuffled_shards_targets, self.transform_train))
   			
   
	def get_local_dataloader(self, client_id):
		## If config.fedgkt is True, shuffle = False
		if config.train_mode == 'FedGKT':
			trainloader = DataLoader(
			self.clients_data[client_id], batch_size=config.B, shuffle=False, num_workers=self.cpu_count)
		else:
			trainloader = DataLoader(
			self.clients_data[client_id], batch_size=config.B, shuffle=True, num_workers=self.cpu_count)
		return trainloader

def label_count(dataloader):
	label_counts = {}
	for data, label in dataloader:
		for l in label:
			if l.item() in label_counts:
				label_counts[l.item()] += 1
			else:
				label_counts[l.item()] = 1
	
	# Calculate total number of examples
	total_examples = sum(label_counts.values())
	print(total_examples)

	# Calculate percentile of each label
	for label, count in label_counts.items():
		percentile =  (count / total_examples) * 100
		print(f"Label {label}: {count} examples ({percentile:.2f}% percentile)")
	return label_counts

			
def unit_test():
	cpu_count, dataset, num_clients, num_shards = 1, 'CIFAR10', 100, 500
	random.seed(0)
	shard_indices = random.sample(range(num_shards), num_shards)
 
	iid_generator = Non_IID_Generator(cpu_count, dataset, num_clients, num_shards, shard_indices)
	client_ids = random.sample(range(num_clients), 10)
	for client_id in client_ids:
		print(client_id)
		client_dataloader = iid_generator.get_local_dataloader(client_id)
		label_counts = label_count(client_dataloader)
		print("-" * 50)
  
	# Test for FedGKT
	client_id = 10
	client_dataloader = iid_generator.get_local_dataloader(client_id)
 
	for batch_idx, (inputs, targets) in enumerate(client_dataloader):
		print(targets)
	print("-" * 50)
	for batch_idx, (inputs, targets) in enumerate(client_dataloader):
		print(targets)
if __name__ == '__main__':
	unit_test()
