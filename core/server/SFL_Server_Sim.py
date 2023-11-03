# Server class
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import threading
from tqdm import tqdm
import time
import numpy as np

import sys
sys.path.append('../')
from Communicator import *
import utils
import config
import pickle

import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SFL_Server_Sim(Communicator):
	def __init__(self, ip_address, server_port, model_name, device, client_sampler, shard_indices):
		super(SFL_Server_Sim, self).__init__()
		self.ip = ip_address
		self.port = server_port
		self.device = device
		self.port = server_port
		self.model_name = model_name
		self.sock.bind((self.ip, self.port))

		self.client_ips = set()
		self.client_socks = {}
		self.testacc = [] # Record for test acc
		self.training_loss = {} # Record for training loss
		self.mode_rec = [] # Record for training mode
		self.aggregrated_model_simulation = None # Aggregrated_model for simulation
		self.simulation_agg_control = 0 # number of aggregration times
		for client_ip in self.client_ips:
			self.training_loss[client_ip] = [] # Record for training loss for each device

		# Global model
		self.uninet = utils.get_model('Cloud', self.model_name, 0, self.device, config.model_cfg)
		# zero model for simulation
		self.zeronet = utils.get_model('Cloud', self.model_name, 0, self.device, config.model_cfg)

		# Test sets
		if config.dataset_name == 'CIFAR10':
			self.transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
			self.testset = torchvision.datasets.CIFAR10(root=config.dataset_path, train=False, download=True, transform=self.transform_test)
		if config.dataset_name == 'MNIST':
			self.transform_test = transforms.Compose([transforms.Grayscale(3), transforms.ToTensor(), transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081))])
			self.testset = torchvision.datasets.MNIST(root=config.dataset_path, train=False, download=True, transform=self.transform_test)
		if config.dataset_name == 'FMNIST':
			self.transform_test = transforms.Compose([transforms.Grayscale(3), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
			self.testset = torchvision.datasets.FashionMNIST(root=config.dataset_path, train=False, download=True, transform=self.transform_test)
		if config.dataset_name == 'CIFAR100':
			self.transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276))])
			self.testset = torchvision.datasets.CIFAR100(root=config.dataset_path, train=False, download=True, transform=self.transform_test)
		self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=100, shuffle=False, num_workers=4)
		self.criterion = nn.CrossEntropyLoss() #Used for test in simulation
		
		# Waiting connection from K devices
		while len(self.client_socks) < config.K:
			self.sock.listen(5)
			logger.info("Waiting Incoming Connections.")
			(client_sock, (ip, port)) = self.sock.accept()
			logger.info('Got connection from ' + str(ip)+':'+str(port))
			logger.info(client_sock)
			self.client_ips.add(str(ip)+':'+str(port))
			self.client_socks[str(ip)+':'+str(port)] = client_sock
		
		# Samplers and shard_indices
		self.client_sampler = client_sampler
		self.shard_indices = shard_indices
		for client_ip in self.client_ips:
			msg = ['MSG_CLIENT_SAMPLER_SERVER_TO_CLIENT', self.client_sampler, self.shard_indices]
			self.send_msg(self.client_socks[client_ip], msg)
		
  		# Locking for multiple GPUs
		self.lock = threading.Lock()

	def initialize(self, OP, LR, R, current_c):
		self.op = OP #OP is a array of all clients
		self.nets = {}
		self.optimizers = {}
		self.criterions = {}
		self.time_ini = {}
			
		for client_ip in self.client_ips:
			## Current version only support all devices have the same OP
			if len([i for i in self.op if i == -1]) == len(self.op): #Only if all clients is device-native training
				## Weight initilization for each round
				if R == 0: # First round initilization
					if config.initilization == 'random': 
						init_cweights = self.uninet.state_dict()
					if config.initilization == 'partial_pretrain':
						cweights = utils.get_model('Device', self.model_name, 2, self.device, config.model_cfg).state_dict()
						#partial_pretrain_cweights = utils.transfer_weights_partial( config.home + 'EcoFed_Project/pretrained/vgg11_bn-6002323d.pth',cweights,self.uninet.state_dict())
						partial_pretrain_cweights = utils.transfer_weights_partial( config.home + 'EcoFed_Project/pretrained/vgg11_tiny_imagenet_32_cpu_unzip.pth',cweights,self.uninet.state_dict())
						#partial_pretrain_cweights = utils.transfer_weights_partial( config.home + 'EcoFed_Project/pretrained/resnet9_tiny_imagenet_32_cpu.pth',cweights,self.uninet.state_dict())
						self.uninet.load_state_dict(partial_pretrain_cweights)
						init_cweights = self.uninet.state_dict()
					if config.initilization == 'holistic_pretrain':
						#holistic_pretrain_weights = utils.transfer_weights_holistic(config.home + 'EcoFed_Project/pretrained/vgg11_bn-6002323d.pth',self.uninet.state_dict())
						holistic_pretrain_weights = utils.transfer_weights_holistic(config.home + 'EcoFed_Project/pretrained/vgg11_tiny_imagenet_32_cpu_unzip.pth',self.uninet.state_dict())
						#holistic_pretrain_weights = utils.transfer_weights_holistic(config.home + 'EcoFed_Project/pretrained/resnet9_tiny_imagenet_32_cpu.pth',self.uninet.state_dict())
						self.uninet.load_state_dict(holistic_pretrain_weights)
						init_cweights = self.uninet.state_dict()
				else: # Other rounds
						init_cweights = self.uninet.state_dict()
			else:
				## Offloading networks
				self.nets[client_ip] = utils.get_model('Cloud', self.model_name, self.op[config.IP2INDEX[client_ip]], self.device, config.model_cfg)
				logger.debug(self.nets[client_ip])
				 
				## cweight is init weights for client's model 
				cweights = utils.get_model('Device', self.model_name, self.op[config.IP2INDEX[client_ip]], self.device, config.model_cfg).state_dict()
				self.optimizers[client_ip] = optim.SGD(self.nets[client_ip].parameters(), lr=LR,
					momentum=0.9)
				self.criterions[client_ip] = nn.CrossEntropyLoss()
				
				if R == 0: # First round initilization
					if config.finetuning:
						self.uninet.load_state_dict(torch.load(config.home+'ActionFed/trained_models/'+config.dataset_name+'_'+config.model_name+'_'+config.train_mode+'.pth'))
					## Weight initilization
					if config.initilization == 'random':
						init_cweights = utils.split_weights_client(self.uninet.state_dict(),cweights)
					if config.initilization == 'partial_pretrain':
						#init_cweights = utils.transfer_weights_client(config.home+'EcoFed_Project/pretrained/vgg11_bn-6002323d.pth',cweights)
						init_cweights = utils.transfer_weights_client(config.home+'EcoFed_Project/pretrained/vgg11_tiny_imagenet_32_cpu_unzip.pth',cweights)
						#init_cweights = utils.transfer_weights_client(config.home+'EcoFed_Project/pretrained/resnet9_tiny_imagenet_32_cpu.pth',cweights)
					if config.initilization == 'holistic_pretrain':
						#holistic_pretrain_weights = utils.transfer_weights_holistic(config.home+'EcoFed_Project/pretrained/vgg11_bn-6002323d.pth',self.uninet.state_dict())
						holistic_pretrain_weights = utils.transfer_weights_holistic(config.home+'EcoFed_Project/pretrained/vgg11_tiny_imagenet_32_cpu_unzip.pth',self.uninet.state_dict())
						#holistic_pretrain_weights = utils.transfer_weights_holistic(config.home+'EcoFed_Project/pretrained/resnet9_tiny_imagenet_32_cpu.pth',self.uninet.state_dict())
						self.uninet.load_state_dict(holistic_pretrain_weights)
						init_cweights = utils.split_weights_client(self.uninet.state_dict(),cweights)
					## pweight is init weights for server's model
					pweights = utils.split_weights_server(self.uninet.state_dict(),cweights,self.nets[client_ip].state_dict())
					self.nets[client_ip].load_state_dict(pweights)
				else: # Other rounds
					init_cweights = utils.split_weights_client(self.uninet.state_dict(),cweights)
					pweights = utils.split_weights_server(self.uninet.state_dict(),cweights,self.nets[client_ip].state_dict())
					self.nets[client_ip].load_state_dict(pweights)
					self.init_cweights = init_cweights
		self.criterion = nn.CrossEntropyLoss() #Used for test
		
		# Weight distribution with multiple threads
		self.threads = {}
		for client_ip in self.client_ips:
			self.threads[client_ip] = threading.Thread(target=self._thread_weights_distribution_, args=(client_ip, init_cweights,))
			logger.debug(str(client_ip) + ' weights distribution start.')
			self.threads[client_ip].start()

		for client_ip in self.client_ips:
			self.threads[client_ip].join()

		for client_ip in self.client_ips:
			logger.debug(str(client_ip) + ' weights distribution finish.')
	

	def train(self, R, current_c):
		# Training start
		
		## Display and save the mode details
		logger.info('Initilization: ' + str(config.initilization))
		logger.info('Training mode: ' + str(config.train_mode))
		#self.mode_rec.append(config.train_mode)
			
		self.time_grad = {}
		self.time_comp = {}
		## Device native training
		if len([i for i in self.op if i == -1]) == len(self.op): # Only if all clients is device-native training
			self.training_no_offloading()
			for client_ip in self.client_ips:
				logger.debug(client_ip + ' no offloading training start')
				pass

			for client_ip in self.client_ips:
				self.recv_msg(self.client_socks[client_ip], 'MSG_ROUND_FINISH')
				logger.debug('MSG_ROUND_FINISH')
		else:
			if config.train_mode == 'FedGKT':
				for client_ip in self.client_ips: #Wait all clients finishing their training
					msg = self.recv_msg(self.client_socks[client_ip], 'MSG_ROUND_FINISH')
					logger.debug('MSG_ROUND_FINISH')
					
			self.threads = {}
			for client_ip in self.client_ips:
				self.threads[client_ip] = threading.Thread(target=self._thread_training_offloading, args=(client_ip, R, current_c,))
				logger.debug(str(client_ip) + ' offloading training start')
				self.threads[client_ip].start()

			for client_ip in self.client_ips:
				self.threads[client_ip].join()

			for client_ip in self.client_ips:
				msg = self.recv_msg(self.client_socks[client_ip], 'MSG_ROUND_FINISH')
				logger.debug('MSG_ROUND_FINISH')
		
		for client_ip in self.client_ips:
			self.send_msg(self.client_socks[client_ip], ['MSG_GLOBAL_ROUND_FINISH']) #MSG_GLOBAL_ROUND_FINISH

	def training_no_offloading(self):
		for client_ip in self.client_ips:
			self.time_grad[client_ip] = 0
			self.time_comp[client_ip] = 0

	def _thread_weights_distribution_(self, client_ip, init_cweights):
		tic_ini = time.time()
		msg = ['MSG_INITIAL_GLOBAL_WEIGHTS_SERVER_TO_CLIENT', init_cweights]
		self.send_msg(self.client_socks[client_ip], msg)
		self.recv_msg(self.client_socks[client_ip])
		self.time_ini[client_ip] = time.time() - tic_ini

	def _thread_training_offloading(self, client_ip, R, current_c):
		self.time_grad[client_ip] = 0
		self.time_comp[client_ip] = 0

		num_clients = (config.K * config.C) / config.ALPHA
		iteration = int((config.N / (num_clients * config.B))) # Simulation

		for i in tqdm(range(iteration)):
			if i == 0:
				for j in range(3):
					msg = self.recv_msg(self.client_socks[client_ip], 'MSG_LOCAL_ACTIVATIONS_CLIENT_TO_SERVER')
					self.send_msg(self.client_socks[client_ip], ['MSG_TIME_RECORD']) #MSG_TIME_RECORD

			msg = self.recv_msg(self.client_socks[client_ip], 'MSG_LOCAL_ACTIVATIONS_CLIENT_TO_SERVER')

			tic_comp = time.time()
			smashed_layers = msg[1]
			labels = msg[2]
			inputs, targets = smashed_layers.to(self.device), labels.to(self.device)
			
			self.optimizers[client_ip].zero_grad()
			with self.lock:
				outputs = self.nets[client_ip](inputs)		
			loss = self.criterions[client_ip](outputs, targets)
			loss.backward()
			self.optimizers[client_ip].step()
			self.time_comp[client_ip] += (time.time() - tic_comp)

			inputs.retain_grad()
			msg = ['MSG_SERVER_GRADIENTS_SERVER_TO_CLIENT_'+str(client_ip), inputs.grad]
			# Send gradients to client
			if i == 0:
				for j in range(3):
					tic_grad = time.time()
					self.send_msg(self.client_socks[client_ip], msg)
					self.recv_msg(self.client_socks[client_ip], 'MSG_TIME_RECORD')
					self.time_grad[client_ip] += (time.time() - tic_grad)
				self.time_grad[client_ip] = (self.time_grad[client_ip] / 3) * iteration
				logger.info('Estimated gradient time: {:} and each iteration time {:}'.format(self.time_grad[client_ip], self.time_grad[client_ip]/iteration))
			
			self.send_msg(self.client_socks[client_ip], msg)

	def _thread_weights_collection_(self, client_ip):
		msg = self.recv_msg(self.client_socks[client_ip], 'MSG_LOCAL_WEIGHTS_CLIENT_TO_SERVER')
		self.send_msg(self.client_socks[client_ip], ['MSG_TIME_RECORD']) #MSG_TIME_RECORD
		self.msgs.append(msg)

	def aggregate(self):
		w_local_list =[]
		self.msgs = []
		self.threads = {}
		for client_ip in self.client_ips:
			self.threads[client_ip] = threading.Thread(target=self._thread_weights_collection_, args=(client_ip,))
			logger.debug(str(client_ip) + ' weights collection start.')
			self.threads[client_ip].start()

		for client_ip in self.client_ips:
			self.threads[client_ip].join()

		for client_ip in self.client_ips:
				logger.debug(str(client_ip) + ' weights collection finish.')
		
		for msg in self.msgs:
			ip = msg[2]
			if self.op[config.IP2INDEX[ip]] == -1:
				w_local = (msg[1],1 / (config.K * config.C))
				w_local_list.append(w_local)
			else:
				w_local = (utils.concat_weights(self.uninet.state_dict(),msg[1],self.nets[ip].state_dict()),1 / (config.K * config.C))
				w_local_list.append(w_local)

		zero_model = utils.zero_init(self.uninet).state_dict()
		aggregrated_model = utils.fed_avg(zero_model, w_local_list)
		self.uninet.load_state_dict(aggregrated_model)
	
	def aggregate_simulation(self):
		w_local_list =[]
		self.msgs = []
		self.threads = {}
		for client_ip in self.client_ips:
			self.threads[client_ip] = threading.Thread(target=self._thread_weights_collection_, args=(client_ip,))
			logger.debug(str(client_ip) + ' weights collection start.')
			self.threads[client_ip].start()

		for client_ip in self.client_ips:
			self.threads[client_ip].join()

		for client_ip in self.client_ips:
				logger.debug(str(client_ip) + ' weights collection finish.')

		for msg in self.msgs:
			ip = msg[2]
			msg[1] = utils.move_state_dict_to_device(msg[1], self.device)
			if self.op[config.IP2INDEX[ip]] == -1:
				w_local = (msg[1],1 / (config.K * config.C))
				w_local_list.append(w_local)
			else:
				w_local = (utils.concat_weights(self.uninet.state_dict(),msg[1],self.nets[ip].state_dict()),1 / (config.K * config.C))
				w_local_list.append(w_local)

		if self.aggregrated_model_simulation is None: 
			self.aggregrated_model_simulation = utils.zero_init(self.zeronet).state_dict()
		self.aggregrated_model_simulation = utils.fed_avg_simulation(self.aggregrated_model_simulation, w_local_list)
		self.simulation_agg_control += config.K
		logger.info('simulation_agg_control: {:}'.format(self.simulation_agg_control))
		if self.simulation_agg_control > (config.K * config.C):
			assert 'Aggregration error!'
		if self.simulation_agg_control == (config.K * config.C):
			# Finish training of one round
			self.uninet.load_state_dict(self.aggregrated_model_simulation)
			self.aggregrated_model_simulation = None
			self.simulation_agg_control = 0
		

	def test(self, r):
		self.uninet.eval()
		test_loss = 0
		correct = 0
		total = 0
		with torch.no_grad():
			for batch_idx, (inputs, targets) in enumerate(tqdm(self.testloader)):
				inputs, targets = inputs.to(self.device), targets.to(self.device)
				outputs = self.uninet(inputs)
				loss = self.criterion(outputs, targets)

				test_loss += loss.item()
				_, predicted = outputs.max(1)
				total += targets.size(0)
				correct += predicted.eq(targets).sum().item()

		acc = 100.*correct/total
		logger.info('Test Accuracy: {}'.format(acc))
		self.testacc.append(acc)

		## Save checkpoint.
		#torch.save(self.uninet.state_dict(), config.home+'ActionFed/trained_models/'+config.dataset_name+'_'+config.model_name+'_'+config.train_mode+'.pth')

		return acc

	def time_profile(self, time_total_s):
		rec = {}
		avg_comm_init = {}
		avg_comm_act = {}
		avg_comm_grad = {}
		avg_comm_dist = {}
		avg_c_comp = {}
		avg_s_comp = {}
		avg_total = {}
		for client_ip in self.client_ips:
			msg = self.recv_msg(self.client_socks[client_ip], 'MSG_TIME_PROFILE')
			ip, time_acv_comm, time_aggre_comm, time_total_c, time_c_comp = msg[1], msg[2], msg[3], msg[4], msg[5] 
			logger.debug('IP: {:}, Total_Client: {:}, Total_Server: {:}'.format(client_ip, time_total_c, time_total_s))
			total_time = max(time_total_c, time_total_s)
			comm_time = self.time_ini[client_ip] + time_acv_comm + self.time_grad[client_ip] + time_aggre_comm
			comp_time = time_c_comp + self.time_comp[client_ip]
			logger.debug('Init_Comm.: {:}, Act_Comm.: {:}, Grad_Comm.: {:}, Aggr_Comm.: {:}'.format(self.time_ini[client_ip], time_acv_comm, self.time_grad[client_ip], time_aggre_comm))
			
			logger.debug('Communication: {:}, Computation: {:}, Total: {:}'.format(comm_time, comp_time, total_time))
			avg_comm_init[client_ip] = self.time_ini[client_ip]
			avg_comm_act[client_ip] = time_acv_comm
			avg_comm_grad[client_ip] = self.time_grad[client_ip]
			avg_comm_dist[client_ip] = time_aggre_comm
			avg_c_comp[client_ip] = time_c_comp
			avg_s_comp[client_ip] = self.time_comp[client_ip]
			avg_total[client_ip] = total_time

			## Save the time record
			rec[client_ip] = [client_ip, time_acv_comm, time_aggre_comm, time_total_c, comm_time, comp_time, total_time]
		avg_comm_init_arr = np.array(list(avg_comm_init.values()))
		avg_comm_act_arr = np.array(list(avg_comm_act.values()))
		avg_comm_grad_arr = np.array(list(avg_comm_grad.values()))
		avg_comm_dist_arr = np.array(list(avg_comm_dist.values()))
		avg_c_comp_arr = np.array(list(avg_c_comp.values()))
		avg_s_comp_arr = np.array(list(avg_s_comp.values()))
		avg_total_arr = np.array(list(avg_total.values()))

		mean_comm_init = np.mean(avg_comm_init_arr)
		mean_comm_act = np.mean(avg_comm_act_arr)
		mean_comm_grad = np.mean(avg_comm_grad_arr)
		mean_comm_dist = np.mean(avg_comm_dist_arr)
		mean_c_comp = np.mean(avg_c_comp_arr)
		mean_s_comp = np.mean(avg_s_comp_arr)

		mean_comm = mean_comm_init + mean_comm_act + mean_comm_grad + mean_comm_dist
		mean_comp = mean_c_comp + mean_s_comp
		mean_total = mean_comp + mean_comm
		logger.info('Avg Communication Init: {:}, Avg Communication Act: {:}, Avg Communication Grad: {:}, Avg Communication Dist: {:}, Avg Communication: {:}, Avg Client Computation: {:}, Avg Server Computation: {:}, Avg Computation: {:}, Avg Total: {:}'
			  .format(mean_comm_init, mean_comm_act, mean_comm_grad, mean_comm_dist, mean_comm, mean_c_comp, mean_s_comp, mean_comp, mean_total))
		return rec
