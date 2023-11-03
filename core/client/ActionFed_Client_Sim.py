#ActionFed Client class
import torch
import torch.nn as nn
import torch.optim as optim
import time
import sys
from tqdm import tqdm
import torchvision
import torchvision.transforms as transforms
import collections
import copy
import numpy as np

sys.path.append('../')
import config
import utils
from Communicator import *
from client.SFL_Client_Sim import *


import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ActionFed_Client_Sim(SFL_Client_Sim):
	def __init__(self, server_addr, server_port, model_name, ip, index, device):
		super(ActionFed_Client_Sim, self).__init__(server_addr, server_port, model_name, ip, index, device)

		#self.auxiliary_net = utils.get_model('Device', 'Auxiliary_Net', -1, self.device, config.model_cfg)
		#logger.debug(self.auxiliary_net)

	def initialize(self, OP, LR, R, current_c):
		self.op = OP
		self.lr = LR
		logger.info('Building Model.')
		self.net = utils.get_model('Device', self.model_name, self.op, self.device, config.model_cfg)
		logger.info(self.net)

		self.criterion = nn.CrossEntropyLoss()
		self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr,
					  momentum=0.9)
		
		# ActionFed initialization
		logger.info('Loading Pretrained Weights..')
		cweights = utils.get_model('Device', self.model_name, self.op, self.device, config.model_cfg).state_dict()
		weights = utils.transfer_weights_client(config.pre_trained_weights_path,cweights)
		self.net.load_state_dict(weights)
		logger.info('Initialize Finished')

	def train(self, trainloader, R, current_c, client_id):
		# Training start
		time_acv_comm = 0
		time_c_comp = 0
		training_loss = 0
		self.net.to(self.device)
	
		# Wait buffer, compressor and quantization signal from server
		msg = self.recv_msg(self.sock)
		self.update_buffer = msg[1]

		if self.update_buffer:
			msg = ['MSG_LOCAL_CLIENT_ID_TO_SERVER', client_id] #quantizaiton
			self.send_msg(self.sock, msg)
   
		self.net.eval()
		#self.net.train()

		## Need an auxiliary net
		#self.auxiliary_net.to(self.device)
		#self.auxiliary_net.train()

		if self.op == -1: # No offloading training
			for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader)):
				inputs, targets = inputs.to(self.device), targets.to(self.device)
				self.optimizer.zero_grad()
				outputs = self.net(inputs)
				loss = self.criterion(outputs, targets)
				training_loss += loss.item()
				loss.backward()
				self.optimizer.step()
		else: # Offloading training
			for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader)):
				#self.optimizer.zero_grad()
				
				## Zero gradients for auxiliary optimizer
				#self.auxiliary_optimizer.zero_grad()
				if self.update_buffer:
					with torch.no_grad():
						tic_c_comp = time.time()
						## If server do NOT use buffer replay, then clients inference through the dataset
						inputs, targets = inputs.to(self.device), targets.to(self.device)
						outputs = self.net(inputs)
						time_c_comp += (time.time() - tic_c_comp)
						msg = ['MSG_LOCAL_ACTIVATIONS_CLIENT_TO_SERVER', utils.minmax_quant(outputs), targets] #quantizaiton

						'''
						# Estimation of communication time. There is a trick bug for communication time.
						# when the server are busy, there is a noticable time for waiting server gets the data from network interface
						# it causes the communication much larger than real transmission time.
						# Therefore, for SFL, LGL, FedGKT and ActionFed, we estimate the communication latency
						if batch_idx == 0:
							for i in range(3):
								msg = ['MSG_LOCAL_ACTIVATIONS_CLIENT_TO_SERVER', utils.minmax_quant(outputs), targets] #quantizaiton
								tic_acv_comm = time.time()
								self.send_msg(self.sock, msg)
								self.recv_msg(self.sock) # MSG_TIME_RECORD
								time_acv_comm += (time.time() - tic_acv_comm)
							time_acv_comm = (time_acv_comm / 3) * len(trainloader)
							logger.info('Estimated activation time: {:} and each iteration time {:}'.format(time_acv_comm, time_acv_comm/len(trainloader)))
						'''
						
						self.send_msg(self.sock, msg)
						#local_outputs = self.auxiliary_net(outputs)
						#loss = self.auxiliary_criterion(local_outputs, targets)
						#training_loss += loss.item()
						#loss.backward()

						#self.auxiliary_optimizer.step()
						#self.optimizer.step()

						# release the threads competition on the server
						msg = self.recv_msg(self.sock)
				else:
					# release the threads competition on the server
					msg = self.recv_msg(self.sock)
					self.send_msg(self.sock, msg)
					
					msg = self.recv_msg(self.sock)

		msg = ['MSG_ROUND_FINISH']
		self.send_msg(self.sock, msg)

		self.recv_msg(self.sock) # MSG_GLOBAL_ROUND_FINISH
		return time_acv_comm, time_c_comp
	
	def upload(self):
		time_aggre_comm = 0
		return time_aggre_comm



