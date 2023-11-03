# SFL Client class
import torch
import torch.nn as nn
import torch.optim as optim
import time
import sys
from tqdm import tqdm
import torchvision
import torchvision.transforms as transforms


sys.path.append('../')
import config
import utils
from Communicator import *


import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SFL_Client_Sim(Communicator):
	def __init__(self, server_addr, server_port, model_name, ip, index, device):
		super(SFL_Client_Sim, self).__init__()
		self.ip = ip
		self.port = config.SERVER_PORT + 1 + index
		self.device = device
		self.model_name = model_name
		self.index = index

		logger.info('Connecting to Server.')
		self.sock.bind((self.ip, self.port))
		self.sock.connect((server_addr,server_port))
		# Replace the ip with ip:port
		self.ip = str(self.ip)+':'+str(self.port)
   
		logger.info('Receiving Client Sampler and Shard Indices..')
		msg = self.recv_msg(self.sock)
		self.client_sampler = msg[1]
		self.shard_indices = msg[2]

	def initialize(self, OP, LR, R, current_c):
		self.op = OP
		self.lr = LR
		logger.info('Building Model.')
		self.net = utils.get_model('Device', self.model_name, self.op, self.device, config.model_cfg)
		logger.info(self.net)

		self.criterion = nn.CrossEntropyLoss()
		self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr,
					  momentum=0.9)
			
		logger.info('Receiving Global Weights..')
		msg = self.recv_msg(self.sock)
		self.send_msg(self.sock, ['MSG_TIME_RECORD'])
		weights = msg[1]
		self.net.load_state_dict(weights)
		logger.info('Initialize Finished')

	def train(self, trainloader, R, current_c, client_id):
		# Training start
		time_acv_comm = 0
		time_c_comp = 0
		training_loss = 0
		self.net.to(self.device)

		self.net.train()
		if self.op == -1: # No offloading training
			tic_c_comp = time.time()
			for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader)):
				inputs, targets = inputs.to(self.device), targets.to(self.device)
				self.optimizer.zero_grad()
				outputs = self.net(inputs)
				loss = self.criterion(outputs, targets)
				training_loss += loss.item()
				loss.backward()
				self.optimizer.step()
			time_c_comp += (time.time() - tic_c_comp)
		else: # offloading training
			for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader)):
				
				tic_c_comp = time.time()
				self.optimizer.zero_grad()
				inputs, targets = inputs.to(self.device), targets.to(self.device)
				outputs = self.net(inputs)
				time_c_comp += (time.time() - tic_c_comp)

				msg = ['MSG_LOCAL_ACTIVATIONS_CLIENT_TO_SERVER', outputs, targets] #normal
				# Estimation of communication time. There is a trick bug for communication time.
						# when the server are busy, there is a noticable time for waiting server gets the data from network interface
						# it causes the communication much larger than real transmission time.
						# Therefore, for SFL, LGL, FedGKT and ActionFed, we estimate the communication latency
				if batch_idx == 0:
					for i in range(3):
						msg = ['MSG_LOCAL_ACTIVATIONS_CLIENT_TO_SERVER', outputs, targets] #normal
						tic_acv_comm = time.time()
						self.send_msg(self.sock, msg)
						self.recv_msg(self.sock) # MSG_TIME_RECORD
						time_acv_comm += (time.time() - tic_acv_comm)
					time_acv_comm = (time_acv_comm / 3) * len(trainloader)
					logger.info('Estimated activation time: {:} and each iteration time {:}'.format(time_acv_comm, time_acv_comm/len(trainloader)))

				self.send_msg(self.sock, msg)

				
				if batch_idx == 0:
					for i in range(3):
						msg = self.recv_msg(self.sock)
						self.send_msg(self.sock, ['MSG_TIME_RECORD'])

				msg = self.recv_msg(self.sock)

				# Wait receiving server gradients
				tic_c_comp = time.time()
				gradients = msg[1].to(self.device)
				outputs.backward(gradients)
				self.optimizer.step()
				time_c_comp += (time.time() - tic_c_comp)
	
		msg = ['MSG_ROUND_FINISH']
		self.send_msg(self.sock, msg)

		self.recv_msg(self.sock) # MSG_GLOBAL_ROUND_FINISH
		return time_acv_comm, time_c_comp
		
	def upload(self):
		tic_aggre_comm = time.time()
		msg = ['MSG_LOCAL_WEIGHTS_CLIENT_TO_SERVER', self.net.state_dict(), self.ip]
		self.send_msg(self.sock, msg)
		self.recv_msg(self.sock) # MSG_TIME_RECORD
		time_aggre_comm = time.time() - tic_aggre_comm
		return time_aggre_comm

	def time_profile(self, time_acv_comm, time_aggre_comm, time_total_c, time_c_comp):
		msg = ['MSG_TIME_PROFILE', self.ip, time_acv_comm, time_aggre_comm, time_total_c, time_c_comp]
		self.send_msg(self.sock, msg)


