#FedGKT Client class
import torch
import torch.nn as nn
import torch.optim as optim
import time
import sys
from tqdm import tqdm

sys.path.append('../')
import config
import utils
from Communicator import *
from client.SFL_Client_Sim import *


import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FedGKT_Client_Sim(SFL_Client_Sim):
	def __init__(self, server_addr, server_port, model_name, ip, index, device):
		super(FedGKT_Client_Sim, self).__init__(server_addr, server_port, model_name, ip, index, device)

		# Simulation need more s_logits ids
		self.s_logits = {}
		for client_id in range(int(((config.C*config.K)/config.ALPHA))):
			self.s_logits[client_id] = []
		

	def initialize(self, OP, LR, R, current_c):
		super().initialize(OP, LR, R, current_c)
		
		self.fedgkt_alpha = 1.0
		self.fedgkt_temperature = 3.0
		self.criterion_KL = utils.KL_Loss(self.fedgkt_temperature)

		## initialize an auxiliary network.
		self.auxiliary_net = utils.get_model('Device', 'Auxiliary_Net', -1, self.device, config.model_cfg)
		logger.debug(self.auxiliary_net)
  
		logger.info('Receiving Global Auxiliary_Net Weights..')
		msg = self.recv_msg(self.sock)
		self.send_msg(self.sock, ['MSG_TIME_RECORD'])
		weights = msg[1]
		self.auxiliary_net.load_state_dict(weights)
  
		self.auxiliary_criterion = nn.CrossEntropyLoss()
		self.auxiliary_optimizer = optim.SGD(self.auxiliary_net.parameters(), lr=LR,
					momentum=0.9)
		logger.info('Initialize Auxiliary_Net Finished')
			
	def train(self, trainloader, R, current_c, client_id):
		# Training start
		time_acv_comm = 0
		time_c_comp = 0
		training_loss = 0
		self.net.to(self.device)

		self.net.train()
		## FedGKT need an auxiliary net
		self.auxiliary_net.to(self.device)
		self.auxiliary_net.train()
  
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
			## FedGKT
			tic_c_comp = time.time()
			for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader)):
				self.optimizer.zero_grad()
				## Zero gradients for auxiliary optimizer
				self.auxiliary_optimizer.zero_grad()
    
				inputs, targets = inputs.to(self.device), targets.to(self.device)
				outputs = self.net(inputs)
	
				local_outputs = self.auxiliary_net(outputs)
				## First round training without server logits
				if len(self.s_logits[client_id]) == 0:
					loss = self.auxiliary_criterion(local_outputs, targets) #normal loss
				else:
					loss_true = self.auxiliary_criterion(local_outputs, targets) #normal loss
					loss_kd = self.criterion_KL(local_outputs, self.s_logits[client_id][batch_idx])
					loss = loss_true + self.fedgkt_alpha * loss_kd #fedgkt distillation loss
				#training_loss += loss.item()
				loss.backward()
				self.auxiliary_optimizer.step()
				self.optimizer.step()
			time_c_comp += (time.time() - tic_c_comp)

		msg = ['MSG_ROUND_FINISH']
		self.send_msg(self.sock, msg)

		#FedGKT: inference phrase
		
		self.net.eval()
		self.auxiliary_net.eval()
		for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader)):
			inputs, targets = inputs.to(self.device), targets.to(self.device)
			with torch.no_grad():
				tic_c_comp = time.time()
				outputs = self.net(inputs)
				c_logits = self.auxiliary_net(outputs)
				time_c_comp += (time.time() - tic_c_comp)

				msg = ['MSG_LOCAL_ACTIVATIONS_CLIENT_TO_SERVER', outputs, targets, c_logits] #normal

				# Estimation of communication time. There is a trick bug for communication time.
						# when the server are busy, there is a noticable time for waiting server gets the data from network interface
						# it causes the communication much larger than real transmission time.
						# Therefore, for SFL, LGL, FedGKT and ActionFed, we estimate the communication latency
				if batch_idx == 0:
					# Waiting for the server
					self.recv_msg(self.sock) # MSG_TIME_RECORD
					for i in range(3):
						msg = ['MSG_LOCAL_ACTIVATIONS_CLIENT_TO_SERVER', outputs, targets, c_logits] #normal
						tic_acv_comm = time.time()
						self.send_msg(self.sock, msg)
						self.recv_msg(self.sock) # MSG_TIME_RECORD
						time_acv_comm += (time.time() - tic_acv_comm)
					time_acv_comm = (time_acv_comm / 3) * len(trainloader)
					logger.info('Estimated activation time: {:} and each iteration time {:}'.format(time_acv_comm, time_acv_comm/len(trainloader)))

				self.send_msg(self.sock, msg)

				if batch_idx == 0:
					msg = 'MSG_COM_FLAG'
					self.send_msg(self.sock, msg)
					for i in range(3):
						msg = self.recv_msg(self.sock)
						self.send_msg(self.sock, ['MSG_TIME_RECORD'])

				# Wait receiving server logits
				msg = self.recv_msg(self.sock)
				if len(self.s_logits[client_id]) < len(trainloader):
					self.s_logits[client_id].append(msg[1].to(self.device))
				else:
					self.s_logits[client_id] = []
					self.s_logits[client_id].append(msg[1].to(self.device))
  	
	
		msg = ['MSG_ROUND_FINISH']
		self.send_msg(self.sock, msg)

		self.recv_msg(self.sock) # MSG_GLOBAL_ROUND_FINISH
		return time_acv_comm, time_c_comp

	def upload(self):
		tic_aggre_comm = time.time()
		msg = ['MSG_LOCAL_WEIGHTS_CLIENT_TO_SERVER', self.net.state_dict(), self.auxiliary_net.state_dict(), self.ip]
		self.send_msg(self.sock, msg)
		self.recv_msg(self.sock) # MSG_TIME_RECORD
		time_aggre_comm = time.time() - tic_aggre_comm
		return time_aggre_comm



