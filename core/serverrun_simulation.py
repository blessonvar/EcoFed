import time
import pickle
import sys
import numpy as np
import random

import config
from server import *
from client_sampler import *
import utils

import logging
logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == '__main__':	
	# Generating shards indices for local data
	random.seed(config.SEED)
	shard_indices = random.sample(range(config.NUM_SHARDS), config.NUM_SHARDS)

	# Sampling clients for current round
	client_sampler = Random_Sampler.Random_Sampler(config.K, config.C, config.ALPHA, config.R)
	client_sampler.get_samplers()

	logger.info('Preparing Server.')
	device = sys.argv[1]
	if config.train_mode == 'FL':
		server = SFL_Server_Sim.SFL_Server_Sim(config.SERVER_ADDR, config.SERVER_PORT, config.model_name, device, client_sampler, shard_indices)
	if config.train_mode == 'SFL':
		server = SFL_Server_Sim.SFL_Server_Sim(config.SERVER_ADDR, config.SERVER_PORT, config.model_name, device, client_sampler, shard_indices)
	if config.train_mode == 'LGL':
		server = LGL_Server_Sim.LGL_Server_Sim(config.SERVER_ADDR, config.SERVER_PORT, config.model_name, device, client_sampler, shard_indices)
	if config.train_mode == 'FedGKT':
		server = FedGKT_Server_Sim.FedGKT_Server_Sim(config.SERVER_ADDR, config.SERVER_PORT, config.model_name, device, client_sampler, shard_indices)
	if config.train_mode == 'ActionFed':
		server = ActionFed_Server_Sim.ActionFed_Server_Sim(config.SERVER_ADDR, config.SERVER_PORT, config.model_name, device, client_sampler, shard_indices)

	res = [] # Record for test accuracy and communication cost
	training_mode = [] # Record for training mode

	for r in range(config.R):
		if r == 0:
			test_acc = server.test(r)
			if config.train_mode == 'ActionFed':
				res.append((test_acc, server.comm_cost, 8, 1))
			else:
				res.append((test_acc, server.comm_cost))

		logger.info('==> Current Client IDs: ')
		logger.info(server.client_sampler.indices_samples[r*config.K*config.C :(r+1)*config.K*config.C])

		# Random client ids for ActionFed buffer
		num_clients_per_round = config.K * config.C
		if config.train_mode == 'ActionFed' and len(server.buffer_data.keys()) > 0:
			if num_clients_per_round > len(server.buffer_data.keys()):
				server.random_client_ids = random.choices(list(server.buffer_data.keys()), k = num_clients_per_round)
			else:
				server.random_client_ids = random.sample(list(server.buffer_data.keys()), num_clients_per_round)
			print(server.random_client_ids)
   
		logger.info('==> Round {:} Start'.format(r))
		for c in range(config.C):
			tic_total = time.time()
			logger.info('====================================>')
			logger.info('==> Cluster {:} Start'.format(c))
			logger.info('Current OP: {}'.format(config.OP[r]))
			server.initialize(config.OP[r], config.LR, R=r, current_c = c)
			logger.info('==> Initialization Finish')
				
			server.train(R = r, current_c = c)
			server.aggregate_simulation()
			logger.info('Cluster Finish')
			
			time_total_s = time.time() - tic_total
			#time_record = server.time_profile(time_total_s)
			#logger.info('Current communication cost: ' + str(server.comm_cost) +' MB')
			#logger.info('Current training time: ' + str(time_total_s))

		logger.info('Round Finish')
		#time_total_s = time.time() - tic_total
		logger.info('Current communication cost: ' + str(server.comm_cost) +' MB')
		#logger.info('Round training time: ' + str(time_total_s))
		
		test_acc = server.test(r)
		if config.train_mode == 'ActionFed':
			res.append((test_acc, server.comm_cost, server.period, server.delta_acc))
		else:
			res.append((test_acc, server.comm_cost))
		
		
		if config.NON_IID:
			with open(config.home + 'EcoFed_Project/results/pp3/'+config.dataset_name+'/'+config.model_name+'/'+'res_acc_comm_'+config.train_mode+'_'+str(config.K * config.C)+
            '_'+'Non_IID_' + config.initilization +'_'+ str(config.ALPHA) +'_'+ str(config.SEED)+
            '.pkl','wb') as f:
					pickle.dump(res,f)
		else:
			with open(config.home + 'EcoFed_Project/results/pp3/'+config.dataset_name+'/'+config.model_name+'/'+'res_acc_comm_'+config.train_mode+'_'+str(config.K * config.C)+
            '_'+'IID_'+ config.initilization +'_'+ str(config.ALPHA) +'_'+ str(config.SEED)+
            '.pkl','wb') as f:
					pickle.dump(res,f)
		
		
		
	
