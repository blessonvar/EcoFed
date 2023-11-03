import socket
import time
import multiprocessing

import sys
import config
import utils
from client import *
from data_generator import *

import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if config.testbed == 'SM':
	ip = '127.0.0.1'
	index = int(sys.argv[1])
if config.testbed == 'VM':
	s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
	s.connect(("8.8.8.8", 80))
	ip = s.getsockname()[0]
	s.close()
	index = config.IP2INDEX[ip]
if config.testbed == 'PI':
	ip = config.HOST2IP[socket.gethostname()]
	index = int(sys.argv[1])
	#index = config.IP2INDEX[ip]

# Building a client instance
device = sys.argv[2]
logger.info('Preparing Client')
if config.train_mode == 'FL':
	client = SFL_Client_Sim.SFL_Client_Sim(config.SERVER_ADDR, config.SERVER_PORT, config.model_name, ip, index, device)
if config.train_mode == 'SFL':
	client = SFL_Client_Sim.SFL_Client_Sim(config.SERVER_ADDR, config.SERVER_PORT, config.model_name, ip, index, device)
if config.train_mode == 'LGL':
	client = LGL_Client_Sim.LGL_Client_Sim(config.SERVER_ADDR, config.SERVER_PORT, config.model_name, ip, index, device)
if config.train_mode == 'FedGKT':
	client = FedGKT_Client_Sim.FedGKT_Client_Sim(config.SERVER_ADDR, config.SERVER_PORT, config.model_name, ip, index, device)
if config.train_mode == 'ActionFed':
	client = ActionFed_Client_Sim.ActionFed_Client_Sim(config.SERVER_ADDR, config.SERVER_PORT, config.model_name, ip, index, device)

# Building dataloader generator
#cpu_count = multiprocessing.cpu_count()
cpu_count = 2
num_clients = int((config.K * config.C) / config.ALPHA)
if config.NON_IID:
	dataloader_generator = Non_IID_Generator.Non_IID_Generator(cpu_count, config.dataset_name, num_clients, config.NUM_SHARDS, client.shard_indices)
else:
    dataloader_generator = IID_Generator.IID_Generator(cpu_count, config.dataset_name, num_clients, config.NUM_SHARDS, client.shard_indices)
    
for r in range(config.R):
    # Client IDs in the current round 
	clinet_ids = client.client_sampler.indices_samples[r*config.K * config.C : (r+1)*config.K * config.C]
	for c in range(config.C):
		logger.info('Preparing Data.')
		
		simulation_index = index + c * config.K
		client_id = clinet_ids[simulation_index]
		logger.info('Current Client ID {:}'.format(client_id))
		trainloader = dataloader_generator.get_local_dataloader(client_id)

		tic_total = time.time()
		logger.info('====================================>')
		logger.info('ROUND: {} START'.format(r))
		logger.info('Current OP: {}'.format(config.OP[r][simulation_index % config.K]))
		client.initialize(config.OP[r][simulation_index % config.K], config.LR, R=r, current_c = c)
		logger.info('==> Initialization Finish')
		
		time_acv_comm, time_c_comp = client.train(trainloader, r, current_c = c, client_id = client_id)
		logger.info('ROUND: {} END'.format(r))

		logger.info('==> Waiting for aggregration')
		time_aggre_comm = client.upload()
		
		time_total_c = time.time() - tic_total
		#client.time_profile(time_acv_comm, time_aggre_comm, time_total_c, time_c_comp)
		#logger.info('Current communication cost: ' + str(client.comm_cost))