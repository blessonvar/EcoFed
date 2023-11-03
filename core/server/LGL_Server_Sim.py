# Server class
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import threading
from tqdm import tqdm
import time

import sys
sys.path.append('../')
from Communicator import *
from server.SFL_Server_Sim import *
import utils
import config

import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LGL_Server_Sim(SFL_Server_Sim):
    def __init__(self, ip_address, server_port, model_name, device, client_sampler, shard_indices):
        super(LGL_Server_Sim, self).__init__(ip_address, server_port, model_name, device, client_sampler, shard_indices)
        self.unit_auxiliary_net = utils.get_model('Device', 'Auxiliary_Net', -1, self.device, config.model_cfg)
        # zero auxiliary model for simulation
        self.zeronet_auxiliary = utils.get_model('Device', 'Auxiliary_Net', -1, self.device, config.model_cfg)
        logger.debug(self.unit_auxiliary_net)

        self.aggregrated_auxiliary_model_simulation = None # Aggregrated_auxiliary_model for simulation
        
    def initialize(self, OP, LR, R, current_c):
        super().initialize(OP, LR, R, current_c)
        
        # Auxiliary_Net Weight distribution with multiple threads
        auxiliary_init_cweights = self.unit_auxiliary_net.state_dict()
        self.threads = {}
        for client_ip in self.client_ips:
            self.threads[client_ip] = threading.Thread(target=self._thread_weights_distribution_, args=(client_ip, auxiliary_init_cweights,))
            logger.debug(str(client_ip) + ' weights distribution start.')
            self.threads[client_ip].start()

        for client_ip in self.client_ips:
            self.threads[client_ip].join()

        for client_ip in self.client_ips:
            logger.debug(str(client_ip) + ' weights distribution finish.')

    def _thread_training_offloading(self, client_ip, R, current_c):
        #training_loss = 0
        self.time_grad[client_ip] = 0
        self.time_comp[client_ip] = 0

        num_clients = (config.K * config.C) / config.ALPHA
        iteration = int((config.N / (num_clients * config.B)))
        
        for i in tqdm(range(iteration)):
            if i == 0:
                for j in range(3):
                    msg = self.recv_msg(self.client_socks[client_ip], 'MSG_LOCAL_ACTIVATIONS_CLIENT_TO_SERVER')
                    self.send_msg(self.client_socks[client_ip], ['MSG_TIME_RECORD']) #MSG_TIME_RECORD

            msg = self.recv_msg(self.client_socks[client_ip], 'MSG_LOCAL_ACTIVATIONS_CLIENT_TO_SERVER')

            smashed_layers = msg[1]
            labels = msg[2]
            
            inputs, targets = smashed_layers.to(self.device), labels.to(self.device)
            tic_comp = time.time()
            self.optimizers[client_ip].zero_grad()
            with self.lock:
                outputs = self.nets[client_ip](inputs)
            loss = self.criterions[client_ip](outputs, targets)
            #training_loss += loss.item()
            loss.backward()
            self.optimizers[client_ip].step()
            self.time_comp[client_ip] += (time.time() - tic_comp)
            
            # Releasing the threads competition
            msg = 'release threads msg'
            self.send_msg(self.client_socks[client_ip], msg)

    def aggregate_simulation(self):
        w_local_list =[]
        self.msgs = []
        self.threads = {}
        auxiliary_w_local_list =[] # Auxiliary_Net
        for client_ip in self.client_ips:
            self.threads[client_ip] = threading.Thread(target=self._thread_weights_collection_, args=(client_ip,))
            logger.debug(str(client_ip) + ' weights collection start.')
            self.threads[client_ip].start()

        for client_ip in self.client_ips:
            self.threads[client_ip].join()

        for client_ip in self.client_ips:
                logger.debug(str(client_ip) + ' weights collection finish.')
        
        for msg in self.msgs:
            ip = msg[3]
            msg[1] = utils.move_state_dict_to_device(msg[1], self.device)
            msg[2] = utils.move_state_dict_to_device(msg[2], self.device)
            if self.op[config.IP2INDEX[ip]] == -1:
                assert "Aggregration Error!"
            else:
                w_local = (utils.concat_weights(self.uninet.state_dict(),msg[1],self.nets[ip].state_dict()),1 / (config.K * config.C))
                w_local_list.append(w_local)

                auxiliary_w_local = (msg[2],1 / (config.K * config.C))
                auxiliary_w_local_list.append(auxiliary_w_local)

        if self.aggregrated_model_simulation is None: 
            self.aggregrated_model_simulation = utils.zero_init(self.zeronet).state_dict()
        if self.aggregrated_auxiliary_model_simulation is None:
            self.aggregrated_auxiliary_model_simulation = utils.zero_init(self.zeronet_auxiliary).state_dict()

        self.aggregrated_model_simulation = utils.fed_avg_simulation(self.aggregrated_model_simulation, w_local_list)
        self.aggregrated_auxiliary_model_simulation = utils.fed_avg_simulation(self.aggregrated_auxiliary_model_simulation, auxiliary_w_local_list)
        
        self.simulation_agg_control += config.K
        logger.info('simulation_agg_control: {:}'.format(self.simulation_agg_control))

        if self.simulation_agg_control > (config.K * config.C):
            assert 'Aggregration error!'

        if self.simulation_agg_control == (config.K * config.C):
            # Finish training of one round
            self.uninet.load_state_dict(self.aggregrated_model_simulation)
            self.aggregrated_model_simulation = None
            
            self.unit_auxiliary_net.load_state_dict(self.aggregrated_auxiliary_model_simulation)
            self.aggregrated_auxiliary_model_simulation = None

            self.simulation_agg_control = 0