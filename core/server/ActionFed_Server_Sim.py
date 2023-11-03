# Server class
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import threading
from tqdm import tqdm
import time
import random

import sys
sys.path.append('../')
from Communicator import *
from server.SFL_Server_Sim import *
import utils
import config

import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ActionFed_Server_Sim(SFL_Server_Sim):
    def __init__(self, ip_address, server_port, model_name, device, client_sampler, shard_indices):
        super(ActionFed_Server_Sim, self).__init__(ip_address, server_port, model_name, device, client_sampler, shard_indices)
        self.update_buffer = True # Buffer flag for each device
        self.buffer_data = {} # Buffer data
        self.buffer_labels = {} # Buffer labels
        #self.loss_s = {} # Record for server loss for each device
        #self.loss_c = {} # record of device loss for each device
            
        self.random_client_ids = None

    def initialize(self, OP, LR, R, current_c):
        self.op = OP #OP is a array of all clients
        self.nets = {}
        self.optimizers = {}
        self.criterions = {}
        self.time_ini = {}
        
        for client_ip in self.client_ips:
            self.time_ini[client_ip] = 0

        for client_ip in self.client_ips:
            ## Current version only support all devices have the same OP
            if len([i for i in self.op if i == -1]) == len(self.op): #Only if all clients is device-native training
                ## Weight initilization for each round
                if R == 0: # First round initilization
                    if config.initilization == 'random': 
                        init_cweights = self.uninet.state_dict()
                    if config.initilization == 'partial_pretrain':
                        cweights = utils.get_model('Device', self.model_name, 2, self.device, config.model_cfg).state_dict()
                        partial_pretrain_cweights = utils.transfer_weights_partial( config.pre_trained_weights_path,cweights,self.uninet.state_dict())
                        self.uninet.load_state_dict(partial_pretrain_cweights)
                        init_cweights = self.uninet.state_dict()
                    if config.initilization == 'holistic_pretrain':
                        holistic_pretrain_weights = utils.transfer_weights_holistic(config.pre_trained_weights_path,self.uninet.state_dict())
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
                        init_cweights = utils.transfer_weights_client(config.pre_trained_weights_path,cweights)
                    if config.initilization == 'holistic_pretrain':
                        holistic_pretrain_weights = utils.transfer_weights_holistic(config.pre_trained_weights_path,self.uninet.state_dict())
                        self.uninet.load_state_dict(holistic_pretrain_weights)
                        init_cweights = utils.split_weights_client(self.uninet.state_dict(),cweights)
                    ## pweight is init weights for server's model
                    pweights = utils.split_weights_server(self.uninet.state_dict(),cweights,self.nets[client_ip].state_dict())
                    self.nets[client_ip].load_state_dict(pweights)
                else: # Other rounds
                    init_cweights = utils.split_weights_client(self.uninet.state_dict(),cweights)
                    pweights = utils.split_weights_server(self.uninet.state_dict(),cweights,self.nets[client_ip].state_dict())
                    self.nets[client_ip].load_state_dict(pweights)
        # self.init_cweights is only used for aggregration
        self.init_cweights = init_cweights
        self.criterion = nn.CrossEntropyLoss() #Used for test

        # Dynamic control of period
        if len(self.testacc) <= 1:
            self.period = 2 # Start with period = 8
            self.delta_acc = 1
        else:
            if len(self.testacc) % 40 == 0:
                self.delta_acc = (self.testacc[-1] - self.testacc[len(self.testacc)-40]) / self.testacc[len(self.testacc)-40]
                if self.delta_acc > 10**-1:
                    self.period = 2
                if  10**-2 < self.delta_acc <= 10**-1:
                    self.period = 2
                if  10**-3 < self.delta_acc <= 10**-2:
                    self.period = 2
                if  self.delta_acc <= 10**-3:
                    self.period = 2
            
        logger.info('Delta Accuracy: ' + str(self.delta_acc))
        logger.info('Period: ' + str(self.period))

    def _thread_training_offloading(self, client_ip, R, current_c):
        self.time_comp[client_ip] = 0
        # Send buffer flag
        if R % self.period == 0:
            self.update_buffer = True
        else:
            self.update_buffer = False
        
        logger.info('Update Buffer: ' + str(self.update_buffer))
        msg = ['Update Buffer', self.update_buffer]
        self.send_msg(self.client_socks[client_ip], msg)
        
        if self.update_buffer:
            msg = self.recv_msg(self.client_socks[client_ip], 'MSG_LOCAL_CLIENT_ID_TO_SERVER')
            client_id = msg[1]

            self.buffer_data[client_id] = []
            self.buffer_labels[client_id] = []
            
    
        #training_loss = 0
        num_clients = (config.K * config.C) / config.ALPHA
        iteration = int((config.N / (num_clients * config.B))) # Simulation
        self.time_grad[client_ip] = 0
        
        
        for i in tqdm(range(iteration)):
            ## Buffer replay
            ## Update the buffer
            if self.update_buffer:
                '''
                if i == 0:
                    for j in range(3):
                        msg = self.recv_msg(self.client_socks[client_ip], 'MSG_LOCAL_ACTIVATIONS_CLIENT_TO_SERVER')
                        self.send_msg(self.client_socks[client_ip], ['MSG_TIME_RECORD']) #MSG_TIME_RECORD
                ''' 

                msg = self.recv_msg(self.client_socks[client_ip], 'MSG_LOCAL_ACTIVATIONS_CLIENT_TO_SERVER')
    
                self.buffer_data[client_id].append(msg[1].cpu()) # Move data from GPU memory to CPU memory
                self.buffer_labels[client_id].append(msg[2].cpu())

            else:
                # Releasing the threads competition
                msg = 'MSG_THREADS_CONTROL'
                self.send_msg(self.client_socks[client_ip], msg)
                self.recv_msg(self.client_socks[client_ip])

                client_round_index = config.IP2INDEX[client_ip] + config.K * current_c
                client_id = self.random_client_ids[client_round_index]

            smashed_layers = utils.dequant(self.buffer_data[client_id][i])
            labels = self.buffer_labels[client_id][i]
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
            msg = 'MSG_THREADS_CONTROL'
            self.send_msg(self.client_socks[client_ip], msg)
            
        ## Record loss_s
        #self.loss_s[client_ip] = training_loss/iteration
        
        ## Receive loss_c
        #msg = self.recv_msg(self.client_socks[client_ip], 'MSG_DEVICE_AVG_LOSS')
        #self.loss_c[client_ip] = msg[1]
        
        #logger.info(str(client_ip) + 'loss_c: ' + str(self.loss_c[client_ip]))
        #logger.info(str(client_ip) + 'loss_s: ' + str(self.loss_s[client_ip]))

    def aggregate_simulation(self):
        w_local_list =[]
        self.msgs = []
        
        for client_ip in self.client_ips:
            msg = ['MSG_LOCAL_WEIGHTS_CLIENT_TO_SERVER', self.init_cweights, client_ip]	
            self.msgs.append(msg)

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