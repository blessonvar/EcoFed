

# EcoFed: Efficient Communication for DNN Partitioning-based Federated Learning

### About the research
Efficiently running federated learning (FL) on resource-constrained devices is challenging since they are required to train computationally intensive deep neural networks (DNN) independently. DNN partitioning-based FL (DPFL) has been proposed as one mechanism to accelerate training where the layers of a DNN (or computation) are offloaded from the device to the server. However, this creates significant communication overheads since the intermediate activation and gradient need to be transferred between the device and the server during training.

While current research reduces the communication introduced by DNN partitioning using local loss-based methods, we demonstrate that these methods are ineffective in improving the overall efficiency (communication overhead and training speed) of a DPFL system. This is because they suffer from accuracy degradation and ignore the communication costs incurred when transferring the activation from the device to the server. 

We proposes EcoFed - a communication efficient framework for DPFL systems. EcoFed eliminates the transmission of the gradient by developing pre-trained initialization of the DNN model on the device for the first time. This reduces the accuracy degradation seen in local loss-based methods. In addition, EcoFed proposes a novel replay buffer mechanism and implements a quantization-based compression technique to reduce the transmission of the activation. Figure 1 illustrates the training pipeline of classic FL, vanilla DPFL, local loss-based DPFL and EcoFed.

<p align = "center">
<img src = "Fig2-bv1.png">
</p>
<p>
<b>Fig.1 The training pipeline of classic FL, vanilla DPFL, local loss-based DPFL and EcoFed for three rounds of training. Classic FL transfers the entire model from the devices to the server at the end of each round. Vanilla DPFL only needs to upload a partitioned device-side model at the end of each round. However, Vanilla DPFL transfers the activation and gradient for each batch sample. Local loss-based DPFL reduces the communication by half since the gradients are computed locally. EcoFed reduces communication further as it transfers the activation only periodically (for example, once in two rounds) and further compresses the size of the activations.</b>
</p>

It is experimentally demonstrated that EcoFed can reduce the communication cost by up to 133x and accelerate training by up to 21x when compared to classic FL. Compared to vanilla DPFL, EcoFed achieves a 16x communication reduction and 2.86x training time speed-up.

Details of EcoFed can be found in our preprint article entitled, [EcoFed: Efficient Communication for DNN Partitioning-based Federated Learning](https://arxiv.org/pdf/2304.05495.pdf), Arxiv, 2023. The final version is under review of TPDS.

### Code Structure
The repository contains the source code of EcoFed. The overall architecture is divided as follows:

- EcoFed
  - core
    - server - Code for server object 
    - client - Code for client object
    - pretraining - Code for centralized pretraining 
    - data_generator - Code for I.I.D. and Non-I.I.D. data generator
    - client_sampler - Code for client sampler of each round

The code currently supports two modes: centralized simulation and real distributed training.

### Setting up the environment
The code is tested on Python 3 with Pytorch version 1.4 and torchvision 0.5.

To run in a real distributed testbed. Please install Pytorch and torchvision on each IoT device (for example, Raspberry Pis as used in this work). One can install from pre-built PyTorch and torchvision pip wheel. Download respective pip wheel as follows:
- Pyotrch: https://github.com/FedML-AI/FedML-IoT/tree/master/pytorch-pkg-on-rpi

All configuration options are given in `config.py`, which contains the architecture, model, and FL training hyperparameters.
Note that `config.py` file must be changed at the source edge server and at each device.  

#### Network configuration
SM is used for simulation test. For real distributed test using Raspberry Pis, please change the ips and hostnames accordingly.
```
testbed = 'SM'
# Network configuration
if testbed == 'SM': # Single machine
	SERVER_ADDR= '127.0.0.1'
	SERVER_PORT = 52000 # 52000, 53000, 54000, 55000, 56000
	IP2INDEX= {'127.0.0.1:'+str(SERVER_PORT+1):0, '127.0.0.1:'+str(SERVER_PORT+2):1, '127.0.0.1:'+str(SERVER_PORT+3):2, '127.0.0.1:'+str(SERVER_PORT+4):3, '127.0.0.1:'+str(SERVER_PORT+5):4}
if testbed == 'PI':
	SERVER_ADDR= '192.168.1.129'
	SERVER_PORT = 52000
	IP2INDEX= {'192.168.1.38:'+str(SERVER_PORT+1):0, '192.168.1.104:'+str(SERVER_PORT+2):1, '192.168.1.212:'+str(SERVER_PORT+3):2, '192.168.1.130:'+str(SERVER_PORT+4):3, '192.168.1.37:'+str(SERVER_PORT+5):4}
	## Mapping the hostname to IP
	HOST2IP = {'pi41':'192.168.1.38', 'pi42':'192.168.1.104', 'pi43':'192.168.1.212', 'pi44':'192.168.1.130', 'pi45':'192.168.1.37'}
```

#### FL hyperparameters
The total number of clients trained at each round is equal to K * C. The code will sequentially train each cluster, and K clients within each cluster will train in parallel. When all C clusters have completed training, one round of FL training is completed.

```
# FL setting
SEED = 0
K = 5 # Number of connected devices
C = 4 # Number of simulation clusters
ALPHA = 0.2 # Client sampling ratio
NON_IID = False # Whether use non_iid data generator
NUM_SHARDS = 500 # Number of shards
R = 500 # FL rounds
LR = 0.01 # Learning rate
B = 10 # Batch size
initilization = 'partial_pretrain' # Initilization, random, partial_pretrain and holistic_pretrain
pre_trained_weights_path = home + 'EcoFed_Project/pretrained/vgg11_imagenet_32.pth'
finetuning = False # Retraining
```

#### To run the code in simulation:
Change 'testbed' to 'SM' in the config and ensure that the number of 'python clientrun_simulation.py' in 'run_sim.sh' is set to K.

```
sh run_sim.sh cuda
```

#### To run the code in distributed testbed:
Change 'testbed' to 'PI' in the config.

##### Launch EcoFed server at the server

```
python serverrun_simulation.py cpu
```

##### Launch EcoFed at each IoT devices, e.g., device k
```
python clientrun_simulation.py k cpu
```

### Citation
TBD
