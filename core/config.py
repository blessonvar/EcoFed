import sys
#Config for ActionFed

testbed = 'SM'
# Network configration
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

# Dataset configration
dataset_name = 'CIFAR10'
home = sys.path[0].split('EcoFed')[0]
dataset_path = home + 'EcoFed_Project/dataset/'+ dataset_name +'/'
## Data length CIFAR10:50000, CIFAR100:50000, MNIST:60000, FMNIST:60000
## num_classes
if dataset_name == 'CIFAR10':
	N = 50000
	num_classes = 10
if dataset_name == 'CIFAR100':
	N = 50000
	num_classes = 100

# Model configration
model_name = 'VGG11'
if dataset_name == 'CIFAR10' or dataset_name == 'CIFAR100':
	model_cfg = {
	# (CR: Conv+Relu (#in, #out, kernel, stride, padding)
	# (CBR: Conv+BN+Relu (#in, #out, kernel, stride, padding)
	# (CTR: ConvTransposed+Relu (#in, #out, kernel, stride, padding)
	# (CTBR: ConvTransposed+BN+Relu (#in, #out, kernel, stride, padding)
	# MP: MaxPooling (kernerl, stride),)
	# AAP: Adaptive Average Pooling (width, height)
	# DP: Dropout (ratio)
	# FCR: Fully Connect+Relu (#in, #out)
	# FC: Fully Connect (#in, #out)
	# RBD Residual Block with Downsampling (#in, #out, kernel, stride, padding, pooling kernel, pooling stride)
	'AlexNet' : [('CBR', 3, 64, 3, 2, 1), ('MP', 2, 2),
				('CBR', 64, 192, 3, 1, 1),('MP', 2, 2),
				('CBR', 192, 384, 3, 1, 1),('CBR', 384, 256, 3, 1, 1),('CBR', 256, 256, 3, 1, 1),('MP', 2, 2),
				('AAP', 2, 2),
				#('DP', 0.5), 
				('FCR', 256 * 2 * 2, 4096),
				#('DP', 0.5), 
				('FCR', 4096, 4096),
				('FC', 4096, num_classes)] #num_classes

	,'VGG11' : [('CBR', 3, 64, 3, 1, 1), ('MP', 2, 2),
				('CBR', 64, 128, 3, 1, 1),('MP', 2, 2),
				('CBR', 128, 256, 3, 1, 1),('CBR', 256, 256, 3, 1, 1),('MP', 2, 2),
				('CBR', 256, 512, 3, 1, 1),('CBR', 512, 512, 3, 1, 1),('MP', 2, 2),
				('CBR', 512, 512, 3, 1, 1),('CBR', 512, 512, 3, 1, 1),
				#('DP', 0.5), 
				('FCR', 512 * 2 * 2, 4096),
				#('DP', 0.5), 
				('FCR', 4096, 4096),
				('FC', 4096, num_classes)]

	,'ResNet9' : [('CBR', 3, 64, 3, 1, 1), ('MP', 2, 2),
				('CBR', 64, 128, 3, 1, 1),('MP', 2, 2), 
				# Each RB block has two CBR layers
				('RBMPD', 128, 256, 3, 1, 1, 2, 2), 
				('RBMPD', 256, 512, 3, 1, 1, 2, 2),
				('RBAAP', 512, 512, 3, 1, 1, 1, 1),
				('FC', 512 * 1 * 1, num_classes)]

	,'Auxiliary_Net' : [('FC', 128 * 8 * 8, num_classes)]
}

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

## training modes: 'FL', 'SFL', 'LGL', 'FedGKT', 'ActionFed'.
train_mode = 'ActionFed'
if train_mode == 'FL':  
	OP = [[-1 for j in range(K)] for i in range(R)]
else:
	OP = [[2 for j in range(K)] for i in range(R)]