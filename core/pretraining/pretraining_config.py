import sys
#Config for pretraining

# Dataset configration
dataset_name = 'imagenet'
## Data length Tiny ImageNet:100000, Fractaldb-60:60000, VisDA_C:280157
## num_classes
if dataset_name == 'tiny-imagenet-200':
	N = 100000
	num_classes = 200
if dataset_name == 'fractaldb_cat60_ins1000':
	N = 60000
	num_classes = 60
if dataset_name == 'VisDA-C':
	N = 280157
	num_classes = 12
if dataset_name == 'SIP-17':
	N = 18000
	num_classes = 15
if dataset_name == 'caltech256':
	N = 30607
	num_classes = 257
if dataset_name == 'shvn':
	N = 73257
	num_classes = 10
if dataset_name == 'cifar-5m':
	N = 50000
	num_classes = 10
if dataset_name == 'imagenet':
	N = 1281167
	num_classes = 1000

model_name = 'VGG11'
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
	'VGG11' : [('CBR', 3, 64, 3, 1, 1), ('MP', 2, 2),
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
}


