import torch

import sys
sys.path.append('../')
import config
import utils

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('==> Building model..')
net = utils.get_model('Cloud', 'VGG11', 0, device, config.model_cfg)
#net = utils.get_model('Cloud', 'ResNet9', 0, device, config.model_cfg)
print(net)
net = net.to(device)

checkpoint = torch.load('../../../pretrained/vgg11_fractaldb_60_32.pth')
net.load_state_dict(checkpoint)
torch.save(net.to('cpu').state_dict(), '../../../pretrained/vgg11_fractaldb_60_32_cpu.pth', _use_new_zipfile_serialization=False)
print('to cpu done!')
