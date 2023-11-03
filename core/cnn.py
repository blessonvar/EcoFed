import torch
import torch.nn as nn

import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CNN(nn.Module):
    def __init__(self, model_name, model_cfg, op, location):
        super(CNN, self).__init__()
        self.model_name = model_name
        self.model_cfg = model_cfg
        self.location = location
        self.op = op
        self.len = self._find_len(self.model_cfg[self.model_name])

        self.features, self.classifier = self._split_layers(self.model_cfg[self.model_name], self.op)

    def _find_len(self, cfg):
        layer_types = ['C', 'CR', 'CBR', 'FC', 'FCR', 'CTR', 'RB', 'RBMPD', 'RBAAP']
        length = 0
        if len(cfg) == 0:
            assert('Model cfg is empty')
        else:
            for l in cfg:
                if l[0] in layer_types:
                    length += 1
        return length

    def _find_op_index(self, cfg, op): # Find correct offloading layer (Conv or FC)
        count = 0
        index = 0
        layer_types = ['C', 'CR', 'CBR', 'FC', 'FCR', 'CTR', 'RB', 'RBMPD', 'RBAAP']
        if len(cfg) == 0:
            assert('Model cfg is empty')
        else:
            for l in cfg:
                index += 1
                if l[0] in layer_types:
                    count += 1
                    if count == op:
                        break

        for i in range(index,len(cfg)):
            if cfg[i][0] in ['C', 'CR', 'DP', 'CBR', 'FC', 'FCR', 'CTR', 'RB', 'RBMPD', 'RBAAP']:
                index = i
                return index

    def _split_layers(self, cfg, op):
        features = []
        classifier = []
        if isinstance(op, int) and op < self.len:
            if op == 0: # Cloud-native mode
                if self.location == 'Cloud':
                    cfg = cfg
                else:
                    assert ("Offloading point and location don't match.")

            elif op == -1: # Device-native model
                if self.location == 'Device':
                    cfg = cfg
                else:
                    assert ("Offloading point and location don't match.")

            else: # Offloading
                index = self._find_op_index(cfg, op)
                if self.location == 'Device':
                    cfg= cfg[0:index]

                if self.location == 'Cloud':
                    cfg= cfg[index:]
        else:
            logger.info('Offloading point (op) should be int type.')

        for x in cfg:
            if x[0] == 'C':
                features += [nn.Conv2d(x[1], x[2], kernel_size=x[3], stride=x[4], padding=x[5])]
            if x[0] == 'CR':
                features += [nn.Conv2d(x[1], x[2], kernel_size=x[3], stride=x[4], padding=x[5]),
                            nn.ReLU(inplace=True)]
            if x[0] == 'CBR':
                features += [nn.Conv2d(x[1], x[2], kernel_size=x[3], stride=x[4], padding=x[5]),
                            nn.BatchNorm2d(x[2], track_running_stats=False),
                            nn.ReLU(inplace=True)]
            if x[0] == 'RB' or x[0] == 'RBMPD' or x[0] == 'RBAAP':
                features += [Residual_Block(x)]
            if x[0] == 'CTR':
                features += [nn.ConvTranspose2d(x[1], x[2], kernel_size=x[3], stride=x[4], padding=x[5]),
                            nn.ReLU(inplace=True)]
            if x[0] == 'CTBR':
                features += [nn.ConvTranspose2d(x[1], x[2], kernel_size=x[3], stride=x[4], padding=x[5]),
                            nn.BatchNorm2d(x[2]),
                            nn.ReLU(inplace=True)]
            if x[0] == 'MP':
                features += [nn.MaxPool2d(kernel_size=x[1], stride=x[2])]
            if x[0] == 'MPP':
                features += [nn.MaxPool2d(kernel_size=x[1], stride=x[2], padding=x[3])]
            if x[0] == 'US':
                features += [nn.Upsample(scale_factor=x[1])]
            if x[0] == 'AAP':
                features += [nn.AdaptiveAvgPool2d((x[1], x[2]))]
            if x[0] == 'DP':
                classifier += [nn.Dropout(p=x[1])]
            if x[0] == 'FCR':
                classifier += [nn.Linear(x[1], x[2]),
                            nn.ReLU(inplace=True),]
            if x[0] == 'FC':
                classifier += [nn.Linear(x[1], x[2])]

        return nn.Sequential(*features), nn.Sequential(*classifier)

    def forward(self, x):
        if len(self.features) > 0:
            out = self.features(x)
        else:
            out = x
        if len(self.classifier) > 0:
            out = torch.flatten(out, 1)
            out = self.classifier(out)

        return out

class Residual_Block(nn.Module): #Residual block
    def __init__(self, cfg):
        super(Residual_Block, self).__init__()
        if cfg[0] == 'RB':
            self.type = 'RB'
            self.residual_block = nn.Sequential(*[nn.Conv2d(cfg[1], cfg[2], kernel_size=cfg[3], stride=cfg[4], padding=cfg[5]),
                            nn.BatchNorm2d(cfg[2], track_running_stats=False),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(cfg[2], cfg[2], kernel_size=cfg[3], stride=cfg[4], padding=cfg[5]),
                            nn.BatchNorm2d(cfg[2], track_running_stats=False),
                            nn.ReLU(inplace=True)])
        if cfg[0] == 'RBMPD':
            self.type = 'RBMPD'
            self.residual_block = nn.Sequential(*[nn.Conv2d(cfg[1], cfg[2], kernel_size=cfg[3], stride=cfg[4], padding=cfg[5]),
                            nn.BatchNorm2d(cfg[2], track_running_stats=False),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(cfg[2], cfg[2], kernel_size=cfg[3], stride=cfg[4], padding=cfg[5]),
                            nn.BatchNorm2d(cfg[2], track_running_stats=False),
                            nn.ReLU(inplace=True),
                            nn.MaxPool2d(kernel_size=cfg[6], stride=cfg[7], ceil_mode=True)])
            self.down_sampling = nn.Conv2d(cfg[1], cfg[2], kernel_size=1, stride=2, padding=0)
        if cfg[0] == 'RBAAP':
            self.type = 'RBAAP'
            self.residual_block = nn.Sequential(*[nn.Conv2d(cfg[1], cfg[2], kernel_size=cfg[3], stride=cfg[4], padding=cfg[5]),
                            nn.BatchNorm2d(cfg[2], track_running_stats=False),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(cfg[2], cfg[2], kernel_size=cfg[3], stride=cfg[4], padding=cfg[5]),
                            nn.BatchNorm2d(cfg[2], track_running_stats=False),
                            nn.ReLU(inplace=True)])
            self.adaptive_avg_pooling = nn.AdaptiveAvgPool2d((cfg[6], cfg[7]))
        
    def forward(self, x):
        if self.type == 'RB':
            out = x + self.residual_block(x)
        if self.type == 'RBMPD':
            out1 = self.down_sampling(x)
            out = self.residual_block(x) + out1
        if self.type == 'RBAAP':
            out = x + self.residual_block(x)
            out = self.adaptive_avg_pooling(out)
        return out

class autoencoder(nn.Module): #compressor
    def __init__(self, loc):
        super(autoencoder, self).__init__()
        self.loc = loc
        if loc == 'Unit' or loc == 'Device':
            self.encoder = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=1, padding=1),  # 64, 8, 8
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # 64, 4, 4
            nn.Conv2d(64, 32, 3, stride=1, padding=1),  # 32, 4, 4
            #nn.Conv2d(64, 8, 3, stride=1, padding=1),  
            nn.ReLU(True),
        )
        if loc == 'Unit' or loc == 'Server':
            self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 64, 2, stride=2, padding=0),  # 64 8, 8
            #nn.ConvTranspose2d(8, 64, 2, stride=2, padding=0),  
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),  # b, 128, 8, 8
            nn.ReLU(True),
        )

    def forward(self, x):
        if self.loc == 'Unit' or self.loc == 'Device':
            x = self.encoder(x)
        if self.loc == 'Unit' or self.loc == 'Server':
            x = self.decoder(x)
        return x

if __name__ == '__main__':
    model_cfg = {
    # (CR: Conv+Relu (#in, #out, kernel, stride, padding)
    # MP: MaxPooling (kernerl, stride),)
    # AAP: Adaptive Average Pooling (width, height)
    # DP: Dropout (ratio)
    # FCR: Fully Connect+Relu (#in, #out)
    # FC: Fully Connect (#in, #out)
    'ResNet9' : [('CBR', 3, 64, 3, 1, 1), ('MP', 2, 2),
                ('CBR', 64, 128, 3, 1, 1),('MP', 2, 2), 
                ('RBMPD', 128, 256, 3, 1, 1, 2, 2), 
                # Each RB block has two CBR layers
                ('RBMPD', 256, 512, 3, 1, 1, 2, 2),
                ('RBAAP', 512, 512, 3, 1, 1, 1, 1),
                ('FC', 512 * 1 * 1, 10)] # num_classes
}

    #test_split = [(-1, 'Device'), (1, 'Cloud'), (1, 'Device'), (2, 'Cloud'), (2, 'Device'),
                #(3, 'Cloud'), (3, 'Device'), (4, 'Cloud'), (4, 'Device'),
                #(5, 'Cloud'), (5, 'Device')]
    test_split = [(-1, 'Device')]
    for t in test_split:
        x = torch.rand(100, 3, 28, 28)
        model = CNN("ResNet9", model_cfg, t[0], t[1])
        logger.info(t[0])
        logger.info(t[1])
        logger.info(model)
        model(x)
        logger.info('#################################')
    