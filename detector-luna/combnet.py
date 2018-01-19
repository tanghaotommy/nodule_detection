import torch
from torch import nn
from layers import *

config = {}
#config['anchors'] = [ 10.0, 30.0, 60.]
config['anchors'] = [5.0, 10.0, 30.0]
config['chanel'] = 1
config['crop_size'] = [128,128,128]
config['crop_size_global'] = [64, 64, 64]
config['stride'] = 4
config['max_stride'] = 16
config['num_neg'] = 800
config['th_neg'] = 0.02
config['th_pos_train'] = 0.5
config['th_pos_val'] = 1
config['num_hard'] = 2
config['bound_size'] = 12
config['reso'] = 1
config['sizelim'] = 6. #mm
config['sizelim2'] = 30
config['sizelim3'] = 40
config['aug_scale'] = True
config['r_rand_crop'] = 0.3
config['pad_value'] = 170
config['augtype'] = {'flip':True,'swap':False,'scale':True,'rotate':False}

# segmentation gives single lung but annotation is in other lung
single_lung_anno_outside = set(['LKDS-00598', # train
                                'LKDS-00648']) # train
# empty masks in preprocessing (see journal 05/29/17)
empty_masks = set([])#'LKDS-01343', # test2
                   #'LKDS-01068', # test2
                   #'LKDS-00678', # train
                   #'LKDS-00949', # train
                   #'LKDS-00602', # val
                   #'LKDS-00926', # val
                   #'LKDS-00137', # train
                   #'LKDS-00192', # train
                   #'LKDS-00322', # test
                   #'LKDS-00238', # train
                   #'LKDS-00823', # train
                   #'LKDS-00651', # train
                   #'LKDS-00476', # train
                   #'LKDS-00359', # train
                   #'LKDS-00385']) # test
config['blacklist'] = set.union(single_lung_anno_outside,\
                                empty_masks)



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # The first few layers consumes the most memory, so use simple convolution to save memory.
        # Call these layers preBlock, i.e., before the residual blocks of later layers.
        self.global_net = GlobalNet()
        self.preBlock = nn.Sequential(
            nn.Conv3d(1, 24, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(24),
            nn.ReLU(inplace = True),
            nn.Conv3d(24, 24, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(24),
            nn.ReLU(inplace = True))

        # 3 poolings, each pooling downsamples the feature map by a factor 2.
        # 3 groups of blocks. The first block of each group has one pooling.
        num_blocks_forw = [2,2,3,3]
        num_blocks_back = [3,3]
        self.featureNum_forw = [24,32,64,64,64]
        self.featureNum_back =    [128,64,64]
        for i in range(len(num_blocks_forw)):
            blocks = []
            for j in range(num_blocks_forw[i]):
                if j == 0:
                    blocks.append(PostRes(self.featureNum_forw[i], self.featureNum_forw[i+1]))
                else:
                    blocks.append(PostRes(self.featureNum_forw[i+1], self.featureNum_forw[i+1]))
            setattr(self, 'forw' + str(i + 1), nn.Sequential(*blocks))


        for i in range(len(num_blocks_back)):
            blocks = []
            for j in range(num_blocks_back[i]):
                if j == 0:
                    if i==0:
                        addition = 3
                    else:
                        addition = 0
                    blocks.append(PostRes(self.featureNum_back[i+1]+self.featureNum_forw[i+2]+addition, self.featureNum_back[i]))
                else:
                    blocks.append(PostRes(self.featureNum_back[i], self.featureNum_back[i]))
            setattr(self, 'back' + str(i + 2), nn.Sequential(*blocks))

        self.maxpool1 = nn.MaxPool3d(kernel_size=2,stride=2,return_indices =True)
        self.maxpool2 = nn.MaxPool3d(kernel_size=2,stride=2,return_indices =True)
        self.maxpool3 = nn.MaxPool3d(kernel_size=2,stride=2,return_indices =True)
        self.maxpool4 = nn.MaxPool3d(kernel_size=2,stride=2,return_indices =True)
        self.unmaxpool1 = nn.MaxUnpool3d(kernel_size=2,stride=2)
        self.unmaxpool2 = nn.MaxUnpool3d(kernel_size=2,stride=2)

        self.path1 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size = 2, stride = 2),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace = True))
        self.path2 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size = 2, stride = 2),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace = True))
        self.drop = nn.Dropout3d(p = 0.5, inplace = False)
        self.output = nn.Sequential(nn.Conv3d(self.featureNum_back[0] + 128, 64, kernel_size = 1),
                                    nn.ReLU(),
                                    #nn.Dropout3d(p = 0.3),
                                   nn.Conv3d(64, 5 * len(config['anchors']), kernel_size = 1))

    def forward(self, x, x2, coord):
        out_global_net = self.global_net(x2)
        size_global_net = out_global_net.size()
        out_global_net = out_global_net.unsqueeze(-1).expand(size_global_net[0], size_global_net[1], (config['crop_size'] / 4)**3).transpose(2,1).contiguous().view(size_global_net[0], config['crop_size'] / 4, config['crop_size'] / 4, config['crop_size'] / 4, size_global_net[1])
        
        out = self.preBlock(x)#16
        out_pool,indices0 = self.maxpool1(out)
        out1 = self.forw1(out_pool)#32
        out1_pool,indices1 = self.maxpool2(out1)
        out2 = self.forw2(out1_pool)#64
        #out2 = self.drop(out2)
        out2_pool,indices2 = self.maxpool3(out2)
        out3 = self.forw3(out2_pool)#96
        out3_pool,indices3 = self.maxpool4(out3)
        out4 = self.forw4(out3_pool)#96
        #out4 = self.drop(out4)

        rev3 = self.path1(out4)
        comb3 = self.back3(torch.cat((rev3, out3), 1))#96+96
        #comb3 = self.drop(comb3)
        rev2 = self.path2(comb3)

        comb2 = self.back2(torch.cat((rev2, out2,coord, out_global_net), 1))#64+64+128
        comb2 = self.drop(comb2)
        out = self.output(comb2)
        size = out.size()
        out = out.view(out.size(0), out.size(1), -1)
        #out = out.transpose(1, 4).transpose(1, 2).transpose(2, 3).contiguous()
        out = out.transpose(1, 2).contiguous().view(size[0], size[2], size[3], size[4], len(config['anchors']), 5)
        #out = out.view(-1, 5)
        return out

class GlobalNet(nn.Module):
    def __init__(self):
        super(GlobalNet, self).__init__()
        self.block1 = PostRes(1, 24)
        self.block2 = PostRes(24, 32)
        self.block3 = PostRes(32, 64)
        self.block4 = PostRes(64, 128)

        self.maxpool1 = nn.MaxPool3d(kernel_size=2,stride=2,return_indices =True)
        self.maxpool2 = nn.MaxPool3d(kernel_size=2,stride=2,return_indices =True)
        self.maxpool3 = nn.MaxPool3d(kernel_size=2,stride=2,return_indices =True)
        self.maxpool4 = nn.MaxPool3d(kernel_size=2,stride=2,return_indices =True)

        in_num = (config['crop_size_global'][0] / 2**4)**3*128
        self.fc1 = nn.Linear(in_num, 128)

    def forward(self, x):
        out1 = self.block1(x)
        out1_pool, _ = self.maxpool1(out1)
        out2 = self.block2(out1_pool)
        out2_pool, _ = self.maxpool2(out2)
        out3 = self.block3(out2_pool)
        out3_pool, _ = self.maxpool3(out3)
        out4 = self.block4(out3_pool)
        out4_pool, _ = self.maxpool4(out4)

        out = self.fc1(out4_pool)
        return out


def get_model():
    net = Net()
    loss = Loss(config['num_hard'])
    get_pbb = GetPBB(config)
    return config, net, loss, get_pbb
