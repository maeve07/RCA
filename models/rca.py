import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import init
affine_par = True
import sys
import utils
import time
import math
import cv2
import numpy as np
import os
import torch.utils.model_zoo as model_zoo
from .util import remove_layer
from .util import initialize_weights
import pdb
from matplotlib.pyplot import cm
import matplotlib.mlab as mlab
from matplotlib.ticker import NullFormatter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.cluster import KMeans
import random

model_urls = {'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth'}

class RCAModel(nn.Module):
    def  __init__(self, features, num_classes, all_channel=20, att_dir='./runs/', training_epoch=10, **kwargs):
        super(RCAModel, self).__init__()
        self.features = features
        self.extra_convs = nn.Sequential(
                        nn.Conv2d(512, 512, kernel_size=3, padding=1),
                        nn.ReLU(True),
                        nn.Conv2d(512, 512, kernel_size=3, padding=1),
                        nn.ReLU(True),
                        nn.Conv2d(512, 512, kernel_size=3, padding=1),
                        nn.ReLU(True),
                        nn.Conv2d(512,20,1)           
                    )
        for i in range(20):
            name = 'extra_convs' +str(i)
            setattr(self, name, 
                nn.Sequential(
                    nn.Conv2d(512, 512, kernel_size=3, padding=1),
                    nn.ReLU(True),
                    nn.Conv2d(512, 512, kernel_size=3, padding=1),
                    nn.ReLU(True),
                    nn.Conv2d(512, 512, kernel_size=3, padding=1),
                    nn.ReLU(True),
                    nn.Conv2d(512,1,1)           
                )
            )
        self.extra_F = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),   
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1)
        )
        
        self.in_channels = 512
        self.inter_channels = 512
        self.Q = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                            kernel_size=1, stride=1, padding=0)
        self.K = nn.Linear(in_features = self.in_channels, out_features = self.inter_channels)
        self.V = nn.Linear(in_features = self.in_channels, out_features = self.inter_channels)
        self.aggre = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                            kernel_size=1, stride=1, padding=0)
        self.concat_project = nn.Conv2d(in_channels=self.in_channels*2, out_channels=self.in_channels,
                            kernel_size=1, stride=1, padding=0)

        self.channel = all_channel
        d = self.channel // 2
        self.training_epoch = training_epoch
        self.att_dir = att_dir
        self.feat_dim = 512
        self.queue_len = 500
        self.cluter_num = 10
        self.momentum = 0.99
        self.alpha = 8

        self.temperature = 0.07

        if not os.path.exists(self.att_dir):
            os.makedirs(self.att_dir) 
        #-------------------------init weight---------------------------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
        #-----------------------set memory-----------------------------
        for i in range(0, 20):
            self.register_buffer("queue" + str(i), torch.randn(self.queue_len, self.feat_dim))
            self.register_buffer("queue_ptr" + str(i), torch.zeros(1, dtype=torch.long))
        
        #-----------------------set feat_cluter-----------------------------
        for i in range(0, 20):
            self.register_buffer("cluter" + str(i), torch.randn(self.cluter_num, self.feat_dim))
            
    def forward(self, input, label=None, epoch=1, index=None):
        
        if index ==0 and epoch > 0:
            #---------compute cluter---------
            for i in range(20):
                queue_i = getattr(self, "queue" + str(i))
                cluter_i = getattr(self, "cluter" + str(i))
                cluster = KMeans(n_clusters=self.cluter_num).fit(queue_i.cpu().numpy())
                cluter_i[:] = torch.from_numpy(cluster.cluster_centers_)

        batch_num  = input.size()[0]
        #extractor map
        x = self.features(input)
        self.x = x.clone()
        x_coarse = x.clone()     
        mid = x.clone()
        self.mid = self.extra_F(mid)
        #---------------------------------
        feat_cluter = getattr(self, "cluter0")
        feat_memory = getattr(self, "queue0")
        for k in range(1, 20):
            feat_memory = torch.cat((feat_memory, getattr(self, "queue" + str(k))), 0)
            feat_cluter = torch.cat((feat_cluter, getattr(self, "cluter" + str(k))), 0)
        query = x.clone()
        b, c, h, w = x.shape
        query = self.Q(query)
        query = query.permute(0, 2, 3, 1).view(b, w*h, c)
        key = self.K(feat_cluter.detach()).permute(1, 0)
        sim = torch.matmul(query, key) / self.cluter_num * 20
        value = self.V(feat_cluter.detach())
        query_ = torch.matmul(sim, value).permute(0, 2, 1).view(b, c, h, w)
        query_ = self.aggre(query_)
        x_ = torch.cat((x, query_), 1)
        x = self.concat_project(x_)
        #---------------------------------

        map_list = []
        for i in range(20):
            extra_convs_i = getattr(self, 'extra_convs' +str(i))
            x_i = extra_convs_i(x)
            map_list.append(x_i)
        map = torch.cat(map_list, 1)
        self.map = map.clone()
        map_coarse = self.extra_convs(x_coarse)
        self.map_coarse = map_coarse.clone()

        xs = F.avg_pool2d(self.map, kernel_size=(self.map.size(2),self.map.size(3)), padding=0).view(-1, 20)
        xss = F.avg_pool2d(self.map_coarse, kernel_size=(self.map_coarse.size(2),self.map_coarse.size(3)), padding=0).view(-1, 20)
        pre_probs = xs.clone() 
        probs = torch.sigmoid(pre_probs)
        if index != None and epoch > 0:
            atts = self.map.clone()
            atts[atts < 0] = 0
            for j in range(0, batch_num):
                ind = torch.nonzero(label[j])
                for i in range(ind.shape[0]):
                    la = ind[i]
                    accu_map_name = '{}/{}_{}.png'.format(self.att_dir, index+j, la.item())
                    att = atts[j, la].cpu().data.numpy()
                    att = np.rint(att / (att.max()  + 1e-8) * 255)

                    if epoch == self.training_epoch - 1 and not os.path.exists(accu_map_name):
                        cv2.imwrite(accu_map_name, np.transpose(att, (1, 2, 0)))
                        continue

                    if probs[j, la] < 0.1:  
                        continue
                            
                    try:
                        if not os.path.exists(accu_map_name):
                            cv2.imwrite(accu_map_name, np.transpose(att, (1, 2, 0)))
                        else:
                            accu_at = cv2.imread(accu_map_name, 0)
                            accu_at1 = accu_at[:,:,np.newaxis]
                            accu_at_max = np.maximum(accu_at1, np.transpose(att, (1, 2, 0)))
                            cv2.imwrite(accu_map_name,  accu_at_max)
                    except Exception as e:
                        print(e)

        
        map_soft = F.softmax(self.map_coarse, dim = 1)
        loss_cl_mixup = torch.zeros(1).cuda()
        for i in range(0, batch_num):
            ind = torch.nonzero(label[i])
            loss_i = torch.zeros(1).cuda()
            for j in range(ind.shape[0]):                
                mask_i = map_soft[i][ind[j]] > (torch.mean(map_soft[i][ind[j]]))
                self.mid_sel = self.mid[i] * mask_i.float().detach()
                x_mid_pool = self.mid_sel.reshape(self.mid_sel.shape[0], -1).sum(1)/mask_i.float().sum().detach()

                #--------embedding mix-up--------
                mix_i = torch.randperm(batch_num)
                ind_mix = torch.nonzero(label[mix_i[i]])
                label_mix = ind_mix[torch.randperm(ind_mix.shape[0])[0]]

                mask_i_mix = map_soft[mix_i[i]][label_mix] > (torch.mean(map_soft[mix_i[i]][label_mix]))
                self.mid_sel_mix = self.mid[mix_i[i]] * mask_i_mix.float().detach()
                x_mid_pool_mix = self.mid_sel_mix.reshape(self.mid_sel_mix.shape[0], -1).sum(1)/mask_i_mix.float().sum().detach()
                #--------------------------------
                lam = np.random.beta(self.alpha, self.alpha)
                x_mid_pool = x_mid_pool * lam + (1-lam) * x_mid_pool_mix

                x_mid_pool_norm = F.normalize(x_mid_pool.unsqueeze(0)).squeeze()
                feat_neg = F.normalize(feat_memory)

            
                similarity_neg = torch.einsum('c,nc->n', [x_mid_pool_norm, feat_neg.detach()])
                logit_neg = torch.div(similarity_neg, self.temperature)
                max_log = torch.max(logit_neg)
                exp_logit_neg = torch.exp(logit_neg - max_log.detach())
                label_q = torch.arange(20)
                label_q = label_q.expand(self.queue_len, 20).permute(1, 0).reshape(self.queue_len * 20).cuda()
                #--------------closs_1------------------
                feat_pos = F.normalize(getattr(self, "queue" + str(ind[j].item())))
                mask = label_q != ind[j]
                similarity_pos = torch.einsum('c,nc->n', [x_mid_pool_norm, feat_pos.detach()])
                logit_pos = torch.div(similarity_pos, self.temperature)
                logit_pos = logit_pos - max_log.detach()
                exp_logit_pos = torch.exp(logit_pos)
                
                l_neg = (exp_logit_neg * mask.float().detach()).sum().expand(self.queue_len)
                loss_i_1 = (-(logit_pos - torch.log((l_neg + exp_logit_pos).clamp(min=1e-4)))).mean()

                #--------------closs_2------------------
                feat_pos = F.normalize(getattr(self, "queue" + str(label_mix.item())))
                mask = label_q != label_mix
                
                similarity_pos = torch.einsum('c,nc->n', [x_mid_pool_norm, feat_pos.detach()])
                logit_pos = torch.div(similarity_pos, self.temperature)
                logit_pos = logit_pos - max_log.detach()
                exp_logit_pos = torch.exp(logit_pos)
                
                l_neg = (exp_logit_neg * mask.float().detach()).sum().expand(self.queue_len)

                loss_i_2 = (-(logit_pos - torch.log((l_neg + exp_logit_pos).clamp(min=1e-4)))).mean()

                loss_i += loss_i_1 * lam + loss_i_2 * (1-lam)
            loss_cl_mixup = loss_cl_mixup + loss_i * 1.0 / ind.shape[0]
        
        #---------------------------enqueue&dequeue------------------------------------
        for j in range(0, batch_num):
            self._dequeue_and_enqueue(self.x[j], self.map_coarse[j], label[j], probs[j])
            
        return xss, xs, loss_cl_mixup.mean()

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Momentum update of the key encoder."""
        for param_q, param_k in zip(self.features.parameters(),
                                    self.features_mom.parameters()):
            param_k.data = param_k.data * self.momentum + \
                           param_q.data * (1. - self.momentum)
        
        for param_q, param_k in zip(self.extra_F.parameters(),
                                    self.extra_F_mom.parameters()):
            param_k.data = param_k.data * self.momentum + \
                           param_q.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, x, map,  label, probs):
        map = F.softmax(map, dim = 0)
        for ind, cla in enumerate(label):
            if cla == 1:
                if(probs[ind] > 0.7):
                    mask = map[ind] > (torch.mean(map[ind]))
                    # x: c x h x w
                    x = x * mask.float()
                    embedding = x.reshape(x.shape[0], -1).sum(1)/mask.float().sum()
                    queue_i = getattr(self, "queue" + str(ind))
                    queue_ptr_i = getattr(self, "queue_ptr" + str(ind))

                    ptr = int(queue_ptr_i)

                    queue_i[ptr:ptr + 1] = queue_i[ptr:ptr + 1] * self.momentum + embedding * (1 - self.momentum)
                    ptr = (ptr + 1) % self.queue_len  # move pointer

                    queue_ptr_i[0] = ptr


 
    def get_parameter_groups(self):
        groups = ([], [], [], [])
        for name, value in self.named_parameters():

            if 'extra' in name:
                if 'weight' in name:
                    groups[2].append(value)
                else:
                    groups[3].append(value)
            else:
                if 'weight' in name:
                    groups[0].append(value)
                else:
                    groups[1].append(value)
        return groups

def adjust_pretrained_model(pretrained_model, current_model):
    def _get_keys(obj, split):
        keys = []
        iterator = obj.items() if split == 'pretrained' else obj
        for key, _ in iterator:
            if key.startswith('features.'):
                keys.append(int(key.strip().split('.')[1].strip()))
        return sorted(list(set(keys)), reverse=True)

    def _align_keys(obj, key1, key2):
        for suffix in ['.weight', '.bias']:
            old_key = 'features.' + str(key1) + suffix
            new_key = 'features.' + str(key2) + suffix
            obj[new_key] = obj.pop(old_key)
        return obj

    pretrained_keys = _get_keys(pretrained_model, 'pretrained')
    current_keys = _get_keys(current_model.named_parameters(), 'model')

    for p_key, c_key in zip(pretrained_keys, current_keys):
        pretrained_model = _align_keys(pretrained_model, p_key, c_key)

    return pretrained_model
   
def load_pretrained_model(model, path=None):
    state_dict = model_zoo.load_url(model_urls['vgg16'], progress=True)
    state_dict = remove_layer(state_dict, 'classifier.')
    state_dict = adjust_pretrained_model(state_dict, model)
    model.load_state_dict(state_dict, strict=False)
    return model

def make_layers(cfg, batch_norm=False,**kwargs):
    layers = []
    in_channels = 3
    for i, v in enumerate(cfg):
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'N':
            layers += [nn.MaxPool2d(kernel_size=3, stride=1, padding=1)]
        else:
            if i > 13:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, dilation=2, padding=2)            
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'D1': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'N', 512, 512, 512],
    'D2':[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 'A', 512, 512, 512, 'A', 512, 512, 512, 'A'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def RCA_VGG(pretrained=True, **kwargs):
    model = RCAModel(make_layers(cfg['D1'],**kwargs), **kwargs)
    if pretrained:
        model = load_pretrained_model(model,path=None)
    return model
