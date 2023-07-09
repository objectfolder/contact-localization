import sys
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T

from .PointNet2.pointnet2 import Pointnet2_msg_backbone
from .ResNet import ResNet18

class CLR(nn.Module):
    def __init__(self, args):
        super(CLR, self).__init__()
        self.args = args
        self.pointnet2 = Pointnet2_msg_backbone()
        pointnet2_state_dict = torch.load('./exp/pretrained_backbones/pointnet2.pth',map_location='cpu')
        self.pointnet2.load_state_dict(pointnet2_state_dict, strict=False)
        self.mlp_point_cloud = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
        )
        
        self.decoder_input_size = 128
        if 'vision' in self.args.modality_list:
            self.vision_resnet18 = ResNet18(pretrained=True)
            self.mlp_vision = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
            )
            self.decoder_input_size += 128
        if 'touch' in self.args.modality_list:
            self.touch_resnet18 = ResNet18(pretrained=True)
            self.mlp_touch = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
            )
            self.decoder_input_size += 128
        if 'audio' in self.args.modality_list:
            self.audio_resnet18 = ResNet18(pretrained=True)
            self.audio_resnet18.net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.mlp_audio = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
            )
            self.decoder_input_size += 128
        
        self.decoder = nn.Sequential(
            nn.Linear(self.decoder_input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
            nn.Tanh()
        )
        self.reg_loss = nn.SmoothL1Loss()
        
    def encode_point_cloud(self, point_cloud):
        feature_point_cloud = self.pointnet2(point_cloud)
        feature_point_cloud = self.mlp_point_cloud(feature_point_cloud)
        return feature_point_cloud

    def encode_vision(self, vision):
        feature_vision = self.vision_resnet18(vision)
        feature_vision = self.mlp_vision(feature_vision)
        return feature_vision
    
    def encode_touch(self, touch):
        feature_touch = self.touch_resnet18(touch)
        feature_touch = self.mlp_touch(feature_touch)
        return feature_touch
    
    def encode_audio(self, audio):
        feature_audio = self.audio_resnet18(audio)
        feature_audio = self.mlp_audio(feature_audio)
        return feature_audio

    def forward(self, batch, calc_loss=True):
        output = {}
        point_cloud = batch['point_cloud'].cuda()
        bs = point_cloud.shape[0]
        feature_point_cloud = self.encode_point_cloud(point_cloud) # (bs, 128)
        
        feature_list=[feature_point_cloud]
        if 'vision' in self.args.modality_list:
            feature_vision = self.encode_vision(batch['vision'].cuda())
            feature_list.append(feature_vision)
        if 'touch' in self.args.modality_list:
            feature_touch = self.encode_touch(batch['touch'].cuda())
            feature_list.append(feature_touch)
        if 'audio' in self.args.modality_list:
            feature_audio = self.encode_audio(batch['audio'].cuda())
            feature_list.append(feature_audio)
            
        feature = torch.cat(feature_list, dim=1) # (bs, decoder_input_size)
        pred = self.decoder(feature) # (bs, 3)
        output['pred'] = pred
        
        if calc_loss:
            target = batch['contact_location'].cuda().float()
            output['target'] = target
            
            output['loss'] = self.reg_loss(target, pred)
            
        return output
