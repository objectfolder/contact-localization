import sys
sys.path.append('/viscam/u/yimingdou/ObjectFolder-Benchmark/benchmarks/Contact_Localization_End_to_End/code/models/CLFDR/VGGish')
import librosa
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
from torchvision import models
from pytorch3d.loss import chamfer_distance

from .PointNet2.pointnet2 import Pointnet2_msg_backbone
from .ResNet import ResNet18, ResNet50

from .VGGish.vggish import VGGish
from .VGGish.audioset import vggish_input, vggish_postprocess


class CLFDR(nn.Module):
    def __init__(self, args):
        super(CLFDR, self).__init__()
        self.args = args
        self.pointnet2 = Pointnet2_msg_backbone()
        pointnet2_state_dict = torch.load('./exp/pretrained_backbones/pointnet2.pth',map_location='cpu')
        self.pointnet2.load_state_dict(pointnet2_state_dict, strict=False)
        self.mlp_point_cloud = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, 128),
        )
        self.point_cloud_recon_decoder = nn.Sequential(
            nn.Linear(32, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 3),
            nn.Tanh()
        )
        
        self.decoder_input_size = 0
        if 'vision' in self.args.modality_list:
            self.vision_resnet50 = ResNet50(pretrained=True)
            self.mlp_vision = nn.Sequential(
                nn.Linear(2048, 512),
                nn.Dropout(0.25),
                nn.LeakyReLU(),
                nn.Linear(512, 256),
                nn.Dropout(0.25),
                nn.LeakyReLU(),
                nn.Linear(256, 128),
            )
            self.decoder_input_size += 128
        if 'touch' in self.args.modality_list:
            self.touch_resnet50 = ResNet50()
            touch_resnet50_state_dict = torch.load('./exp/pretrained_backbones/touch_resnet50.pth',map_location='cpu')
            self.touch_resnet50.load_state_dict(touch_resnet50_state_dict, strict=True)
            self.mlp_touch = nn.Sequential(
                nn.Linear(2048, 512),
                nn.Dropout(0.25),
                nn.LeakyReLU(),
                nn.Linear(512, 256),
                nn.Dropout(0.25),
                nn.LeakyReLU(),
                nn.Linear(256, 128),
            )
            self.decoder_input_size += 128
        if 'audio' in self.args.modality_list:
            self.vggish = VGGish()
            vggish_state_dict = torch.load('./exp/pretrained_backbones/audio_vggish.pth',map_location='cpu')
            self.vggish.load_state_dict(vggish_state_dict, strict=True)
            self.mlp_audio = nn.Sequential(
                nn.Linear(128, 128),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(),
                nn.Linear(128, 128),
            )
            self.decoder_input_size += 128
        
        self.decoder = nn.Sequential(
            nn.Linear(self.decoder_input_size, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 32),
        )
        self.reg_loss = nn.SmoothL1Loss()
        self.recon_loss = chamfer_distance
        
        self.disentangle_weight = nn.Parameter(torch.Tensor(512, 128, 64))
        self.disentangle_mlp = nn.Sequential(
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 32),
        )
        
        self.dist_predictor = nn.Sequential(
            nn.Linear(64, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 4),
            nn.LeakyReLU(),
            nn.Linear(4, 1),
            nn.Sigmoid()
        )
        nn.init.normal_(self.disentangle_weight)
        
        # state_dict = torch.load('./exp/test/latest.pth',map_location='cpu')
        # self.load_state_dict(state_dict)

    def encode_point_cloud(self, point_cloud):
        feature_point_cloud = self.pointnet2(point_cloud)
        feature_point_cloud = self.mlp_point_cloud(feature_point_cloud)
        return feature_point_cloud

    def disentangle_feature_point_cloud(self, feature_point_cloud):
        feature_point_cloud_disentangle = torch.einsum('ik,jkl->ijl',feature_point_cloud,self.disentangle_weight) # (bs, 512, 64)
        feature_point_cloud_disentangle = F.leaky_relu(feature_point_cloud_disentangle) # (bs, 512, 64)
        feature_point_cloud_disentangle = self.disentangle_mlp(feature_point_cloud_disentangle) # (bs, 512, 32)
        return feature_point_cloud_disentangle
    
    def decode_point_cloud(self, feature_point_cloud):
        point_cloud_recon = self.point_cloud_recon_decoder(feature_point_cloud)
        return point_cloud_recon
    
    def encode_vision(self, vision):
        feature_vision = self.vision_resnet50(vision)
        feature_vision = self.mlp_vision(feature_vision)
        return feature_vision
    
    def encode_touch(self, touch):
        feature_touch = self.touch_resnet50(touch)
        feature_touch = self.mlp_touch(feature_touch)
        return feature_touch
    
    def encode_audio(self, audio):
        feature_audio = self.vggish(audio)
        feature_audio = self.mlp_audio(feature_audio)
        return feature_audio

    def forward(self, batch, calc_loss=True):
        output = {}
        point_cloud = batch['point_cloud'].cuda()
        target = batch['contact_location'].cuda().float()
        bs=point_cloud.shape[0]
        feature_point_cloud = self.encode_point_cloud(point_cloud) # (bs, 128)
        import ipdb
        feature_point_cloud_disentangle = self.disentangle_feature_point_cloud(feature_point_cloud) # (bs, 512, 32)
        point_cloud_recon = self.decode_point_cloud(feature_point_cloud_disentangle) # (bs, 512, 3)
        
        feature_list=[]
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
        
        feature = self.decoder(feature).unsqueeze(1) # (bs, 1, 32)
        feature = feature.repeat(1,512,1) # (bs, 512, 32)
        feature = torch.cat((feature,feature_point_cloud_disentangle),dim=2) # (bs, 512, 64)
        target_dist_pred = self.dist_predictor(feature).squeeze(2) # (bs, 512)
        
        pred_index = torch.argmin(target_dist_pred, 1)
        pred = torch.stack([point_cloud_recon[i][pred_index[i]] for i in range(bs)])
        
        output['pred'] = pred
        
        if calc_loss:
            output['target'] = target
            target_dist = torch.cdist(target.unsqueeze(1), point_cloud_recon, 2)[:, 0] # (bs, 512)
            output['reg_loss'] = torch.mean((2*target_dist_pred-target_dist)**2)
            output['recon_loss'] = self.recon_loss(point_cloud, point_cloud_recon)[0]
            output['loss'] = 1*output['reg_loss'] + 10*output['recon_loss']
            
            if self.training:
                global_gt_points = point_cloud.detach().cpu().numpy()
                global_pred_points = point_cloud_recon.detach().cpu().numpy()
                for i in range(global_gt_points.shape[0]):
                    np.random.shuffle(global_gt_points[i])
                    np.random.shuffle(global_pred_points[i])
                np.save('gt.npy',global_gt_points[:512])
                np.save('pred.npy',global_pred_points[:512])
            
        return output
