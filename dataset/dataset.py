# Datasets of Contact Localization
# Yiming Dou (yimingdou@cs.stanford.edu)
# July 2022

from operator import mod
import os, sys
import os.path as osp
sys.path.append('/viscam/u/yimingdou/ObjectFolder-Benchmark/benchmarks/Contact_Localization_End_to_End/code/models/CLFDR/VGGish')
import json
from tqdm import tqdm
from itertools import product
import random
import trimesh
import librosa
import csv

import numpy as np
import torch
from torchvision import transforms as T
from PIL import Image
from scipy import signal
from scipy.io import wavfile
import logging
logger = logging.getLogger("trimesh")
logger.setLevel(logging.ERROR)

class retrieval_dataset(object):
    def __init__(self, args, set_type='train'):
        self.args = args
        self.set_type = set_type  # 'train' or 'val' or 'test'
        self.modality_list = args.modality_list # choose from ['vision', 'touch', 'audio']
        self.data_location = self.args.data_location
        self.normalize = True
            
        with open(osp.join(self.data_location,'scale.json')) as f:
            self.scale_dict=json.load(f)
        
        # preprocessing function of each modality
        self.preprocess = {
            'vision': T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ]),
            'touch': T.Compose([
                T.CenterCrop(320),
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ]),
            'audio': T.Compose([
                T.ToTensor(),
            ])
        }
        self.sample_idx = np.random.choice(1024, 1024, replace=False)
        # load candidates
        self.cand = []
        with open(self.args.split_location) as f:
            self.cand = json.load(f)[self.set_type]  # [[obj, contact]]
            
    def __len__(self):
        return len(self.cand)
    
    def load_modality_data(self, modality, obj, contact):
        if modality == 'touch' or modality == 'vision':
            data = Image.open(
                osp.join(self.data_location, modality,
                            str(obj), f'{contact}.png')
            ).convert('RGB')
            data = self.preprocess[modality](data)
        elif modality == 'audio':
            data = np.load(osp.join(self.data_location, 'audio_spectrogram',
                            str(obj), f'{contact}.npy'))
            data = torch.from_numpy(data).unsqueeze(dim=0).float()
        return torch.FloatTensor(data)
    
    def load_contact_location(self, obj, contact):
        contact_path=osp.join(self.data_location,'contacts',str(obj),f'{contact}.npy')
        contact_location = np.load(contact_path)[:3]
        return torch.tensor(contact_location)
    
    def load_point_cloud(self, obj, sample=1024):
        point_cloud_path = osp.join(
            self.data_location, 'global_gt_points', f'{obj}.npy')
        point_cloud = np.load(point_cloud_path)
        return torch.FloatTensor(point_cloud)
    
    # normalize the point cloud into the unit globe
    def normalize_point_cloud(self, batch):
        if self.normalize:
            mean = torch.mean(batch['point_cloud'], dim=1)
            batch['point_cloud'] -= mean.unsqueeze(1)
            scale = torch.max(torch.sqrt(torch.sum(batch['point_cloud'] ** 2, dim=2)),dim=1).values.unsqueeze(1)
            batch['point_cloud_scale'] = scale
            batch['point_cloud'] /= scale.unsqueeze(1)
            batch['contact_location'] -= mean
            batch['contact_location'] /= scale
        else:
            batch['point_cloud_scale'] = torch.ones((batch['point_cloud'].shape[0],1,1))
        batch['point_cloud'] = batch['point_cloud'][:,self.sample_idx]
        return batch
    
    def __getitem__(self, index):
        obj, contact = self.cand[index]
        data = {}
        data['names'] = [obj, contact]
        # load trajectory
        data['contact_location'] = self.load_contact_location(obj, contact)
        data['mesh_scale'] = self.scale_dict[str(obj)]
        data['point_cloud'] = self.load_point_cloud(obj, sample=1024)
        # load modality data
        for modality in self.modality_list:
            data[modality]=self.load_modality_data(modality, obj, contact)
        return data
    
    def collate(self, data):
        batch = {}
        batch['names'] = [item['names'] for item in data]
        batch['contact_location'] = torch.stack([item['contact_location'] for item in data])
        batch['mesh_scale'] = [item['mesh_scale'] for item in data]
        batch['point_cloud'] = torch.stack([item["point_cloud"] for item in data])
        for modality in self.modality_list:
            batch[modality] = torch.stack([item[modality] for item in data])
        batch = self.normalize_point_cloud(batch)
        return batch
        