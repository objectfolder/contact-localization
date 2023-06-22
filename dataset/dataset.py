# Datasets of Contact Localization
# Yiming Dou (yimingdou@cs.stanford.edu)
# July 2022

from operator import mod
import os
import os.path as osp
import csv
import json
from tqdm import tqdm
from itertools import product
import random
import trimesh
import librosa

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
        
        with open(osp.join(self.data_location,'scale.json')) as f:
            self.scale_dict=json.load(f)
        
        # preprocessing function of each modality
        self.preprocess = {
            'touch': T.Compose([
                T.CenterCrop(160),
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ]),
            'audio': T.Compose([
                T.ToTensor(),
            ])
        }
        
        # load candidates
        with open(self.args.split_location) as f:
            self.cand = json.load(f)[self.set_type]  # [[obj, traj]]
            
    def __len__(self):
        return len(self.cand)
    
    def load_data(self, modality, obj, traj):
        data = []
        if modality == 'touch':
            contact_index = np.load(osp.join(self.data_location,'contact_index', str(obj),f'{traj}.npy'))
            for i in range(self.args.trajectory_length):
                cur_data = Image.open(
                    osp.join(self.data_location, modality,
                                str(obj), f'{contact_index[i]}.png')
                ).convert('RGB')
                cur_data = self.preprocess[modality](cur_data)
                data.append(cur_data)
            data = torch.stack(data)
        elif modality == 'audio':
            data = np.load(osp.join(self.data_location,'audio_mfcc',f'{obj}.npy'))
        return torch.FloatTensor(data)
    
    def load_trajectory(self, obj, traj):
        traj_path=osp.join(self.data_location,'trajectory',str(obj),f'{traj}.npy')
        trajectory = np.load(traj_path)
        return torch.tensor(trajectory)
    
    def load_mesh(self, obj):
        obj_path=osp.join(self.data_location,'OF_all',str(obj),'model.obj')
        obj_mesh = trimesh.load(obj_path,force='mesh',file_type='obj')
        obj_mesh_scale = self.scale_dict[str(obj)]
        return obj_mesh, obj_mesh_scale
    
    def __getitem__(self, index):
        obj, traj = self.cand[index]
        data = {}
        data['names'] = [obj, traj]
        # load trajectory
        data['trajectory']=self.load_trajectory(obj, traj)
        data['mesh'], data['mesh_scale'] = self.load_mesh(obj)
        # load modality data
        for modality in self.modality_list:
            data[modality]=self.load_data(modality, obj, traj)
        return data
    
    def collate(self, data):
        batch = {}
        batch['names'] = [item['names'] for item in data]
        batch['trajectory'] = torch.stack([item['trajectory'] for item in data])
        batch['mesh'] = [item['mesh'] for item in data]
        batch['mesh_scale'] = [item['mesh_scale'] for item in data]
        for modality in self.modality_list:
            if modality == 'touch':
                batch[modality] = torch.stack([item[modality] for item in data])
            elif modality == 'audio':
                batch[modality] = [item[modality] for item in data]
        return batch
        