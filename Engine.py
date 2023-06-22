import os
import os.path as osp
import sys
import json
from pprint import pprint

from tqdm import tqdm, trange
import numpy as np
from scipy.spatial.distance import cdist
import torch
import torch.optim as optim

import utils.meters as meters
from models.build import build as build_model
from dataset.build import build as build_dataset

class Engine():
    def __init__(self, args, cfg):
        self.args = args
        self.cfg = cfg
        # set seeds
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed(self.args.seed)
        # build dataloaders
        self.train_loader, self.val_loader, self.test_loader = build_dataset(self.args)
        # # build model & optimizer
        self.model, self.optimizer = build_model(self.args, self.cfg)
        # experiment dir
        self.exp_dir = osp.join('./exp_new',self.args.exp)
        os.makedirs(self.exp_dir, exist_ok=True)
        
    def train_epoch(self, epoch):
        pass
        
    @torch.no_grad()
    def eval_epoch(self, epoch=0, test = False):
        L2_dist,L2_dist_normalized = {},{}
        data_loader = self.test_loader if test else self.val_loader
        for i, batch in tqdm(enumerate(data_loader), leave = False):
            output = self.model(batch)
            pred_contact_point = output['pred_contact_point']
            gt_contact_point = output['gt_contact_point']
            for i in range(gt_contact_point.shape[0]):
                obj = batch['names'][i][0]
                if not obj in L2_dist:
                    L2_dist[obj]=meters.AverageMeter()
                    L2_dist_normalized[obj]=meters.AverageMeter()
                dist = float(torch.dist(pred_contact_point[i], gt_contact_point[i], 2))*100
                L2_dist[obj].update(dist)
                L2_dist_normalized[obj].update(dist/batch['mesh_scale'][i])
            for obj, dist in L2_dist.items():
                print("Obj: {}, avg distance: {:.2f} (cm), avg normalized distacne: {:.2f}".format(obj, dist.avg, L2_dist_normalized[obj].avg))
            result_path = osp.join(self.exp_dir, 'result.json')
            json.dump({"Test Result":{'Distance (cm)':{obj:dist.avg for obj, dist in L2_dist.items()},'Normalized distance (distance/scale*100%)':{obj:dist.avg for obj, dist in L2_dist_normalized.items()}}}, open(result_path,'w'))
                
        L2_dist = {obj:dist.avg for obj, dist in L2_dist.items()}
        L2_dist_normalized = {obj:dist.avg for obj, dist in L2_dist_normalized.items()}
        L2_dist['all']=np.mean(list(L2_dist.values()))
        L2_dist_normalized['all']=np.mean(list(L2_dist_normalized.values()))
        return L2_dist, L2_dist_normalized
    
    def test(self):
        print("Start Testing")
        L2_dist, L2_dist_normalized = self.eval_epoch(test = True)
        
        result_path = osp.join(self.exp_dir, 'result.json')
        print("Finish Testing, saving results to {}".format(result_path))
        json.dump({"Test Result":{'Distance (cm)':L2_dist,'Normalized distance (distance/scale*100%)':L2_dist_normalized}}, open(result_path,'w'))
        
    def __call__(self):
        # if not self.args.eval:
        #     self.train()
        self.test()
            