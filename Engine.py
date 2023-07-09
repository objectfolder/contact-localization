import os
import os.path as osp
import sys
import json
from pprint import pprint

from tqdm import tqdm, trange
import numpy as np
import torch
import torch.optim as optim

from utils.meters import AverageMeter
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
        self.train_loader, self.val_loader, self.test_loader = build_dataset(
            self.args)
        # build model & optimizer
        self.model, self.optimizer = build_model(self.args, self.cfg)
        self.model.cuda()
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=5, gamma=0.8)
        # experiment dir
        self.exp_dir = osp.join('./exp_new', self.args.exp)
        os.makedirs(self.exp_dir, exist_ok=True)
        self.exp_viz_dir = osp.join(self.exp_dir,'viz')
        os.makedirs(self.exp_viz_dir, exist_ok=True)

    def train_epoch(self, epoch):
        self.model.train()
        epoch_loss = {}
        for i, batch in tqdm(enumerate(self.train_loader), leave=False):
            self.optimizer.zero_grad()
            output = self.model(batch, calc_loss=True)
            loss = output['loss']
            loss.backward()
            self.optimizer.step()
            for k,v in output.items():
                if 'loss' in k:
                    if not k in epoch_loss:
                        epoch_loss[k]=AverageMeter()
                    epoch_loss[k].update(v.item(), output['pred'].shape[0])
            if i % 20 == 0:
                message = 'Train Epoch: {}'.format(epoch)
                for k,v in epoch_loss.items():
                    message+=', {}: {:.4f}'.format(k,v.avg)
                tqdm.write(message)

    @torch.no_grad()
    def eval_epoch(self, epoch=0, test=False):
        self.model.eval()
        data_loader = self.test_loader if test else self.val_loader
        epoch_loss = {}
        eval_results = {}
        eval_results_normalized = {}
        for i, batch in tqdm(enumerate(data_loader), leave=False):
            output = self.model(batch, calc_loss=True)
            pred = output['pred']
            target = batch['contact_location'].cuda()
            for k,v in output.items():
                if 'loss' in k:
                    if not k in epoch_loss:
                        epoch_loss[k] = AverageMeter()
                    epoch_loss[k].update(v.item(), output['pred'].shape[0])
            for j, name in enumerate(batch['names']):
                obj = name[0]
                if not obj in eval_results:
                    eval_results[obj] = AverageMeter()
                    eval_results_normalized[obj] = AverageMeter()
                distance = float(torch.dist(pred[j]*batch['point_cloud_scale'][j].cuda(), 
                                            target[j]*batch['point_cloud_scale'][j].cuda()))
                eval_results[obj].update(distance)
                eval_results_normalized[obj].update(distance/batch['mesh_scale'][j])
            viz_data = {
                'names':batch['names'],
                'target':target.detach().cpu().numpy(),
                'pred':pred.detach().cpu().numpy(),
                'point_cloud':batch['point_cloud'].detach().cpu().numpy(),
            }
            np.save(osp.join(self.exp_viz_dir,f'viz_data_{i}.npy'),viz_data,allow_pickle=True)
                
        eval_results = {k: float(v.avg)*100 for k, v in eval_results.items()}
        eval_results_normalized = {k: float(v.avg)*100 for k, v in eval_results_normalized.items()}
        # pprint(eval_results)
        pprint(eval_results_normalized)
        message = 'Eval Epoch: {}, normalized_dist_mean: {:.2f} (%)'.format(epoch, np.mean(list(eval_results_normalized.values())))
        for k,v in epoch_loss.items():
            message+=', {}: {:.4f}'.format(k,v.avg)
        tqdm.write(message)
        return eval_results_normalized, epoch_loss['loss'].avg

    def train(self):
        bst = 1e8
        for epoch in range(self.args.epochs):
            print("Start Validation Epoch {}".format(epoch))
            eval_results, eval_loss = self.eval_epoch(epoch)
            eval_results = np.mean(list(eval_results.values()))
            if eval_results < bst:
                bst = eval_results
                save_dir = osp.join(self.exp_dir, 'bst.pth')
                print(f"saving the best model to {save_dir}")
                state_dict = self.model.state_dict()
                torch.save(state_dict, save_dir)
            save_dir = osp.join(self.exp_dir, 'latest.pth')
            print(f"saving the latest model to {save_dir}")
            state_dict = self.model.state_dict()
            torch.save(state_dict, save_dir)
            print("Start Training Epoch {}".format(epoch))
            self.train_epoch(epoch)
            self.scheduler.step()

    def test(self):
        print("Start Testing")
        print("Loading best model from {}".format(osp.join(self.exp_dir,'bst.pth')))
        self.model.load_state_dict(torch.load(osp.join(self.exp_dir, 'bst.pth')))
        test_results_normalized, test_loss = self.eval_epoch(test=True)
        result_dir = osp.join(self.exp_dir, 'result.json')
        json.dump(test_results_normalized, open(result_dir,'w'))
        print("Finish Testing, test_loss = {:.4f}, results saved to {}".format(test_loss, result_dir))
        
    def __call__(self):
        if not self.args.eval:
            self.train()
        self.test()
