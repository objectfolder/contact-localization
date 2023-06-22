import os, sys
import os.path as osp
os.environ['PYOPENGL_PLATFORM'] = 'egl'
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('/viscam/u/yimingdou/ObjectFolder-Benchmark/benchmarks/Contact_Localization/code/models/OF_CNN_MFCC/ObjectFolder')
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms as T

from mesh_to_sdf import *
from sklearn.neighbors import KDTree
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist

from project_object_folder.renderer.DepthRender import TacRender
from project_object_folder.renderer.ObjectLoader import ObjectLoader
from project_object_folder.renderer.utils import *

from ObjectFolder.TouchNet_model import *
from ObjectFolder.TouchNet_utils import *
from ObjectFolder.utils import *

press_depth = 0.0015 # in meter
shear_range = 0.0
z_axis = np.array([0,0,1])
theta = 0.0
phi = np.pi
phi_x = np.cos(phi)
phi_y = np.sin(phi)
# model_path = '/home/zsi/project_object_folder/data/TouchNet_models/23/model.pt'
rotation_max = 15
displacement_min = 0.0005
displacement_max = 0.0020
depth_max = 0.04
depth_min = 0.0339
rgb_width = 120
rgb_height = 160
network_depth = 8
    
class TouchBackbone(nn.Module):
    def __init__(self, pretrained=False):
        super(TouchBackbone, self).__init__()
        original_resnet = models.resnet18(pretrained)
        layers = list(original_resnet.children())[0:-1]
        self.feature_extraction = nn.Sequential(*layers) 
    def forward(self, x):
        x = self.feature_extraction(x)
        x = x.view(x.shape[0], -1)
        return x

class OF_CNN_MFCC(nn.Module):
    def __init__(self, args, use_touch=True, use_audio=True):
        super(OF_CNN_MFCC, self).__init__()
        self.args = args
        self.use_touch = use_touch
        self.use_audio = use_audio
        self.resize_dict = {
            "touch": T.Resize((224, 224)),
        }
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
        # OF-Touch
        self.gel_path = './models/OF_CNN_MFCC/project_object_folder/calibs/gel_surface_mm.obj'
        self.calib_path = './models/OF_CNN_MFCC/project_object_folder/calibs'
        self.embed_fn, input_ch = get_embedder(10, 0)
        self.of_touch_net = NeRF(D = 8, input_ch = input_ch, output_ch = 1)
        self.touch_feature_extractor = TouchBackbone()
        touch_ckpt = torch.load(f'./exp_new/pretrain/touch_backbone.pth')
        self.touch_feature_extractor.load_state_dict(touch_ckpt)
        
    def find_closest_points(self, query_points, kd_tree):
        distances, indices = kd_tree.query(query_points)
        return indices
    
    def gen_input(self, batch_size):
        #initialize horizontal and vertical features
        w_feats = np.repeat(np.repeat(np.arange(rgb_width).reshape((rgb_width, 1)), rgb_height, axis=1).reshape((1, 1, rgb_width, rgb_height)), batch_size, axis=0)
        h_feats = np.repeat(np.repeat(np.arange(rgb_height).reshape((1, rgb_height)), rgb_width, axis=0).reshape((1, 1, rgb_width, rgb_height)), batch_size, axis=0)
        #normalize horizontal and vertical features to [-1, 1]
        w_feats_min = w_feats.min()
        w_feats_max = w_feats.max()
        h_feats_min = h_feats.min()
        h_feats_max = h_feats.max()
        w_feats = (w_feats - w_feats_min) / (w_feats_max - w_feats_min)
        h_feats = (h_feats - h_feats_min) / (h_feats_max - h_feats_min)

        w_feats = torch.FloatTensor(w_feats).reshape((batch_size, -1, 1))
        h_feats = torch.FloatTensor(h_feats).reshape((batch_size, -1, 1))

        displacement_batch = np.full((batch_size), press_depth) # N
        theta_batch = np.full((batch_size), theta)
        phi_x_batch = np.full((batch_size), phi_x)
        phi_y_batch = np.full((batch_size), phi_y)
        # normalize values
        theta_batch = (theta_batch - np.radians(0)) / (np.radians(rotation_max) - np.radians(0))
        displacement_batch = (displacement_batch - displacement_min) / (displacement_max - displacement_min)

        theta_batch = torch.FloatTensor(np.repeat(theta_batch.reshape((batch_size, 1, 1)), rgb_width * rgb_height, axis=1))
        phi_x_batch = torch.FloatTensor(np.repeat(phi_x_batch.reshape((batch_size, 1, 1)), rgb_width * rgb_height, axis=1))
        phi_y_batch = torch.FloatTensor(np.repeat(phi_y_batch.reshape((batch_size, 1, 1)), rgb_width * rgb_height, axis=1))
        displacement_batch = torch.FloatTensor(np.repeat(displacement_batch.reshape((batch_size, 1, 1)), rgb_width * rgb_height, axis=1))
        return theta_batch, phi_x_batch, phi_y_batch, displacement_batch, w_feats, h_feats
        
    def forward(self, batch, calc_loss = False):
        output = {}
        output['pred_contact_point'] = []
        output['gt_contact_point'] = []
        for i in range(batch['trajectory'].shape[0]): # batch
            if i==0 or batch['names'][i][0] != batch['names'][i-1][0]: # re-load if different object
                print("Reloading mesh...")
                mesh = batch['mesh'][i]
                point_cloud = get_surface_point_cloud(mesh, scan_count=10)
                kd_tree = KDTree(mesh.vertices)
                obj_path = '../DATA/OF_all/{}/model.obj'.format(batch['names'][i][0])
                tacRender = TacRender(obj_path, self.gel_path, self.calib_path)
                if 'touch' in self.args.modality_list:
                    print("Reloading touchnet...")
                    model_path = '../DATA/OF_all/{}/ObjectFile.pth'.format(batch['names'][i][0])
                    checkpoint = torch.load(model_path)
                    vertex_min = checkpoint['TouchNet']['xyz_min']
                    vertex_max = checkpoint['TouchNet']['xyz_max']
                    state_dic = checkpoint['TouchNet']["model_state_dict"]
                    state_dic = strip_prefix_if_present(state_dic, 'module.')
                    self.of_touch_net.load_state_dict(state_dic)
            
            # loop -> convergence
            num_iteration = self.args.trajectory_length
            num_points = 512
            points = point_cloud.get_random_surface_points(num_points)
            np.save(osp.join('./tmp', 'orig_points.npy'), points)
            sdf, normals = point_cloud.get_sdf_in_batches(
                points, return_gradients=True)
            points = points + -1 * normals * sdf[:, np.newaxis]
            
            contact_points = batch['trajectory'][i].detach().cpu().numpy()
            if 'touch' in self.args.modality_list:
                # get reference touch feature
                ref_touch = self.resize_dict['touch'](batch['touch'][i].cuda())
                ref_touch_feature = self.touch_feature_extractor(ref_touch) # (traj_len, 4096)
            
            if 'audio' in self.args.modality_list:
                audio_features = batch['audio'][i]
                sdf, normals = point_cloud.get_sdf_in_batches(contact_points, return_gradients=True)
                ref_points = contact_points + -1*normals * sdf[:,np.newaxis]
                ref_idx_on_mesh = self.find_closest_points(ref_points, kd_tree)
                ref_audio_features = audio_features[ref_idx_on_mesh,:].squeeze()
                idx_on_mesh = self.find_closest_points(points, kd_tree)
            
            tacRender.start_offline()
            for iter in range(num_iteration):
                batch_size = 128
                valid_num_particles = points.shape[0]
                print("Iteration {}, total valid particles: {}".format(iter, valid_num_particles))
                sim_records = []
                if 'audio' in self.args.modality_list:
                    particle_audio_features = audio_features[idx_on_mesh,:].squeeze() # N * 194
                    sim_audio = cosine_similarity(ref_audio_features[iter].reshape(1, -1), particle_audio_features).squeeze()
                    sim_records.append(sim_audio)
                if 'touch' in self.args.modality_list:
                    sim_tactile = np.zeros((valid_num_particles))
                    count = 0
                    theta_batch, phi_x_batch, phi_y_batch, displacement_batch, w_feats, h_feats = self.gen_input(batch_size)
                    while (count + batch_size) < valid_num_particles:
                        # fill in values
                        vertex_batch = points[count:count+batch_size,:] # N * 3
                        vertex_batch = (vertex_batch - vertex_min) / (vertex_max - vertex_min)
                        vertex_batch = torch.FloatTensor(np.repeat(vertex_batch.reshape((batch_size, 1, 3)), rgb_width * rgb_height, axis=1))
                        feats = torch.cat((vertex_batch, theta_batch, phi_x_batch, phi_y_batch, displacement_batch, w_feats, h_feats), dim=2).reshape((-1, 9))
                        feats = feats.cuda()
                        embedded = self.embed_fn(feats)
                        preds = self.of_touch_net(embedded).detach().cpu().numpy().reshape((batch_size, rgb_width, rgb_height))
                        preds = preds  * (depth_max - depth_min) + depth_min
                        batch_data = []
                        for _i in range(batch_size):
                            depth = preds[_i,:,:]
                            height_map, contact_mask, tactile_img = tacRender.taxim_render(depth, press_depth)
                            img = Image.fromarray(tactile_img, mode="RGB")
                            img = self.preprocess['touch'](img)
                            batch_data.append(img)
                        batch_data = torch.stack(batch_data).cuda()
                        features = self.touch_feature_extractor(batch_data) # (bs, 4096)
                        sim = cosine_similarity(ref_touch_feature[iter].reshape(1, -1).detach().cpu().numpy(), features.detach().cpu().numpy())
                        sim_tactile[count:count+batch_size] = sim.squeeze()
                        count += batch_size
                        if (count + batch_size) > valid_num_particles:
                            batch_size = valid_num_particles - count
                    sim_records.append(sim_tactile)
                
                sim_records = np.mean(np.array(sim_records), 0)
                top_k = len(sim_records) if len(sim_records) < 10 else 10
                sort_sim = np.argsort(sim_records)
                idx_highest = sort_sim[-top_k:]
                idx_rest = sort_sim[:-top_k]
                print("Highest similarity: {}".format(sim_records[idx_highest]))
                print("top sim points:")
                top_points = points[idx_highest,:]
                import ipdb
                print(points[idx_highest,:])
                print("ref point: ")
                print(contact_points[iter,:])
                ref_point = np.array(contact_points[iter,:])
                sim_points = np.array(top_points)
                diff = sim_points - ref_point
                diff_scale = np.sqrt(diff[:,0]**2 + diff[:,1]**2 + diff[:,2]**2)
                diff_scale_mm = diff_scale * 100
                print("Distance: {}".format(diff_scale_mm))
                print("Distance min: {:.2f}, mean: {:.2f}, median: {:.2f} (cm)".format(np.min(diff_scale_mm),np.mean(diff_scale_mm),np.median(diff_scale_mm)))
                
                print("converge distance: {:.6f}".format(np.max(cdist(top_points,top_points))/batch['mesh_scale'][i]))
                if iter == num_iteration - 1:
                    output['pred_contact_point'].append(torch.tensor(sim_points[-1]))
                    output['gt_contact_point'].append(torch.tensor(ref_point))
                    break
                elif np.max(cdist(top_points,top_points))/batch['mesh_scale'][i]<1e-3 \
                    and self.args.convergence_eval: # converge
                    print("Converge, Early Stop...")
                    output['pred_contact_point'].append(torch.tensor(sim_points[-1]))
                    output['gt_contact_point'].append(torch.tensor(ref_point))
                    break
                relative_transform = contact_points[iter+1,:]-contact_points[iter,:]
                # print("relative motion is ", relative_transform)

                # update particles
                ## update particle's location
                updated_points = points + relative_transform
                sdf, normals = point_cloud.get_sdf_in_batches(updated_points, return_gradients=True)
                # print("updated points' sdf range: ")
                # print(np.max(sdf))
                # print(np.min(sdf))
                
                # surface_point_mask = np.abs(sdf) < max(1e-2, np.min(np.abs(sdf))+1e-3)
                surface_point_mask = np.abs(sdf) < max(np.min(np.abs(sdf))*2,1e-2)
                num_valid_point = np.sum(1*surface_point_mask)
                surface_points = updated_points[surface_point_mask,:]
                surface_sim = sim_records[surface_point_mask]
                surface_sdf = sdf[surface_point_mask]
                surface_normals = normals[surface_point_mask,:]
                print("surface points: ", num_valid_point)
                
                ## resample
                surface_sim = np.abs(surface_sim)
                prob = surface_sim/np.sum(surface_sim)
                print("range of the probability: ({:.4f}, {:.4f})".format(np.min(prob), np.max(prob)))
                new_point_idx = np.random.choice(num_valid_point, num_points, p = prob)
                new_points = surface_points[new_point_idx,:]
                # add guassian noise
                new_points = new_points + np.random.normal(scale=2e-3, size=(num_points, 3))
                new_sdf, new_normals = point_cloud.get_sdf_in_batches(new_points, return_gradients=True)
                # print("new sampled points' sdf range: ")
                # print(np.max(new_sdf))
                # print(np.min(new_sdf))

                points = new_points
                sdf = new_sdf
                normals = new_normals
                points = points + -1*normals*sdf[:,np.newaxis]
                idx_on_mesh = self.find_closest_points(points, kd_tree)
        output['pred_contact_point']=torch.stack(output['pred_contact_point'])
        output['gt_contact_point']=torch.stack(output['gt_contact_point'])
        return output