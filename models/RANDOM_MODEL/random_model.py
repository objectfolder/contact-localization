import numpy as np
import torch

class RANDOM_MODEL():
    def __init__(self) -> None:
        pass
    def __call__(self, batch):
        output={}
        output['pred_contact_point']=[]
        for i in range(batch['trajectory'].shape[0]):
            cur_vertices=batch['mesh'][i].vertices
            rand_idx=np.random.randint(cur_vertices.shape[0])
            output['pred_contact_point'].append(torch.tensor(cur_vertices[rand_idx,:]))
            
        output['pred_contact_point']=torch.stack(output['pred_contact_point'])
        output['gt_contact_point']=batch['trajectory'][:,-1,:]
        return output