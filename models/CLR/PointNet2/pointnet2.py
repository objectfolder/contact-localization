import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from .pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction

class Pointnet2_msg_backbone(nn.Module):
    def __init__(self, normal_channel=False):
        super(Pointnet2_msg_backbone, self).__init__()
        in_channel = 3 if normal_channel else 0
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel, [
                                             [32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320, [
                                             [64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(
            None, None, None, 640 + 3, [256, 512, 1024], True)
        
    def forward(self, point_cloud):
        xyz = point_cloud.permute(0,2,1).cuda()
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        feature = l3_points.view(B, 1024)
        return feature


class Pointnet2_msg(nn.Module):
    def __init__(self, num_class, normal_channel=False):
        super(Pointnet2_msg, self).__init__()
        self.backbone = Pointnet2_msg_backbone(normal_channel=normal_channel)
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_class)
        )
        self.loss = get_loss()

    def forward(self, batch, calc_loss=True):
        output = {}
        feature = self.backbone(batch['point_cloud'])
        pred = self.classifier(feature)
        
        output['feature'] = feature
        output['pred'] = pred
        if calc_loss:
            target = batch['label'].cuda()
            output['loss'] = self.loss(pred, target)

        return output


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target):
        total_loss = F.cross_entropy(pred, target)

        return total_loss