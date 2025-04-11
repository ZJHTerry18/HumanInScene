import torch
import torch.nn as nn
from models.utils import _init_weights

class ActivityHead_old(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, label_smoothing=0.001):
        super(ActivityHead_old, self).__init__()

        self.proj = nn.Linear(input_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.cls_head = nn.Linear(hidden_dim, num_classes)

        self.loss_func = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        self.apply(_init_weights)

    def forward(self, input, label):
        '''
            input: feature sequence, [B, N, C]
            label: ground-truth label, [B]
        '''
        if input.ndim == 2:
            input = input.unsqueeze(0)
        
        x = input.mean(dim=1)
        x = self.proj(x)
        x = self.gelu(x)
        logit = self.cls_head(x)
        loss = self.loss_func(logit, label)
        return logit, loss

class ActivityHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, label_smoothing=0.001, nearest_k=3):
        super(ActivityHead, self).__init__()

        self.proj = nn.Linear(input_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.cls_head = nn.Linear(hidden_dim, num_classes)
        self.nearest_k = nearest_k

        self.loss_func = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        self.apply(_init_weights)

    def forward(self, scene_feat, motion_feat, scene_loc, motion_loc, label):
        '''
            scene_feat: scene feature sequence, [B, No, d]
            motion_feat: motion feature sequence, [B, Nm, d]
            scene_loc: scene location (x,y), [B, No, 2]
            motion_loc: motion location (x,y), [B, Nm, 2]
            label: ground-truth label, [B]
        '''
        if scene_feat.ndim == 2:
            scene_feat = scene_feat.unsqueeze(0)
        if motion_feat.ndim == 2:
            motion_feat = motion_feat.unsqueeze(0)
        if scene_loc.ndim == 2:
            scene_loc = scene_loc.unsqueeze(0)
        if motion_loc.ndim == 2:
            motion_loc = motion_loc.unsqueeze(0)
        B, Nm, d = motion_feat.shape
        No = scene_feat.shape[1]
        
        # search nearest k objects for each motion timestamp
        motion_loc_exp = motion_loc.unsqueeze(2).repeat(1, 1, No, 1)  # [B, Nm, No, 2]
        scene_loc_exp = scene_loc.unsqueeze(1).repeat(1, Nm, 1, 1)  # [B, Nm, No, 2]
        
        distances = torch.norm(motion_loc_exp - scene_loc_exp, dim=3)
        _, nearest_indices = torch.topk(distances, self.nearest_k, dim=2, largest=False)
        
        indices_exp = nearest_indices.unsqueeze(-1).expand(-1, -1, -1, d)
        scene_feat_exp = scene_feat.unsqueeze(1).expand(-1, Nm, -1, -1)
        nearest_scene_feat = torch.gather(scene_feat_exp, 2, indices_exp).mean(dim=2)
        motion_fuse_feat = motion_feat + 0.3 * nearest_scene_feat
        
        # predict activity class
        x = motion_fuse_feat.mean(dim=1)
        x = self.proj(x)
        x = self.gelu(x)
        logit = self.cls_head(x)
        loss = self.loss_func(logit, label)
        return logit, loss

class ContactHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_joints, temp=1.0, pos_weight=100.0):
        super(ContactHead, self).__init__()

        self.scene_proj = nn.Linear(input_dim, hidden_dim)
        self.motion_proj = nn.Linear(input_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.joint_deconv = nn.Conv1d(in_channels=hidden_dim, out_channels=num_joints * hidden_dim, kernel_size=1)

        self.loss_func = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
        self.temp = temp

        self.apply(_init_weights)
    
    def forward(self, scene_feat, motion_feat, contact_label, contact_mask=None):
        '''
            scene_feat: scene features, [B, No, C]
            motion_feat: motion features, [B, Nm, C]
            contact_label: ground-truth binary label indicating contact, [B, Nm, J, No]
            contact_mask: binary mask indicating the region that are included when calculating loss, [B, Nm, J, No]
        '''
        if scene_feat.ndim == 2:
            scene_feat = scene_feat.unsqueeze(0)
        if motion_feat.ndim == 2:
            motion_feat = motion_feat.unsqueeze(0)
        if contact_label.ndim == 3:
            contact_label = contact_label.unsqueeze(0)
        if contact_mask.ndim == 3:
            contact_mask = contact_mask.unsqueeze(0)

        sf = self.scene_proj(scene_feat) # [B, No, d]
        mf = self.gelu(self.motion_proj(motion_feat))
        B, Nm, d = mf.size()
        mf = mf.view(B * Nm, d).unsqueeze(-1) # [B*Nm, d, 1]
        # decouple motion feature to per-joint embedding
        mjf = self.joint_deconv(mf).view(B, Nm, -1, d)

        sim_mat = torch.einsum('bmjd,bod->bmjo', mjf, sf) # [B, Nm, J, No]
        sim_mat = sim_mat / self.temp
        if contact_mask is not None:
            sim_mat = sim_mat.masked_fill(contact_mask == 0, 0.0)
            contact_label = contact_label.masked_fill(contact_mask == 0, 0.0)

        loss = self.loss_func(sim_mat, contact_label)
        return loss

class PositionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, temp=0.7):
        super(PositionHead, self).__init__()

        self.scene_proj = nn.Linear(input_dim, hidden_dim)
        self.motion_proj = nn.Linear(input_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.pos_deconv = nn.Conv1d(in_channels=hidden_dim, out_channels=num_classes * hidden_dim, kernel_size=1)
        self.cls_head = nn.Linear(hidden_dim, num_classes)
        self.num_classes = num_classes

        self.loss_func = nn.CrossEntropyLoss(reduction='none')
        self.temp = temp

        self.apply(_init_weights)
    
    def forward(self, scene_feat, motion_feat, pos_label, pos_weight=None):
        '''
            scene_feat: scene features, [B, No, C]
            motion_feat: motion features, [B, Nm, C]
            pos_label: ground-truth label indicating position relationship, [B, Nm, No]
            pos_weight: weight of each location when calculating loss, [B, Nm, No]
        '''
        if scene_feat.ndim == 2:
            scene_feat = scene_feat.unsqueeze(0)
        if motion_feat.ndim == 2:
            motion_feat = motion_feat.unsqueeze(0)
        if pos_label.ndim == 2:
            pos_label = pos_label.unsqueeze(0)
        if pos_weight.ndim == 2:
            pos_weight = pos_weight.unsqueeze(0)

        sf = self.scene_proj(scene_feat)
        mf = self.gelu(self.motion_proj(motion_feat))
        B, No, d = sf.size()
        B, Nm, d = mf.size()
        mf = mf.view(-1, d).unsqueeze(-1) # [B*Nm, d, 1]
        mpf = self.pos_deconv(mf).view(B, Nm, -1, d)

        pos_mat = torch.einsum('bmpd,bod->bmop', mpf, sf) # [B, Nm, No, p]
        pos_mat = pos_mat / self.temp
        pos_logits = pos_mat.reshape(-1, self.num_classes)

        # loss calculation
        pos_label = pos_label.flatten()
        loss_matrix = self.loss_func(pos_logits, pos_label)
        loss_matrix = loss_matrix.view(B, Nm, No)
        weighted_loss = loss_matrix * pos_weight
        loss = weighted_loss.sum() / pos_weight.sum()
        return loss