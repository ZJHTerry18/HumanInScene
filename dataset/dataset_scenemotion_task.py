import logging
import os
from os.path import join as pjoin
import json

import numpy as np
import torch
import torch.nn.functional as F

from common.constants import MOTION_TOKEN, ACTIVITY_LABELS, JOINT_INDICES, POS_LABELS
from dataset.base_dataset import BaseDataset, update_caption
import random

logger = logging.getLogger(__name__)



class SceneMotionTaskTrainDataset(BaseDataset):

    cached_feats = {}

    def __init__(self, config, dataset_name, task, ann_root, scene_files, motion_files, int_files, ann_folder, sample_ratio, **kwargs):
        super().__init__()
        self.feat_dim = config.model.scene.obj_input_dim
        self.img_feat_dim = config.model.scene.img_input_dim
        self.max_obj_num = config.model.scene.max_obj_num

        feat_file = pjoin(ann_root, scene_files["seg_feat_file"])
        img_feat_file = pjoin(ann_root, scene_files["seg_img_feat_file"])
        attribute_file = pjoin(ann_root, scene_files["seg_train_attr_file"])
        #print(feat_file)
        #print(img_feat_file)
        #print(attribute_file)
        #print(anno_file)
        self.attributes = torch.load(attribute_file, map_location='cpu') if attribute_file is not None else None

        self.sample_ratio = sample_ratio
        self.task = task
        self.dataset_name = dataset_name
        self.dataset_source, self.subtask = dataset_name.split('_')
        
        # load scene features
        if feat_file in SceneMotionTaskTrainDataset.cached_feats and img_feat_file in SceneMotionTaskTrainDataset.cached_feats:
            self.scene_feats, self.scene_masks = SceneMotionTaskTrainDataset.cached_feats[feat_file]
            self.scene_img_feats = SceneMotionTaskTrainDataset.cached_feats[img_feat_file]
        else:
            if feat_file is not None and os.path.exists(feat_file):
                #print(1)
                self.feats = torch.load(feat_file, map_location='cpu')
            else:
                self.feats = None
                #print(2)
            if img_feat_file is not None and os.path.exists(img_feat_file):
                self.img_feats = torch.load(img_feat_file, map_location='cpu')
            else:
                self.img_feats = None
            if self.attributes is None:
                self.scene_feats = self.feats
                self.scene_img_feats = self.scene_masks = self.scene_obj_labels = None
            else:
                self.scene_feats, self.scene_img_feats, self.scene_masks, self.scene_obj_labels = self.prepare_scene_features()
            SceneMotionTaskTrainDataset.cached_feats[feat_file] = (self.scene_feats, self.scene_masks)
            SceneMotionTaskTrainDataset.cached_feats[img_feat_file] = self.scene_img_feats

        # load motion features
        self.motion_tokens = dict()
        token_dir = pjoin(ann_root, self.dataset_source, motion_files["tokens"])
        token_files = os.listdir(token_dir)
        for token_f in token_files:
            token_dat = np.load(pjoin(token_dir, token_f))
            token_id = token_f[:-4]
            self.motion_tokens[token_id] = token_dat

        self.motion_trajs = dict()
        traj_dir = pjoin(ann_root, self.dataset_source, motion_files["trajs"])
        traj_files = os.listdir(traj_dir)
        for traj_f in traj_files:
            traj_dat = np.load(pjoin(traj_dir, traj_f))
            traj_id = traj_f[:-4]
            self.motion_trajs[traj_id] = traj_dat

        # load scene-motion interaction annotations
        ## activity
        self.activity_anns = dict()
        activity_file = pjoin(ann_root, self.dataset_source, int_files["activity"])
        with open(activity_file, "r") as f:
            activity_dat = json.load(f)
        for a_dat in activity_dat:
            data_id = f"{a_dat['scene']}_{a_dat['motion']}"
            if "id" in a_dat.keys():
                data_id = f"{a_dat['id']:06d}_{data_id}"
            raw_label = a_dat["label"].lower()
            activity_label = len(ACTIVITY_LABELS)
            for i, action_label in enumerate(ACTIVITY_LABELS):
                if action_label in raw_label:
                    activity_label = i
            self.activity_anns[data_id] = activity_label
        
        ## contact&position
        self.contact_anns = dict()
        self.position_anns = dict()
        hsi_dir = pjoin(ann_root, self.dataset_source, int_files["hsi"])
        hsi_files = os.listdir(hsi_dir)
        for data_id in hsi_files:
            with open(pjoin(hsi_dir, data_id, "keyframe_annotations.json"), "r") as f:
                hsi_dat = json.load(f)
            contact_dat = [v["contacts"] for k, v in hsi_dat.items()]
            position_dat = [v["pos"] for k, v in hsi_dat.items()]
            self.contact_anns[data_id] = contact_dat
            self.position_anns[data_id] = position_dat

        # load text data
        anno_file = pjoin(ann_root, self.dataset_source, ann_folder,
                        f'{self.dataset_source}_{self.subtask}_train.json')
        with open(anno_file, 'r') as f:
            self.anno = json.load(f)

        self.num_body_joints = len(JOINT_INDICES)

    def get_motion_anno(self, index):
        motion_id = self.anno[index]["motion_id"]
        data_id = self.anno[index]["data_id"]

        motion_token_ids = self.motion_tokens[motion_id]
        motion_token_strs = []
        for tok_id in motion_token_ids:
            tok_str = MOTION_TOKEN.format(tok_id)
            motion_token_strs.append(tok_str)
        motion_token = ''.join(motion_token_strs)
        motion_traj = self.motion_trajs[data_id]

        token_len = len(motion_token_ids)
        traj_len = len(motion_traj)

        # downsample trajectory to the same length as motion tokens
        ds_indices = np.linspace(0, traj_len - 1, num=token_len, dtype=int)
        motion_traj = motion_traj[ds_indices]
        motion_traj = torch.from_numpy(motion_traj)
        motion_traj = torch.nan_to_num(motion_traj, 0.0)

        return data_id, motion_token, motion_traj
    
    def get_scene_motion_act_anno(self, index):
        data_id = self.anno[index]["data_id"]
        # activity annotation
        activity_label = self.activity_anns[data_id]
        return torch.tensor([activity_label])

    def get_scene_motion_cont_anno(self, index, obj_labels, motion_len):
        data_id = self.anno[index]["data_id"]
        # contact annotation
        contact_rawdat = self.contact_anns[data_id] # list [Nf]
        raw_len = len(contact_rawdat)
        contact_raw_label = torch.zeros((raw_len, self.num_body_joints, self.max_obj_num))
        for i_frame, cont_info in enumerate(contact_rawdat):
            for joint, obj_id in cont_info:
                joint_id = JOINT_INDICES[joint]
                contact_raw_label[i_frame, joint_id, obj_id] = 1

        # randomly mask out floor contact
        contact_raw_mask = torch.ones_like(contact_raw_label)
        if random.random() < 0.8:
            floor_indices = [i for i, v in enumerate(obj_labels) if v == "floor"]
            contact_raw_mask[:, :, floor_indices] = 0

        # interpolate to the length of motion tokens
        contact_label = F.interpolate(contact_raw_label[None, None, :, :], 
                            size=(motion_len, *contact_raw_label.shape[1:]),
                            mode="nearest")
        contact_label = contact_label.squeeze(0).squeeze(0)
        contact_mask = F.interpolate(contact_raw_mask[None, None, :, :], 
                            size=(motion_len, *contact_raw_mask.shape[1:]),
                            mode="nearest")
        contact_mask = contact_mask.squeeze(0).squeeze(0)

        return contact_label, contact_mask # [L, Nj, No], [L, Nj, No]
    
    def get_scene_motion_pos_anno(self, index, obj_labels, motion_len):
        data_id = self.anno[index]["data_id"]
        # position annotation
        pos_rawdat = self.position_anns[data_id] # list [Nf]
        raw_len = len(pos_rawdat)
        num_pos_labels = len(POS_LABELS) + 1
        pos_raw_label = torch.zeros((raw_len, self.max_obj_num))
        pos_raw_weight = torch.zeros_like(pos_raw_label)
        for i_frame, pos_infos in enumerate(pos_rawdat):
            for pos_info in pos_infos:
                dist, pos, _ = pos_info[:3]
                pos_id = POS_LABELS[pos] if pos in POS_LABELS else num_pos_labels - 1
                w = 0.5 if dist == "far" else 1.0
                for obj_id in pos_info[3:]:
                    if obj_id < self.max_obj_num:
                        pos_raw_label[i_frame, obj_id] = pos_id
                        pos_raw_weight[i_frame, obj_id] = w
        
        # randomly discard the prediction of floor position (at index 0)
        if random.random() < 0.8:
            pos_raw_weight[:, 0] = 0.0
        
        # interpolate to the length of motion tokens
        pos_label = F.interpolate(pos_raw_label[None, None, :, :], 
                        size=(motion_len, *pos_raw_label.shape[1:]),
                        mode="nearest")
        pos_label = pos_label.squeeze(0).squeeze(0).to(torch.long)
        pos_weight = F.interpolate(pos_raw_weight[None, None, :, :], 
                            size=(motion_len, *pos_raw_weight.shape[1:]),
                            mode="nearest")
        pos_weight = pos_weight.squeeze(0).squeeze(0)

        return pos_label, pos_weight # [L, No, P], [L, No, P]

    def __len__(self):
        return len(self.anno)

    def __getitem__(self, index):
        if self.attributes is not None and self.anno[index]['scene_id'] not in self.attributes:
            # print(f"{self.anno[index]['scene_id']} not in attribute file!")
            return self.__getitem__(random.randint(0, len(self.anno)-1))
        if "obj_id" in self.anno[index]:
            obj_id = int(self.anno[index]["obj_id"])
        else:
            obj_id = random.randint(0, self.max_obj_num - 1)
        
        # question&answer annotation
        # TODO: implementation of dialogue
        qas = self.anno[index]["qa"]
        task = self.anno[index]["task"]
        question = qas[0]["question"]
        answer = qas[0]["answer"]
        scene_id, scene_feat, scene_img_feat, scene_mask, scene_locs, scene_obj_labels, assigned_ids = self.get_scene_anno(index)
        obj_num = torch.tensor([len(assigned_ids)])
        scene_motion_id, motion_token, motion_traj = self.get_motion_anno(index)
        activity_label = self.get_scene_motion_act_anno(index)
        contact_label, contact_mask = self.get_scene_motion_cont_anno(index, scene_obj_labels, len(motion_token))
        position_label, position_weight = self.get_scene_motion_pos_anno(index, scene_obj_labels, len(motion_token))

        return scene_feat, scene_img_feat, scene_mask, scene_locs, scene_obj_labels, \
            obj_num, assigned_ids, motion_token, motion_traj, \
            activity_label, contact_label, contact_mask, position_label, position_weight, \
            answer, question