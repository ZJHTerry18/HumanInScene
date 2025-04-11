import logging
import os
from os.path import join as pjoin
import json

import numpy as np
import torch

from common.constants import MOTION_TOKEN
from dataset.base_dataset import BaseDataset, update_caption
import random

logger = logging.getLogger(__name__)



class SceneMotionTrainDataset(BaseDataset):

    cached_feats = {}

    def __init__(self, config, dataset_name, task, ann_root, scene_files, motion_files, ann_folder, sample_ratio, **kwargs):
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
        
        # load scene features
        if feat_file in SceneMotionTrainDataset.cached_feats and img_feat_file in SceneMotionTrainDataset.cached_feats:
            self.scene_feats, self.scene_masks = SceneMotionTrainDataset.cached_feats[feat_file]
            self.scene_img_feats = SceneMotionTrainDataset.cached_feats[img_feat_file]
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
                self.scene_img_feats = self.scene_masks = None
            else:
                self.scene_feats, self.scene_img_feats, self.scene_masks, self.scene_obj_labels = self.prepare_scene_features()
            SceneMotionTrainDataset.cached_feats[feat_file] = (self.scene_feats, self.scene_masks)
            SceneMotionTrainDataset.cached_feats[img_feat_file] = self.scene_img_feats

        # load motion features
        self.motion_tokens = dict()
        token_dir = pjoin(ann_root, dataset_name, motion_files["tokens"])
        token_files = os.listdir(token_dir)
        for token_f in token_files:
            token_dat = np.load(pjoin(token_dir, token_f))
            token_id = token_f[:-4]
            self.motion_tokens[token_id] = token_dat

        self.motion_trajs = dict()
        traj_dir = pjoin(ann_root, dataset_name, motion_files["trajs"])
        traj_files = os.listdir(traj_dir)
        for traj_f in traj_files:
            traj_dat = np.load(pjoin(traj_dir, traj_f))
            traj_id = traj_f[:-4]
            self.motion_trajs[traj_id] = traj_dat

        # load text data
        self.anno = []
        ann_dir = pjoin(ann_root, dataset_name, ann_folder)
        ann_files = os.listdir(ann_dir)
        for ann in ann_files:
            anno_file = pjoin(ann_dir, ann)
            # if 'open' in ann:
            with open(anno_file, 'r') as f:
                self.anno.extend(json.load(f))

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

        return data_id, motion_token, motion_traj

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
        # question = ""
        answer = qas[0]["answer"]
        scene_id, scene_feat, scene_img_feat, scene_mask, scene_locs, assigned_ids = self.get_scene_anno(index)
        scene_motion_id, motion_token, motion_traj = self.get_motion_anno(index)
        answer = update_caption(answer, assigned_ids)
        question = update_caption(question, assigned_ids)

        return scene_feat, scene_img_feat, scene_mask, scene_locs, obj_id, assigned_ids, motion_token, motion_traj, answer, question

class SceneMotionValDataset(BaseDataset):

    cached_feats = {}

    def __init__(self, config, dataset_name, task, ann_root, scene_files, motion_files, ann_folder, **kwargs):
        super().__init__()
        self.feat_dim = config.model.scene.obj_input_dim
        self.img_feat_dim = config.model.scene.img_input_dim
        self.max_obj_num = config.model.scene.max_obj_num

        feat_file = pjoin(ann_root, scene_files["seg_feat_file"])
        img_feat_file = pjoin(ann_root, scene_files["seg_img_feat_file"])
        attribute_file = pjoin(ann_root, scene_files["seg_train_attr_file"])
        self.attributes = torch.load(attribute_file, map_location='cpu') if attribute_file is not None else None

        self.task = task
        self.dataset_name = dataset_name
        
        # load scene features
        if feat_file in SceneMotionValDataset.cached_feats and img_feat_file in SceneMotionValDataset.cached_feats:
            self.scene_feats, self.scene_masks = SceneMotionValDataset.cached_feats[feat_file]
            self.scene_img_feats = SceneMotionValDataset.cached_feats[img_feat_file]
        else:
            if feat_file is not None and os.path.exists(feat_file):
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
                self.scene_img_feats = self.scene_masks = None
            else:
                self.scene_feats, self.scene_img_feats, self.scene_masks, self.scene_obj_labels = self.prepare_scene_features()
            SceneMotionValDataset.cached_feats[feat_file] = (self.scene_feats, self.scene_masks)
            SceneMotionValDataset.cached_feats[img_feat_file] = self.scene_img_feats

        # load motion features
        self.motion_tokens = dict()
        token_dir = pjoin(ann_root, dataset_name, motion_files["tokens"])
        token_files = os.listdir(token_dir)
        for token_f in token_files:
            token_dat = np.load(pjoin(token_dir, token_f))
            token_id = token_f[:-4]
            self.motion_tokens[token_id] = token_dat

        self.motion_trajs = dict()
        traj_dir = pjoin(ann_root, dataset_name, motion_files["trajs"])
        traj_files = os.listdir(traj_dir)
        for traj_f in traj_files:
            traj_dat = np.load(pjoin(traj_dir, traj_f))
            traj_id = traj_f[:-4]
            self.motion_trajs[traj_id] = traj_dat

        # load text data
        self.anno = []
        ann_dir = pjoin(ann_root, dataset_name, ann_folder)
        ann_files = os.listdir(ann_dir)
        for ann in ann_files:
            anno_file = pjoin(ann_dir, ann)
            with open(anno_file, 'r') as f:
                self.anno.extend(json.load(f))
        
        # print("feats", self.feats.keys())
        # print("scene feats", self.scene_feats.keys())
        # print("attributes", self.attributes.keys())

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
        motion_traj = torch.from_numpy(motion_traj).to(torch.float32)
        motion_traj = torch.nan_to_num(motion_traj, 0.0)

        return data_id, motion_token, motion_traj

    def __len__(self):
        return len(self.anno)

    def __getitem__(self, index):
        scene_id, scene_feat, scene_img_feat, scene_mask, scene_locs, _, assigned_ids = self.get_scene_anno(index)
        obj_num = torch.tensor([len(assigned_ids)])
        scene_motion_id, motion_token, motion_traj = self.get_motion_anno(index)
        obj_id = int(self.anno[index].get('obj_id', 0))
        pred_id = int(self.anno[index].get('pred_id', 0))
        type_info = self.anno[index].get('task', "")
        
        # question&answer annotation
        # TODO: implementation of dialogue
        qas = self.anno[index]["qa"]
        task = self.anno[index]["task"]
        question = qas[0]["question"]
        # question = ""
        ref_answers = [qas[0]["answer"]]

        return scene_feat, scene_img_feat, scene_mask, scene_locs, obj_num, assigned_ids, \
            motion_token, motion_traj, question, ref_answers, scene_id, scene_motion_id, index, pred_id, type_info