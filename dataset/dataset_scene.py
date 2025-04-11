import logging
import os
from os.path import join as pjoin
import json

import numpy as np
import torch

from dataset.base_dataset import BaseDataset, update_caption
import glob
import random
from prompts.prompts import obj_caption_wid_prompt
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)



class SceneTrainDataset(BaseDataset):

    cached_feats = {}

    def __init__(self, config, dataset_name, task, ann_root, files, anns, sample_ratio, **kwargs):
        super().__init__()
        self.feat_dim = config.model.scene.obj_input_dim
        self.img_feat_dim = config.model.scene.img_input_dim
        self.max_obj_num = config.model.scene.max_obj_num

        feat_file = pjoin(ann_root, files["seg_feat_file"])
        img_feat_file = pjoin(ann_root, files["seg_img_feat_file"])
        attribute_file = pjoin(ann_root, files["seg_train_attr_file"])
        self.attributes = torch.load(attribute_file, map_location='cpu') if attribute_file is not None else None
        self.anno = []
        for ann in anns:
            anno_file = pjoin(ann_root, dataset_name, ann)
            with open(anno_file, 'r') as f:
                self.anno.extend(json.load(f))

        self.sample_ratio = sample_ratio
        self.task = task
        self.dataset_name = dataset_name
        
        if feat_file in SceneTrainDataset.cached_feats and img_feat_file in SceneTrainDataset.cached_feats:
            self.scene_feats, self.scene_masks = SceneTrainDataset.cached_feats[feat_file]
            self.scene_img_feats = SceneTrainDataset.cached_feats[img_feat_file]
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
            SceneTrainDataset.cached_feats[feat_file] = (self.scene_feats, self.scene_masks)
            SceneTrainDataset.cached_feats[img_feat_file] = self.scene_img_feats


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
        if 'prompt' not in self.anno[index]:
            question = random.choice(obj_caption_wid_prompt).replace('<id>', f"<OBJ{obj_id:03}>")
        else:
            question = self.anno[index]["prompt"]
        caption = self.anno[index]["caption"]
        scene_id, scene_feat, scene_img_feat, scene_mask, scene_locs, _, assigned_ids = self.get_scene_anno(index)
        obj_num = torch.tensor([len(assigned_ids)])
        caption = update_caption(caption, assigned_ids)
        question = update_caption(question, assigned_ids)
        return scene_feat, scene_img_feat, scene_mask, scene_locs, obj_num, assigned_ids, caption, question

class SceneValDataset(BaseDataset):

    cached_feats = {}

    def __init__(self, config, dataset_name, task, ann_root, files, anns, **kwargs):
        super().__init__()
        self.dataset_name = dataset_name
        self.feat_dim = config.model.scene.obj_input_dim
        self.img_feat_dim = config.model.scene.img_input_dim
        self.max_obj_num = config.model.scene.max_obj_num

        feat_file = pjoin(ann_root, files["seg_feat_file"])
        img_feat_file = pjoin(ann_root, files["seg_img_feat_file"])
        attribute_file = pjoin(ann_root, files["seg_train_attr_file"])
        self.attributes = torch.load(attribute_file, map_location='cpu') if attribute_file is not None else None
        
        anno_file = pjoin(ann_root, dataset_name, anns[0])
        with open(anno_file, "r") as f:
            self.anno = json.load(f)

        if feat_file in SceneValDataset.cached_feats and img_feat_file in SceneValDataset.cached_feats:
            self.scene_feats, self.scene_masks = SceneValDataset.cached_feats[feat_file]
            self.scene_img_feats = SceneValDataset.cached_feats[img_feat_file]
        else:
            if feat_file is not None and os.path.exists(feat_file):
                self.feats = torch.load(feat_file, map_location='cpu')
            else:
                self.feats = None
            if img_feat_file is not None and os.path.exists(img_feat_file):
                self.img_feats = torch.load(img_feat_file, map_location='cpu')
            else:
                self.img_feats = None
            if self.attributes is None:
                self.scene_feats = self.feats
                self.scene_img_feats = self.scene_masks = None
            else:
                self.scene_feats, self.scene_img_feats, self.scene_masks, self.scene_obj_labels = self.prepare_scene_features()
            SceneValDataset.cached_feats[feat_file] = (self.scene_feats, self.scene_masks)
            SceneValDataset.cached_feats[img_feat_file] = self.scene_img_feats

    def __len__(self):
        return len(self.anno)

    def __getitem__(self, index):
        scene_id, scene_feat, scene_img_feat, scene_mask, scene_locs, _, assigned_ids = self.get_scene_anno(index)
        obj_num = torch.tensor([len(assigned_ids)])
        obj_id = int(self.anno[index].get('obj_id', 0))
        pred_id = int(self.anno[index].get('pred_id', 0))
        type_info = int(self.anno[index].get('sqa_type', 0))
        if 'sqa_type' in self.anno[index]:
            type_info = self.anno[index]['sqa_type']
        elif 'eval_type' in self.anno[index]:
            type_info = self.anno[index]['eval_type'] 
        elif 'type_info' in self.anno[index]:
            type_info = self.anno[index]['type_info']
        if 'prompt' not in self.anno[index]:
            question = random.choice(obj_caption_wid_prompt).replace('<id>', f"<OBJ{obj_id:03}>")
        else:
            question = self.anno[index]["prompt"]
        ref_answers = self.anno[index]["ref_captions"].copy() if "ref_captions" in self.anno[index] else [self.anno[index]["caption"]]
        qid = self.anno[index]["qid"] if "qid" in self.anno[index] else 0
        return scene_feat, scene_img_feat, scene_mask, scene_locs, obj_num, assigned_ids, question, ref_answers, scene_id, qid, pred_id, type_info