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



class MotionTrainDataset(BaseDataset):

    cached_feats = {}

    def __init__(self, config, dataset_name, task, ann_root, motion_files, sample_ratio, **kwargs):
        super().__init__()
        self.sample_ratio = sample_ratio
        self.task = task
        self.dataset_name = dataset_name

        # load motion features
        self.motion_tokens = dict()
        token_dir = pjoin(ann_root, self.dataset_name, motion_files["tokens"])
        token_files = os.listdir(token_dir)
        for token_f in token_files:
            token_dat = np.load(pjoin(token_dir, token_f))
            token_id = token_f[:-4]
            self.motion_tokens[token_id] = token_dat

        # load text data
        anno_file = pjoin(ann_root, self.dataset_name, motion_files["ann"])
        with open(anno_file, 'r') as f:
            self.anno = json.load(f)

    def get_motion_anno(self, index):
        motion_id = self.anno[index]["motion_id"]

        motion_token_ids = self.motion_tokens[motion_id]
        motion_token_strs = []
        for tok_id in motion_token_ids:
            tok_str = MOTION_TOKEN.format(tok_id)
            motion_token_strs.append(tok_str)
        motion_token = ''.join(motion_token_strs)

        return motion_id, motion_token

    def __len__(self):
        return len(self.anno)

    def __getitem__(self, index):
        # question&answer annotation
        # TODO: implementation of dialogue
        task = self.anno[index]["task"]
        question = self.anno[index]["prompt"]
        # question = ""
        answer = random.choice(self.anno[index]["caption"])
        motion_id, motion_token = self.get_motion_anno(index)

        return motion_token, answer, question

class MotionValDataset(BaseDataset):

    cached_feats = {}

    def __init__(self, config, dataset_name, task, ann_root, motion_files, sample_ratio, **kwargs):
        super().__init__()
        self.sample_ratio = sample_ratio
        self.task = task
        self.dataset_name = dataset_name

        # load motion features
        self.motion_tokens = dict()
        token_dir = pjoin(ann_root, self.dataset_name, motion_files["tokens"])
        token_files = os.listdir(token_dir)
        for token_f in token_files:
            token_dat = np.load(pjoin(token_dir, token_f))
            token_id = token_f[:-4]
            self.motion_tokens[token_id] = token_dat

        # load text data
        anno_file = pjoin(ann_root, self.dataset_name, motion_files["ann"])
        with open(anno_file, 'r') as f:
            self.anno = json.load(f)

    def get_motion_anno(self, index):
        motion_id = self.anno[index]["motion_id"]

        motion_token_ids = self.motion_tokens[motion_id]
        motion_token_strs = []
        for tok_id in motion_token_ids:
            tok_str = MOTION_TOKEN.format(tok_id)
            motion_token_strs.append(tok_str)
        motion_token = ''.join(motion_token_strs)

        return motion_id, motion_token

    def __len__(self):
        return len(self.anno)

    def __getitem__(self, index):
        motion_id, motion_token = self.get_motion_anno(index)
        # question&answer annotation
        # TODO: implementation of dialogue
        question = self.anno[index]["prompt"]
        # question = ""
        ref_answers = self.anno[index]["caption"]
        pred_id = int(self.anno[index].get('pred_id', 0))
        type_info = self.anno[index].get('task', "")

        return motion_token, question, ref_answers, motion_id, index, pred_id, type_info