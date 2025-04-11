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



class LanguageTrainDataset(BaseDataset):

    cached_feats = {}

    def __init__(self, config, dataset_name, task, ann_root, ann_file, sample_ratio, **kwargs):
        super().__init__()

        self.sample_ratio = sample_ratio
        self.task = task
        self.dataset_name = dataset_name

        # load text data
        self.anno = []
        anno_file = pjoin(ann_root, dataset_name, ann_file)
        with open(anno_file, 'r') as f:
            self.anno.extend(json.load(f))

    def __len__(self):
        return len(self.anno)

    def __getitem__(self, index):
        # question&answer annotation
        # TODO: implementation of dialogue
        qas = self.anno[index]["qa"]
        task = self.anno[index]["task"]
        question = qas[0]["question"]
        # question = ""
        answer = qas[0]["answer"]

        return answer, question

class LanguageValDataset(BaseDataset):

    cached_feats = {}

    def __init__(self, config, dataset_name, task, ann_root, ann_file, **kwargs):
        super().__init__()
        self.task = task
        self.dataset_name = dataset_name

        # load text data
        self.anno = []
        anno_file = pjoin(ann_root, dataset_name, ann_file)
        with open(anno_file, 'r') as f:
            self.anno.extend(json.load(f))

    def __len__(self):
        return len(self.anno)

    def __getitem__(self, index):
        pred_id = int(self.anno[index].get('pred_id', 0))
        type_info = self.anno[index].get('task', "")
        
        # question&answer annotation
        # TODO: implementation of dialogue
        qas = self.anno[index]["qa"]
        task = self.anno[index]["task"]
        question = qas[0]["question"]
        # question = ""
        ref_answers = [qas[0]["answer"]]

        return question, ref_answers, index, pred_id, type_info