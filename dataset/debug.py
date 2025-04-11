import sys
import os
import torch
# 添加 scripts 文件夹的父目录到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '/home/zhaojiaohe/codes/Chat-Scene/')))
from scripts import config
from dataset.dataset_train import TrainDataset
train_tag="scanrefer#scan2cap#scanqa#sqa3d#multi3dref#nr3d_caption#obj_align"
train_files = []
#print(train_tag.split('#'))
train_name="scan2cap"
ann_list=config.train_file_dict[train_name]
feat_file, img_feat_file, attribute_file, anno_file = ann_list[:4]
print(feat_file)
print(img_feat_file)
print(attribute_file)
print(anno_file)
feats = torch.load(feat_file, map_location='cpu')
scan_ids = set('_'.join(x.split('_')[:2]) for x in feats.keys())
#print(scan_ids)
scan_id='scene0187_01'
attributes = torch.load(attribute_file, map_location='cpu')
print(attributes)
scene_attr = attributes[scan_id]
obj_ids = scene_attr['obj_ids']