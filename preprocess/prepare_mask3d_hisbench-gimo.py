import os
import argparse
import json
import numpy as np
import pprint
import time
import multiprocessing as mp
from functools import partial

from plyfile import PlyData

import torch

import csv
import glob
from collections import defaultdict
from tqdm import tqdm

import mmengine

from mask3d_scannet200_constants import VALID_CLASS_IDS_200, SCANNET_SEMANTICS
CLASS_ID_SEMANTICS = {VALID_CLASS_IDS_200[i]: SCANNET_SEMANTICS[i] for i in range(len(VALID_CLASS_IDS_200))}

ids = set()
def process_one_scene(params, tmp_dir, id2class):
    cur_dir, file_path = params
    scene_id = file_path.split("/")[-2]
    cur_scene_out = []
    save_file = os.path.join(tmp_dir, f"{scene_id}.pt")
    if os.path.exists(save_file):
        return
    with open(file_path, "r") as f:
        for line in f.readlines():
            predict_path, class_id, score = line.split(" ")
            class_id = int(class_id)
            ids.add(class_id)
            predict_path = os.path.join(cur_dir, scene_id, predict_path)
            segments = []
            with open(predict_path, "r") as f2:
                for i, l in enumerate(f2.readlines()):
                    if l[0] == "1":
                        segments.append(i)
            cur_scene_out.append({
                "label": id2class[class_id],
                "segments": segments
            })
    torch.save(cur_scene_out, save_file)

def process_per_scan(scan_id, scan_dir, out_dir, tmp_dir, apply_global_alignment=True, is_test=False):
    pcd_out_dir = os.path.join(out_dir, 'pcd_all')
    # if os.path.exists(os.path.join(pcd_out_dir, '%s.pth'%(scan_id))):
    #     print(f"skipping {scan_id}...")
    #     return
    print(f"processing {scan_id}...")
    os.makedirs(pcd_out_dir, exist_ok=True)
    # obj_out_dir = os.path.join(out_dir, 'instance_id_to_name_all')
    # os.makedirs(obj_out_dir, exist_ok=True)

    # Load point clouds with colors
    with open(os.path.join(scan_dir, scan_id + '.ply'), 'rb') as f:
        plydata = PlyData.read(f) # elements: vertex, face
    points = np.array([list(x) for x in plydata.elements[0]]) # [[x, y, z, r, g, b, alpha]]
    coords = np.ascontiguousarray(points[:, :3])
    colors = np.ascontiguousarray(points[:, 3:6])

    # # TODO: normalize the coords and colors
    # coords = coords - coords.mean(0)
    # colors = colors / 127.5 - 1

    if apply_global_alignment:
        print('useapply')
        align_matrix = np.array([
            [1, 0, 0, 0],
            [0, 0, -1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        # Transform the points
        pts = np.ones((coords.shape[0], 4), dtype=coords.dtype)
        pts[:, 0:3] = coords
        coords = np.dot(pts, align_matrix.transpose())[:, :3]  # Nx4
        # Make sure no nans are introduced after conversion
        assert (np.sum(np.isnan(coords)) == 0)

    # Load point labels if any
    # colored by nyu40 labels (ply property 'label' denotes the nyu40 label id)
    # with open(os.path.join(scan_dir, scan_id, '%s_vh_clean_2.labels.ply'%(scan_id)), 'rb') as f:
    #     plydata = PlyData.read(f)
    # sem_labels = np.array(plydata.elements[0]['label']).astype(np.long)
    # assert len(coords) == len(colors) == len(sem_labels)
    # sem_labels = None

    # Map each point to segment id
    # if not os.path.exists(os.path.join(scan_dir, scan_id, '%s_vh_clean_2.0.010000.segs.json'%(scan_id))):
    #     return
    # with open(os.path.join(scan_dir, scan_id, '%s_vh_clean_2.0.010000.segs.json'%(scan_id)), 'r') as f:
    #     d = json.load(f)
    # seg = d['segIndices']
    # segid_to_pointid = {}
    # for i, segid in enumerate(seg):
    #     segid_to_pointid.setdefault(segid, [])
    #     segid_to_pointid[segid].append(i)

    # Map object to segments
    instance_class_labels = []
    instance_segids = []

    # cur_instances = pointgroup_instances[scan_id].copy()
    if not os.path.exists(os.path.join(tmp_dir, f"{scan_id}.pt")):
        return
    cur_instances = torch.load(os.path.join(tmp_dir, f"{scan_id}.pt"))
    for instance in cur_instances:
        instance_class_labels.append(instance["label"])
        instance_segids.append(instance["segments"])

    torch.save(
        (coords, colors, instance_class_labels, instance_segids), 
        os.path.join(pcd_out_dir, '%s.pth'%(scan_id))
    )
    

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--scannet_dir', type=str, default="/home/zhaojiaohe/data/zhaojiahe/HIS_dataset/GIMO/scenes",
                        help='the path to the scans')
    parser.add_argument('--output_dir', type=str, default="/home/zhaojiaohe/data/zhaojiahe/HIS_dataset/",
                        help='the path of the directory to be saved preprocessed scans')

    # Optional arguments.
    parser.add_argument('--inst_seg_dir', default=None, type=str)
    parser.add_argument('--segment_dir', default="/home/zhaojiaohe/data/zhaojiahe/HIS_dataset/GIMO/scenes", type=str,
                        help='the path to the predicted masks of pretrained segmentor')
    parser.add_argument('--num_workers', default=-1, type=int,
                        help='the number of processes, -1 means use the available max')
    parser.add_argument('--parallel', default=False, action='store_true',
                        help='use mmengine to process in a parallel manner')
    parser.add_argument('--apply_global_alignment', default=False, action='store_true',
                        help='rotate/translate entire scan globally to aligned it with other scans')
    args = parser.parse_args()

    # Print the args
    args_string = pprint.pformat(vars(args))
    print(args_string)

    return args


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    num_workers = args.num_workers

    id2class = CLASS_ID_SEMANTICS
    
    if args.scannet_dir:
        tmp_dir = os.path.join(args.scannet_dir, 'mask3d_inst_seg')
        if not os.path.exists(tmp_dir):
            os.mkdir(tmp_dir)
        params = []
        cur_dir = os.path.join(args.scannet_dir, 'pred_mask3d_denoise')
        for file_path in glob.glob(os.path.join(cur_dir, "*/mask3d_predictions.txt")):
            params.append((cur_dir, file_path))
        fn = partial(
            process_one_scene,
            tmp_dir=tmp_dir,
            id2class=id2class
        )
        if args.parallel:
            mmengine.utils.track_parallel_progress(fn, params, num_workers)
        else:
            for param in tqdm(params):
                fn(param)
                print(len(ids))
    else:
        tmp_dir = args.inst_seg_dir

    print("args.apply_global_alignment",args.apply_global_alignment)

    fn = partial(
        process_per_scan,
        scan_dir=args.scannet_dir,
        out_dir=args.output_dir,
        tmp_dir=tmp_dir,
        apply_global_alignment=args.apply_global_alignment,
    )

    scan_ids = [x[:-4] for x in os.listdir(args.scannet_dir) if x.endswith('.ply') and not x.endswith('_ds.ply')]
    scan_ids.sort()
    print('%d scans' % (len(scan_ids)))

    if args.parallel:
        mmengine.utils.track_parallel_progress(fn, scan_ids, num_workers)
    else:
        for scan_id in scan_ids:
            fn(scan_id)

    # all_feats = {}
    # for split in ['train', 'val']:
    #     cur_feat_dir = os.path.join(args.segment_dir, split, 'features')
    #     for filename in tqdm(os.listdir(cur_feat_dir)):
    #         if not filename.endswith('.pt'):
    #             continue
    #         scene_id = filename.split('.pt')[0]
    #         tmp_feat = torch.load(os.path.join(cur_feat_dir, filename), map_location='cpu')
    #         for i in range(tmp_feat.shape[1]):
    #             all_feats[f"{scene_id}_{i:02}"] = tmp_feat[0, i]
    # torch.save(all_feats, "annotations/scannet_mask3d_mask3d_feats.pt")


if __name__ == '__main__':
    main()

