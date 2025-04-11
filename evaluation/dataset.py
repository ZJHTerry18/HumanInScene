import os
from os.path import join as pjoin
import json
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import decord
decord.bridge.set_bridge("torch")
from decord import VideoReader, cpu

def get_segment_indices(start_frame, end_frame, num_segments):
    seg_size = float(end_frame - start_frame) / num_segments
    start = start_frame + int(seg_size / 2)
    offsets = np.array([
        start + int(np.round(seg_size * idx)) for idx in range(num_segments)
    ])
    return offsets

def load_video(video_path, start=None, end=None, num_segments=8, size=(224, 224)):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    num_frames = len(vr)
    fps = float(vr.get_avg_fps())
    
    start_frame = min(max(0, int(start * fps) if start is not None else 0), num_frames - 1)
    end_frame = min(num_frames, int(end * fps) if end is not None else num_frames)
    frame_indices = get_segment_indices(start_frame=start_frame, end_frame=end_frame - 1, num_segments=num_segments)

    images_group = list()
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].numpy())
        img = img.resize(size)
        images_group.append(img)
    
    return images_group

class HISBench(Dataset):
    def __init__(self, ann_dir, data_root, video_type='render', num_frames=8, frame_size=(224, 224)):
        self.ann_dir = ann_dir
        self.data_root = data_root
        self.video_type = video_type
        self.num_frames = num_frames
        self.frame_size = frame_size

        # load json annotations of all tasks
        ann_jsons = sorted(os.listdir(ann_dir))
        self.task_ann_dict = dict()
        self.all_anns = []
        for jf in ann_jsons:
            task = jf[:-5]
            with open(os.path.join(ann_dir, jf), 'r') as f:
                anns = json.load(f)
            self.task_ann_dict[task] = anns
            self.all_anns.extend(anns)

    def __getitem__(self, index):
        ann_dat = self.all_anns[index]
        data_dir = pjoin(self.data_root, ann_dat["source"], 'motions', ann_dat["motion"])
        video_path = pjoin(data_dir, f"video_{self.video_type}.mp4")
        if not os.path.isfile(video_path):
            video_path = pjoin(data_dir, f"video_render.mp4")
        start, end = ann_dat["start"], ann_dat["end"]
        
        video_frames = load_video(video_path, start=start, end=end, num_segments=self.num_frames, size=self.frame_size)
        qas = ann_dat["qa"]
        metadata = {
            "task": ann_dat["task"],
            "source": ann_dat["source"],
            "motion": ann_dat["motion"],
            "start": ann_dat["start"],
            "end": ann_dat["end"]
        }

        return video_frames, qas, metadata

    def __len__(self,):
        return len(self.all_anns)

class HISBench_cap(Dataset):
    def __init__(self, ann_dir, data_root, video_type='render'):
        self.ann_dir = ann_dir
        self.data_root = data_root
        self.video_type = video_type

        # load json annotations of all tasks
        ann_jsons = sorted(os.listdir(ann_dir))
        self.task_ann_dict = dict()
        self.all_anns = []
        for jf in ann_jsons:
            task = jf[:-5]
            with open(os.path.join(ann_dir, jf), 'r') as f:
                anns = json.load(f)
            self.task_ann_dict[task] = anns
            self.all_anns.extend(anns)

    def __getitem__(self, index):
        ann_dat = self.all_anns[index]
        data_dir = pjoin(self.data_root, ann_dat["source"], 'motions', ann_dat["motion"])
        video_caps_path = pjoin(data_dir, f"video_caption_{self.video_type}_qwen.json")
        if not os.path.isfile(video_caps_path):
            video_caps_path = pjoin(data_dir, f"video_caption_rgb_qwen.json")
        try:
            with open(video_caps_path, 'r') as f:
                video_captions = json.load(f)
        except Exception as e:
            print(e)
            print(video_caps_path)

        qas = ann_dat["qa"]
        metadata = {
            "task": ann_dat["task"],
            "source": ann_dat["source"],
            "motion": ann_dat["motion"],
            "start": ann_dat["start"],
            "end": ann_dat["end"]
        }

        return video_captions, qas, metadata

class HISBench_scenemot(Dataset):
    def __init__(self, ann_dir, data_root):
        self.ann_dir = ann_dir
        self.data_root = data_root

        # load json annotations of all tasks
        ann_jsons = sorted(os.listdir(ann_dir))
        self.task_ann_dict = dict()
        self.all_anns = []
        for jf in ann_jsons:
            task = jf[:-5]
            with open(os.path.join(ann_dir, jf), 'r') as f:
                anns = json.load(f)
            self.task_ann_dict[task] = anns
            self.all_anns.extend(anns)
        
        scene_caption_file = pjoin(data_root, "scene_cap_ll3da.jsonl")
        self.scene_cap_dict = dict()
        with open(scene_caption_file, 'r') as f:
            for line in f:
                content = json.loads(line)
                scene_name = content["scene"]
                scene_cap = content["qa"][0]["pred"]
                self.scene_cap_dict[scene_name] = scene_cap

    def __getitem__(self, index):
        ann_dat = self.all_anns[index]

        # load scene caption
        scene_caption = self.scene_cap_dict[ann_dat["scene"]]

        # load motion caption
        motion_cap_dir = pjoin(self.data_root, "motion_cap_avatargpt")
        dat_source = ann_dat["source"]
        motion_parts = '#'.join(ann_dat["motion"].split('/'))
        if dat_source == "PROX":
            start, end = ann_dat["start"], ann_dat["end"]
        else:
            start = int(ann_dat["start"] * 25)
            end = int(ann_dat["end"] * 25)
        motion_file_name = '#'.join([dat_source, motion_parts, f"{start}_{end}.txt"])
        with open(pjoin(motion_cap_dir, motion_file_name), 'r') as f:
            lines = f.readlines()
        motion_caption = lines[1].strip()

        qas = ann_dat["qa"]
        metadata = {
            "task": ann_dat["task"],
            "source": ann_dat["source"],
            "motion": ann_dat["motion"],
            "start": ann_dat["start"],
            "end": ann_dat["end"]
        }

        return scene_caption, motion_caption, qas, metadata

if __name__ == "__main__":
    ann_dir = 'benchmark/output_checked'
    data_root = '/home/zhaojiaohe/data/zhaojiahe/HIS_dataset'

    benchmark_dataset = HISBench(ann_dir=ann_dir, data_root=data_root)