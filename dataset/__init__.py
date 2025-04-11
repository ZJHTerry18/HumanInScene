import torch
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from dataset.dataset_scene import SceneTrainDataset, SceneValDataset
from dataset.dataset_motion import MotionTrainDataset, MotionValDataset
from dataset.dataset_scenemotion import SceneMotionTrainDataset, SceneMotionValDataset
from dataset.dataset_scenemotion_task import SceneMotionTaskTrainDataset
from dataset.dataset_language import LanguageTrainDataset, LanguageValDataset
from dataset.collates import (
    scene_train_collate_fn, motion_train_collate_fn, scene_motion_train_collate_fn, language_train_collate_fn,
    scene_val_collate_fn, motion_val_collate_fn, scene_motion_val_collate_fn, language_val_collate_fn
)

import logging
logger = logging.getLogger(__name__)

_TRAIN_DATASETS = {
    "scene": SceneTrainDataset,
    "motion": MotionTrainDataset,
    "scene-motion": SceneMotionTaskTrainDataset,
    "language": LanguageTrainDataset,
}

_VAL_DATASETS = {
    "scene": SceneValDataset,
    "motion": MotionValDataset,
    "scene-motion": SceneMotionValDataset,
    "language": LanguageValDataset,
}

_TRAIN_COLLATES = {
    "scene": scene_train_collate_fn,
    "motion": motion_train_collate_fn,
    "scene-motion": scene_motion_train_collate_fn,
    "language": language_train_collate_fn,
}

_VAL_COLLATES = {
    "scene": scene_val_collate_fn,
    "motion": motion_val_collate_fn,
    "scene-motion": scene_motion_val_collate_fn,
    "language": language_val_collate_fn,
}

def create_dataset(config):
    train_dataset_cfg = config.data.train
    val_dataset_cfg = config.data.val

    train_datasets = []
    if not config.evaluate:
        for dataset_name, dataset_kwargs in train_dataset_cfg.items():
            task = dataset_kwargs["task"]
            dataset = _TRAIN_DATASETS[task](config=config, dataset_name=dataset_name, **dataset_kwargs)
            train_datasets.append(dataset)
    
    val_datasets = []
    for dataset_name, dataset_kwargs in val_dataset_cfg.items():
        task = dataset_kwargs["task"]
        dataset = _VAL_DATASETS[task](config=config, dataset_name=dataset_name, **dataset_kwargs)
        val_datasets.append(dataset)

    return train_datasets, val_datasets

def create_collators(config):
    train_dataset_cfg = config.data.train
    val_dataset_cfg = config.data.val

    train_collators = []
    if not config.evaluate:
        for dataset_name, dataset_kwargs in train_dataset_cfg.items():
            task = dataset_kwargs["task"]
            train_collators.append(_TRAIN_COLLATES[task])

    val_collators = []
    for dataset_name, dataset_kwargs in val_dataset_cfg.items():
        task = dataset_kwargs["task"]
        val_collators.append(_VAL_COLLATES[task])

    return train_collators, val_collators

def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset, shuffle in zip(datasets, shuffles):
        sampler = torch.utils.data.DistributedSampler(
            dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle
        )
        samplers.append(sampler)
    return samplers


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset, sampler, bs, n_worker, is_train, collate_fn in zip(
        datasets, samplers, batch_size, num_workers, is_trains, collate_fns
    ):
        if is_train:
            shuffle = sampler is None
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
            persistent_workers=True if n_worker > 0 else False,
        )
        loaders.append(loader)
    return loaders


def iterate_dataloaders(dataloaders):
    """Alternatively generate data from multiple dataloaders,
    since we use `zip` to concat multiple dataloaders,
    the loop will end when the smaller dataloader runs out.

    Args:
        dataloaders List(DataLoader): can be a single or multiple dataloaders
    """
    for data_tuples in zip(*dataloaders):
        for idx, data in enumerate(data_tuples):
            yield dataloaders[idx].dataset.media_type, data
