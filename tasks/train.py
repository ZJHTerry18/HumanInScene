import datetime
import logging
import time
from os.path import join

import pandas as pd
import torch
import torch.distributed as dist
import wandb
from torch.utils.data import ConcatDataset

from dataset import create_dataset, create_loader, create_sampler, create_collators
from dataset.dataloader import MetaSampleLoader, PrefetchLoader, IterLoader
from models.his_llm import HIS_LLM
from tasks.shared_utils import get_dataset_attrs, setup_model
from utils.basic_utils import (MetricLogger, SmoothedValue, setup_seed)
from utils.config_utils import setup_main
from utils.distributed import get_rank, get_world_size, is_main_process
from utils.logger import log_dict_to_wandb, setup_wandb
from utils.eval import calc_scanrefer_score, clean_answer, calc_scan2cap_score, calc_scanqa_score, calc_sqa3d_score, calc_multi3dref_score, calc_referit3d_score, calc_scanrefer_location_score, calc_multi3dref_location_score

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
# from pycocoevalcap.spice.spice import Spice
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer


import numpy as np
from tqdm import tqdm

import json
import os

logger = logging.getLogger(__name__)
max_bleus = [0.] * 4

tokenizer = PTBTokenizer()
scorers = [
    (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
    (Meteor(), "METEOR"),
    (Rouge(), "ROUGE_L"),
    (Cider(), "CIDEr"),
    # (Spice(), "SPICE")
]

max_global_step = 200000000


def train(
    model,
    model_without_ddp,
    train_loader,
    val_loaders,
    optimizer,
    epoch,
    global_step,
    device,
    scheduler,
    scaler,
    config,
    do_eval=True
):
    model.train()
    # model_without_ddp.llama_model.config.use_cache = False

    metric_logger = MetricLogger(delimiter="  ")
    eval_metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window=1, fmt="{value:.6f}"))
    loss_names = ["loss", "llm_loss", "act_loss", "cont_loss", "pos_loss", "obj_norm", "motion_norm"]

    # tot_param = sum(p.numel() for p in model_without_ddp.parameters())
    # trainable_param = sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad)
    # print(f"Total Params: {tot_param / 1e6}M")
    # print(f"Trainable Params: {trainable_param / 1e6}M")
    # exit()

    for name in loss_names:
        metric_logger.add_meter(
            f"{name}", SmoothedValue(window=1, fmt="{value:.6f}")
        )

    header = f"Train Epoch: [{epoch}]"
    log_freq = config.log_freq

    # if config.distributed:
    #     for d in train_loader:
    #         d.sampler.set_epoch(epoch)

    accum_iter = config.accum_iters
    eval_freq = config.eval_freq

    optimizer.zero_grad()
    iters_per_epoch = config.scheduler.iters_per_epoch
    
    for i in metric_logger.log_every(range(iters_per_epoch), log_freq, header):
        if i >= iters_per_epoch:
            break
        
        loader_name, batch = next(train_loader)
        for k in batch.keys():
            if type(batch[k]) == torch.Tensor:
                batch[k] = batch[k].to(device)
        
        # ## check batch input
        # print("Dataloader:", loader_name)
        # for k in batch.keys():
        #     if type(batch[k]) == torch.Tensor:
        #         print(k, batch[k].shape)
        #     else:
        #         print(k, batch[k])
        # ##
        loss_dict = model(device=device, **batch)
        loss = loss_dict["loss"] / accum_iter
        
        scaler.scale(loss).backward()

        if ((i + 1) % accum_iter == 0) or (i + 1 == iters_per_epoch):
            if config.optimizer.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.optimizer.max_grad_norm)
            scaler.step(optimizer)
            optimizer.zero_grad()
            scaler.update()
        scheduler.step()

        # logging
        for name in loss_names:
            if name not in loss_dict:
                continue
            value = loss_dict[name]
            value = value if isinstance(value, float) else value.item()
            metric_logger.update(**{f"{name}": value})
        metric_logger.update(lr=optimizer.param_groups[-1]["lr"])

        if is_main_process() and config.wandb.enable and global_step % log_freq == 0:
            logs = metric_logger.get_avg_dict()
            log_dict_to_wandb(logs, step=global_step, prefix="train/")

        global_step += 1

    if do_eval:
        val_metrics = evaluate_all(model, model_without_ddp, val_loaders, epoch, global_step, device, config)
        if is_main_process():
            for k, v in val_metrics.items():
                if k not in eval_metric_logger.meters:
                    eval_metric_logger.add_meter(k, SmoothedValue(window=1, fmt="{value:.4f}"))
            eval_metric_logger.update(**val_metrics)
        if is_main_process() and config.wandb.enable:
            logs = eval_metric_logger.get_avg_dict()
            log_dict_to_wandb(logs, step=global_step, prefix="val/")
        
        # if is_main_process():
        #     param_grad_dic = {
        #         k: v.requires_grad for (k, v) in model_without_ddp.named_parameters()
        #     }
        #     state_dict = model_without_ddp.state_dict()
        #     for k in list(state_dict.keys()):
        #         if k in param_grad_dic.keys() and not param_grad_dic[k]:
        #             # delete parameters that do not require gradient
        #             del state_dict[k]
        #     save_obj = {
        #         "model": state_dict,
        #         # "optimizer": optimizer.state_dict(),
        #         # "scheduler": scheduler.state_dict(),
        #         # "scaler": scaler.state_dict(),
        #         "config": config,
        #         "epoch": epoch,
        #         "global_step": global_step,
        #     }
        #     if i != len(train_loader) - 1 and config.do_save and not config.debug:
        #         torch.save(save_obj, join(config.output_dir, f"ckpt_{epoch:02d}.pth"))
    if global_step > max_global_step:
        return global_step

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged stats: {metric_logger.global_avg()}")
    return global_step


def evaluate_all(
    model,
    model_without_ddp,
    val_loaders,
    epoch,
    global_step,
    device,
    config
):
    logger.info("Start evaluating...")
    model.eval()
    model_without_ddp.llama_model.config.use_cache = True
    val_scores = {}
    for val_loader in val_loaders:
        new_val_scores = evaluate(model, val_loader, epoch, global_step, device, config)
        val_scores = {**val_scores, **new_val_scores}
    
    logger.info(f"[epoch={epoch}, global steps={global_step}] Val Results:")
    for k, v in val_scores.items():
        logger.info(f"{k}: {v}")
    
    model.train()
    model.module.llama_model.config.use_cache = False
    return val_scores


def evaluate(
    model,
    val_loader,
    epoch,
    global_step,
    device,
    config
):
    eval_name = val_loader.dataset.dataset_name
    logger.info(f"Evaluating {eval_name}...")
    if config.distributed:
        val_loader.sampler.set_epoch(epoch)

    sample_freq = len(val_loader) // 5 + 1
    cosine_scores, l2_distances = [], []
    save_preds = []
    logger.info(f"batch-size={val_loader.batch_size} length(#batches)={len(val_loader)}")
    for i, batch in tqdm(enumerate(val_loader)):
        for k in batch.keys():
            if type(batch[k]) == torch.Tensor:
                batch[k] = batch[k].to(device)
        with torch.no_grad():
            pred = model(**batch, device=device, is_eval=True)
        # if "target_captions" in batch:
        #     cosine_scores.append(pred["cosine_score"])
        #     l2_distances.append(pred["l2_dis"])

        # if len(batch["ref_captions"][0]) > 0:
        #     target = batch["ref_captions"]
        #     prompt = batch["custom_prompt"]
        #     tmp_pred = [p.replace("\n", " ").strip() for p in pred]
        #     tmp_target = ['\n'.join(p) for p in target]
        #     if i % sample_freq == 0:
        #         logger.info(f"\n[Prompt]\n{prompt[0]}\n[Pred]\n{tmp_pred[0]}\n[Target(s)]\n{tmp_target[0]}")
        batch_size = len(pred)
        for bi in range(batch_size):
            scene_id = batch["scene_id"][bi] if "scene_id" in batch.keys() else None
            motion_id = batch["motion_id"][bi] if "motion_id" in batch.keys() else None
            scene_motion_id = batch["scene_motion_id"][bi] if "scene_motion_id" in batch.keys() else None
            question = batch["questions"][bi]
            ref_answer = batch["ref_answers"][bi]
            type_info = batch['type_infos'][bi]
            tmp_pred = pred[bi]
            save_preds.append({
                "task": type_info,
                "scene_id": scene_id,
                "motion_id": motion_id,
                "scene_motion_id": scene_motion_id,
                "qa": [
                    {
                        "question": question,
                        "gt": ref_answer,
                        "pred": tmp_pred
                    }
                ]
            })
        # if i % sample_freq == 0:
        #     print(save_preds[-1])

    if len(save_preds) > 0:
        save_preds = sorted(save_preds, key=lambda x: f"{x['task']}_{x['scene_id']}_{x['scene_motion_id']}")
        with open(os.path.join(config.output_dir, f"preds_epoch{epoch}_step{global_step}_rank{get_rank()}_{eval_name}.json"),
                  "w") as f:
            json.dump(save_preds, f, indent=4)

    dist.barrier()
    if is_main_process():
        save_preds = []
        for rank in range(config.gpu_num):
            path = os.path.join(config.output_dir, f"preds_epoch{epoch}_step{global_step}_rank{rank}_{eval_name}.json")
            if os.path.exists(path):
                preds = json.load(open(path, "r"))
                save_preds += preds
                os.remove(path)
        save_preds = sorted(save_preds, key=lambda x: f"{x['task']}_{x['scene_id']}_{x['scene_motion_id']}")
        with open(os.path.join(config.output_dir, f"preds_epoch{epoch}_step{global_step}_{eval_name}.json"), "w") as f:
            json.dump(save_preds, f, indent=4)

    val_scores = {}
    if is_main_process() and len(save_preds) > 0:
        if eval_name == 'scanqa':
            val_scores = calc_scanqa_score(save_preds, tokenizer, scorers, config)
        elif eval_name == 'scanrefer':
            val_scores = calc_scanrefer_score(save_preds, config)
        elif eval_name in ["scan2cap", "scan2cap_location"]:
            val_scores = calc_scan2cap_score(save_preds, tokenizer, scorers, config)
        elif eval_name in ["sqa3d", "sqa3d_val"]:
            val_scores = calc_sqa3d_score(save_preds, tokenizer, scorers, config)
        elif eval_name == 'multi3dref':
            val_scores = calc_multi3dref_score(save_preds, config)
        elif eval_name in ['nr3d', 'sr3d']:
            val_scores = calc_referit3d_score(save_preds, eval_name, config)
        elif eval_name == "scanrefer_location":
            val_scores = calc_scanrefer_location_score(save_preds, config)
        elif eval_name == "multi3dref_location":
            val_score = calc_multi3dref_location_score(save_preds, config)
        elif eval_name in ["scanrefer_test", "scan2cap_test"]:
            pass
        # else:
        #     raise NotImplementedError
            # tmp_preds = {}
            # tmp_targets = {}
            # acc = 0
            # print("Total samples:", len(save_preds))
            # for i, output in enumerate(save_preds):
            #     item_id = f"{output['scene_id']}_{output['gt_id']}_{output['qid']}_{i}"
            #     pred = output["pred"]
            #     pred = clean_answer(pred)
            #     ref_captions = [clean_answer(caption) for caption in output['ref_captions']]
            #     if pred in ref_captions:
            #         acc += 1
            #     tmp_preds[item_id] = [{'caption': pred}]
            #     ref_captions = [p.replace("\n", " ").strip() for p in ref_captions]
            #     tmp_targets[item_id] = [{'caption': caption} for caption in ref_captions]
            # tmp_preds = tokenizer.tokenize(tmp_preds)
            # tmp_targets = tokenizer.tokenize(tmp_targets)
            # acc = acc / len(save_preds)
            # val_scores[f"[{eval_name}] Acc"] = acc
            # for scorer, method in scorers:
            #     score, scores = scorer.compute_score(tmp_targets, tmp_preds)
            #     if type(method) == list:
            #         for sc, scs, m in zip(score, scores, method):
            #             val_scores[f"[{eval_name}] {m}"] = sc
            #     else:
            #         val_scores[f"[{eval_name}] {method}"] = score
        print(json.dumps(val_scores, indent=4))
    return val_scores


def setup_dataloaders(config):
    # create datasets
    train_datasets, val_datasets = create_dataset(config)

    # create samplers
    if config.distributed:
        num_tasks = get_world_size()
        global_rank = get_rank()
        train_samplers = create_sampler(
            train_datasets, [True] * len(train_datasets), num_tasks, global_rank
        )
        val_samplers = create_sampler(
            val_datasets, [False] * len(val_datasets), num_tasks, global_rank
        )
    else:
        train_samplers = {k: None for k in train_datasets.keys()}
        val_samplers = {k: None for k in val_datasets.keys()}

    # create collators
    train_collate_fns, val_collate_fns = create_collators(config)

    train_loaders = create_loader(
        train_datasets,
        train_samplers,
        batch_size=[config.batch_size] * len(train_datasets),
        num_workers=[config.num_workers] * len(train_datasets),
        is_trains=[True] * len(train_datasets),
        collate_fns=train_collate_fns,
    )
    val_loaders = create_loader(
        val_datasets,
        val_samplers,
        batch_size=[config.batch_size] * len(val_datasets),
        num_workers=[config.num_workers] * len(val_datasets),
        is_trains=[False] * len(val_datasets),
        collate_fns=val_collate_fns,
    )

    return train_loaders, val_loaders


def main(config):
    if is_main_process() and config.wandb.enable:
        run = setup_wandb(config)

    # torch.autograd.set_detect_anomaly(True)
    setup_seed(config.seed + get_rank())
    device = torch.device(config.device)

    train_loaders, val_loaders = setup_dataloaders(config)

    num_steps_per_epoch = config.scheduler.iters_per_epoch
    config.scheduler.num_training_steps = num_steps_per_epoch * config.scheduler.epochs
    config.scheduler.num_warmup_steps = num_steps_per_epoch * config.scheduler.warmup_epochs
    torch.backends.cudnn.benchmark = True

    model_cls = eval(config.model.get('model_cls', 'HIS_LLM'))
    (
        model,
        model_without_ddp,
        optimizer,
        scheduler,
        scaler,
        start_epoch,
        global_step,
    ) = setup_model(
        config,
        model_cls=model_cls,
        find_unused_parameters=True,
    )
    if is_main_process() and config.wandb.enable:
        wandb.watch(model)
    
    save_step_interval = config.save_step_interval
    start_time = time.time()
    if not config.evaluate:
        logger.info("Start training")

        # Set train dataloader
        dataset_names = get_dataset_attrs(train_loaders, "dataset_name")
        dataset_ratios = get_dataset_attrs(train_loaders, "sample_ratio")
        train_loaders = [PrefetchLoader(loader) for loader in train_loaders]
        train_loaders = [IterLoader(loader, use_distributed=config.distributed) 
                         for loader in train_loaders]
        
        train_meta_loader = MetaSampleLoader(
            name2loader=dict(list(zip(dataset_names, train_loaders))),
            ratios=dataset_ratios)
        if not hasattr(train_meta_loader, "__next__"):
            # convert to iterator if not already
            train_meta_loader = iter(train_meta_loader)

        for epoch in range(start_epoch, config.scheduler.epochs):
            global_step = train(
                model,
                model_without_ddp,
                train_meta_loader,
                val_loaders,
                optimizer,
                epoch,
                global_step,
                device,
                scheduler,
                scaler,
                config
            )
            if is_main_process():
                logger.info(f"Epoch {epoch}")
                param_grad_dic = {
                    k: v.requires_grad for (k, v) in model_without_ddp.named_parameters()
                }
                state_dict = model_without_ddp.state_dict()
                for k in list(state_dict.keys()):
                    if k in param_grad_dic.keys() and not param_grad_dic[k]:
                        # delete parameters that do not require gradient
                        del state_dict[k]
                save_obj = {
                    "model": state_dict,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "global_step": global_step,
                }
                if ((epoch+1) % save_step_interval == 0 or epoch == config.scheduler.epochs - 1) and config.do_save and not config.debug:
                    if config.get("save_latest", False):
                        torch.save(save_obj, join(config.output_dir, "ckpt_latest.pth"))
                    else:
                        torch.save(save_obj, join(config.output_dir, f"ckpt_{epoch:02d}.pth"))

            if global_step > max_global_step:
                break
            dist.barrier()

    if config.evaluate:
        evaluate_all(model, model_without_ddp, val_loaders, start_epoch - 1, global_step, device, config)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f"Training time {total_time_str}")
    logger.info(f"Checkpoints and Logs saved at {config.output_dir}")

    if is_main_process() and config.wandb.enable:
        run.finish()


if __name__ == "__main__":
    cfg = setup_main()
    main(cfg)
