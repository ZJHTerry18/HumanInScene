import random
import logging
from abc import ABC

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
import torch.nn.functional as F

from common.constants import MOTION_TOKEN, ACTIVITY_LABELS, JOINT_INDICES, POS_LABELS
from .modeling_llama import LlamaForCausalLM
from .hsi.scene_motion_fusion import SMFusion
from .hsi.interaction_head import ActivityHead, ContactHead, PositionHead
from .utils import _init_weights
from transformers import LlamaTokenizer, LlamaConfig
from models.position_embedding import PositionEmbeddingCoordsSine, PositionEmbeddingSpaTempSine
from peft import LoraConfig, get_peft_model
# from models.load_llama import init_llama_model
from torch.nn.utils.rnn import pad_sequence
from timm.models.layers import trunc_normal_

import contextlib
from dataset.base_dataset import update_caption, recover_caption

logger = logging.getLogger(__name__)


def nclamp(input, min, max):
    return input.clamp(min=min, max=max).detach() + input - input.detach()


def print_grad_status(model):
    """Call this function after losses.backward()
    and it will find out all variables without grad, which
    means that the varaible is not in the graph.
    """
    for name, p in model.named_parameters():
        print('{:80s}{:20s}{:20s}{}'.format(name,
            '(Trainable)' if p.requires_grad else '(Fixed)',
            '(Has grad):' if p.grad is not None else '(No grad backward):',
            list(p.shape)))


class HIS_LLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        llama_model_path = config.model.llama_model_path
        self.low_resource = config.model.low_resource
        self.max_txt_len = config.model.max_txt_len
        self.end_sym = config.model.end_sym
        self.system_path = config.model.system_path
        self.instruction_paths = config.model.instruction_paths
        self.role = config.model.role

        # scene input settings
        scene_cfg = config.model.scene
        self.no_obj = scene_cfg.no_obj
        self.add_scene_token = scene_cfg.add_scene_token
        self.add_img_token = scene_cfg.add_img_token
        self.train_obj_emb = scene_cfg.train_obj_emb
        self.train_img_proj = scene_cfg.train_img_proj
        self.obj_input_dim = scene_cfg.obj_input_dim
        self.img_input_dim = scene_cfg.img_input_dim
        self.attr_dim = scene_cfg.attr_dim
        self.scene_dim = scene_cfg.scene_dim
        self.pos_dim = scene_cfg.pos_dim
        self.max_obj_num = scene_cfg.max_obj_num
        self.bidirection = scene_cfg.bidirection
        self.add_scene_pos_emb = scene_cfg.add_scene_pos_emb
        self.feat_fusion = scene_cfg.feat_fusion
        self.fuse_with_id = scene_cfg.fuse_with_id
        self.use_scene_loc_token = scene_cfg.use_scene_loc_token

        # motion input settings
        motion_cfg = config.model.motion
        self.motion_token_num = motion_cfg.motion_token_num
        self.add_motion_pos_emb = motion_cfg.add_motion_pos_emb
        self.train_motion_emb = motion_cfg.train_motion_emb
        self.motion_input_dim = motion_cfg.motion_input_dim
        self.motion_emb_file = motion_cfg.motion_emb_file

        # scene-motion interaction settings
        int_config = config.model.interaction
        self.scene_motion_fuse = int_config.scene_motion_fuse
        self.sm_hidden_dim = int_config.sm_hidden_dim
        self.sm_num_block = int_config.sm_num_block
        self.activity_w = int_config.activity_w
        self.contact_w = int_config.contact_w
        self.position_w = int_config.position_w
        self.int_mid_dim = int_config.int_mid_dim

        self.debug = config.debug
        if not self.debug:
            logger.info('Loading LLaMA')
            self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model_path, use_fast=False, legacy=False)
            # self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
            if self.low_resource:
                self.llama_model = LlamaForCausalLM.from_pretrained(
                    llama_model_path,
                    torch_dtype=torch.bfloat16,
                    load_in_8bit=True,
                    device_map="auto",
                    attn_implementation="flash_attention_2"
                )
            else:
                self.llama_model = LlamaForCausalLM.from_pretrained(
                    llama_model_path,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2"
                )
            # print(torch.cuda.memory_allocated(device="cuda:0")/1e9)
            # self.llama_model = self.llama_model.to("cuda")
            # print(torch.cuda.memory_allocated(device="cuda:0")/1e9)
            # breakpoint()
            logger.info("freeze LLAMA")
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = False

            if config.model.use_lora:
                def find_linear_layers(model, lora_target_modules):
                    cls = torch.nn.Linear
                    lora_module_names = set()
                    for name, module in model.named_modules():
                        if (
                            isinstance(module, cls)
                            and all(
                                [
                                    x not in name
                                    for x in [
                                        "instance2embed",
                                        "hidden_state2query"
                                    ]
                                ]
                            )
                            and any([x in name for x in lora_target_modules])
                        ):
                            lora_module_names.add(name)
                    return sorted(list(lora_module_names))
            
                lora_target_modules = find_linear_layers(self.llama_model, config.model.lora.lora_target_modules)

                lora_config = LoraConfig(
                    r=config.model.lora.lora_r,
                    lora_alpha=config.model.lora.lora_alpha,
                    target_modules=lora_target_modules,
                    lora_dropout=config.model.lora.lora_dropout,
                    bias="none",
                    task_type="CAUSAL_LM",
                )
                self.llama_model = get_peft_model(self.llama_model, lora_config)
                self.llama_model.print_trainable_parameters()
                self.llama_model.model.lm_head.weight.requires_grad = True
                self.llama_model.model.lm_head.weight.data = self.llama_model.model.lm_head.weight.data.float()
                self.llama_model.print_trainable_parameters()
                self.llama_model.model.model.embed_tokens.weight.requires_grad = True
                self.llama_model.model.model.embed_tokens.weight.data = self.llama_model.model.model.embed_tokens.weight.data.float()
                self.llama_model.print_trainable_parameters()
            else:
                pass
                
            self.llama_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})

            self.llama_dim = self.llama_model.config.hidden_size
            logger.info('Loading LLAMA Done')
        else:
            self.llama_model = None
            self.llama_dim = 4096

        # scene-motion fusion module
        if self.scene_motion_fuse:
            self.object_proj = nn.Linear(self.obj_input_dim, self.sm_hidden_dim)
            self.motion_proj = nn.Linear(self.motion_input_dim, self.sm_hidden_dim)
            self.pos_proj = nn.Linear(self.pos_dim, self.sm_hidden_dim)
            self.fusion_module = SMFusion(
                hidden_dim=self.sm_hidden_dim,
                num_blocks=self.sm_num_block,
            )
            self.object_post_proj = nn.Sequential(
                nn.Linear(self.sm_hidden_dim, self.llama_dim),
                nn.GELU(),
                nn.Linear(self.llama_dim, self.llama_dim)
            )
            self.motion_post_proj = nn.Sequential(
                nn.Linear(self.sm_hidden_dim, self.llama_dim),
                nn.GELU(),
                nn.Linear(self.llama_dim, self.llama_dim)
            )
            self.object_img_post_proj = nn.Sequential(
                nn.Linear(self.sm_hidden_dim, self.llama_dim),
                nn.GELU(),
                nn.Linear(self.llama_dim, self.llama_dim)
            )
            self.pos_post_proj = nn.Linear(self.pos_dim, self.llama_dim)
        else:
            self.object_proj = nn.Sequential(
                nn.Linear(self.obj_input_dim, self.llama_dim),
                nn.GELU(),
                nn.Linear(self.llama_dim, self.llama_dim)
            )
            self.motion_proj = nn.Sequential(
                nn.Linear(self.motion_input_dim, self.llama_dim),
                nn.GELU(),
                nn.Linear(self.llama_dim, self.llama_dim)
            )
            self.pos_proj = nn.Sequential(
                nn.Linear(self.pos_dim, self.llama_dim)
            )
            self.object_img_proj = nn.Sequential(
                nn.Linear(self.img_input_dim, self.llama_dim),
                nn.GELU(),
                nn.Linear(self.llama_dim, self.llama_dim)
            )
            if not self.train_img_proj:
                for p in self.object_img_proj.parameters():
                    p.requires_grad = False
        self.pos_embedding = PositionEmbeddingSpaTempSine(d_pos=self.pos_dim)

        # scene-motion interaction subtask
        int_input_dim = self.sm_hidden_dim if self.scene_motion_fuse else self.llama_dim
        if self.activity_w > 0:
            num_activity_cls = len(ACTIVITY_LABELS) + 1
            self.activity_head = ActivityHead(
                input_dim=int_input_dim,
                hidden_dim=self.int_mid_dim,
                num_classes=num_activity_cls
            )
        if self.contact_w > 0:
            num_joints = len(JOINT_INDICES)
            self.contact_head = ContactHead(
                input_dim=int_input_dim,
                hidden_dim=self.int_mid_dim,
                num_joints=num_joints
            )
        if self.position_w > 0:
            num_pos_cls = len(POS_LABELS) + 1
            self.position_head = PositionHead(
                input_dim=int_input_dim,
                hidden_dim=self.int_mid_dim,
                num_classes=num_pos_cls
            )

        self.motion_pretrain_embeds = torch.load(self.motion_emb_file)
        
        self.object_proj.apply(_init_weights)
        self.pos_proj.apply(_init_weights)
        self.motion_proj.apply(_init_weights)
        if self.scene_motion_fuse:
            self.object_post_proj.apply(_init_weights)
            self.motion_post_proj.apply(_init_weights)
            self.pos_post_proj.apply(_init_weights)

        self.instructions = dict()
        with open(self.system_path, "r") as f:
            self.system = "\n".join([x.strip() for x in f.readlines()])
        for task, path in self.instruction_paths.items():
            with open(path, "r") as f:
                self.instructions[task] = "\n".join([x.strip() for x in f.readlines()])

        if not self.debug:
            self.pre_embeds = dict()
            for task in self.instructions.keys():
                self.pre_embeds[task] = self.prepare_fixed_embed(task)
        self.last_embed = None

        # print_grad_status(self)

    def llama_embed_tokens(self, token_ids):
        if self.config.model.use_lora:
            return self.llama_model.model.model.embed_tokens(token_ids)
        else:
            return self.llama_model.model.embed_tokens(token_ids)

    def prepare_fixed_embed(self, task):
        prompt = self.system + " " + self.instructions[task] + " " + self.role[0] + ": " 
        p_splits = prompt.split("<REPLACE>")
        p_embeds = []
        for i, p_seg in enumerate(p_splits):
            if i == 0:
                p_token = self.llama_tokenizer(p_seg, return_tensors="pt", add_special_tokens=True)
            else:
                p_token = self.llama_tokenizer(p_seg, return_tensors="pt", add_special_tokens=False)
            p_embed = self.llama_embed_tokens(p_token.input_ids).squeeze(0).detach()
            p_embeds.append(p_embed)
        return p_embeds
    
    def get_text_emb(self, text, device="cpu"):
        text_tokens = self.llama_tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
        embeds = self.llama_embed_tokens(text_tokens.input_ids)
        embeds = embeds.detach()
            
        return embeds

    def encode_object_feat(self, feat, img_feat, locs):
        feat = torch.nn.functional.normalize(feat, dim=-1)
        img_feat = torch.nn.functional.normalize(img_feat, dim=-1)
        return feat, img_feat

    @staticmethod
    def get_dist_attention(pos, dist_exp=1):
        # pos (bs, obj_num, 3)
        dist = pos.unsqueeze(1) - pos.unsqueeze(2)
        dist = torch.sum(dist.abs()**dist_exp, dim=-1)
        dist_attn = torch.nn.functional.softmax(-dist, dim=-1)
        return dist_attn
    
    def get_object_list_embed(self, embed_obj, embed_img, embed_scene, scene_mask, assigned_ids):
        valid_ids = torch.where(scene_mask)[0].tolist()

        assigned_ids = assigned_ids[valid_ids]
        selected_embed_obj = embed_obj[assigned_ids]
        
        if embed_img is None and embed_scene is None:
            object_list_embed = torch.zeros((selected_embed_obj.shape[0], selected_embed_obj.shape[1]), dtype=selected_embed_obj.dtype, device=selected_embed_obj.device)
            object_list_embed[:, :] = selected_embed_obj
            return object_list_embed
        if embed_img is None and embed_scene is not None:
            object_list_embed = torch.zeros((selected_embed_obj.shape[0] * 2, selected_embed_obj.shape[1]), dtype=selected_embed_obj.dtype, device=selected_embed_obj.device)
            object_list_embed[0::2, :] = selected_embed_obj
            object_list_embed[1::2, :] = embed_scene[assigned_ids]
            return object_list_embed
        if embed_img is not None and embed_scene is None:
            object_list_embed = torch.zeros((selected_embed_obj.shape[0] * 2, selected_embed_obj.shape[1]), dtype=selected_embed_obj.dtype, device=selected_embed_obj.device)
            object_list_embed[0::2, :] = selected_embed_obj
            object_list_embed[1::2, :] = embed_img[assigned_ids]
            return object_list_embed
        if embed_img is not None and embed_scene is not None:
            object_list_embed = torch.zeros((selected_embed_obj.shape[0] * 3, selected_embed_obj.shape[1]), dtype=selected_embed_obj.dtype, device=selected_embed_obj.device)
            object_list_embed[0::3, :] = selected_embed_obj
            object_list_embed[1::3, :] = embed_scene[assigned_ids]
            object_list_embed[2::3, :] = embed_img[assigned_ids]
            return object_list_embed
        return object_list_embed

    def get_min_max_coord(self, xyz, scene_mask):
        scene_mask = scene_mask.unsqueeze(-1).expand_as(xyz)
        masked_xyz_min = torch.where(scene_mask, xyz, torch.full_like(xyz, float('inf')))
        masked_xyz_max = torch.where(scene_mask, xyz, torch.full_like(xyz, float('-inf')))
        mins = masked_xyz_min.min(dim=1)[0]
        maxs = masked_xyz_max.max(dim=1)[0]
        return mins, maxs

    def forward_train(self,
        questions, 
        answers, 
        scene_feat=None, 
        scene_img_feat=None, 
        scene_locs=None, 
        scene_mask=None, 
        obj_nums=None, 
        assigned_ids=None,
        motion_tokens=None,
        motion_trajs=None,
        activity_labels=None,
        contact_labels=None,
        contact_masks=None,
        position_labels=None,
        position_weights=None,
        device='cuda:0',
        is_eval=False, **kwargs
    ):
        ## define task
        if scene_feat is not None and motion_tokens is not None:
            task = "scene-motion"
        elif scene_feat is not None:
            task = "scene"
        elif motion_tokens is not None:
            task = "motion"
        else:
            task = "language"
        
        proj_object_embed = None
        scene_pos_embeds = None
        ## prepare scene embeddings
        if scene_feat is not None:
            object_embed, object_img_embed = self.encode_object_feat(scene_feat, scene_img_feat, scene_locs)
            # device = object_embed.device
            batch_size = object_embed.shape[0]
            proj_object_embed = self.object_proj(object_embed)
            if self.add_scene_pos_emb:
                # get spatial coords
                scene_mins, scene_maxs = self.get_min_max_coord(scene_locs[:, :, :3], scene_mask) # [B, 3]
                # get temporal coords
                B, _ = scene_mins.shape
                t_mins = torch.zeros((B, 1), dtype=scene_mins.dtype, device=scene_mins.device)
                t_maxs = torch.zeros((B, 1), dtype=scene_maxs.dtype, device=scene_maxs.device)
                if motion_tokens is not None:
                    for i, motion_token in enumerate(motion_tokens):
                        motion_token_strs = motion_token.split('>')[:-1]
                        motion_len = len(motion_token_strs)
                        t_maxs[i].fill_(motion_len)
                else:
                    t_maxs.fill_(10.0)
                # combine spatial and temporal coords
                mins = torch.cat([scene_mins, t_mins], dim=1)
                maxs = torch.cat([scene_maxs, t_maxs], dim=1)
                
                # calculate scene position embeddings
                scene_pos_embeds = self.pos_embedding(scene_locs[:, :, :3], input_range=[mins, maxs]) / 10
                proj_pos_embeds = self.pos_proj(scene_pos_embeds)
                proj_object_embed = proj_object_embed + proj_pos_embeds
        
        ## prepare motion embeddings
        motion_pos_embeds = []
        if motion_tokens is not None:
            motion_embeds = []

            for i, motion_token in enumerate(motion_tokens):
                motion_token_strs = motion_token.split('>')[:-1]
                motion_len = len(motion_token_strs)
                motion_token_ids = torch.tensor([int(x[2:]) for x in motion_token_strs], dtype=torch.long)
                motion_embed = self.motion_pretrain_embeds[motion_token_ids].to(device)
                # normalization
                motion_embed = torch.nn.functional.normalize(motion_embed, dim=-1)
                motion_embed = self.motion_proj(motion_embed)
                if self.add_motion_pos_emb and motion_trajs is not None:
                    motion_traj = motion_trajs[i]
                    motion_traj = motion_traj[:motion_len, 0, :3].unsqueeze(0) #[1, len_traj, 3]
                    motion_mask = torch.ones_like(motion_traj[:, :, 0], dtype=torch.bool)
                    # get spatial coords
                    if not self.add_scene_pos_emb:
                        scene_min, scene_max = self.get_min_max_coord(motion_traj, motion_mask)
                    else:
                        scene_min, scene_max = scene_mins[i:i+1, :], scene_maxs[i:i+1, :]
                    # get temporal coords
                    t_mins = torch.full((1, 1), 0, dtype=scene_min.dtype, device=scene_mins.device)
                    t_maxs = torch.full((1, 1), motion_len, dtype=scene_max.dtype, device=scene_maxs.device)
                    mins = torch.cat([scene_min, t_mins], dim=1)
                    maxs = torch.cat([scene_max, t_maxs], dim=1)

                    # calculate position embeddings
                    motion_pos_embed = self.pos_embedding(motion_traj, input_range=[mins, maxs]) / 10
                    motion_pos_embed = motion_pos_embed.squeeze(0)
                    motion_pos_embeds.append(motion_pos_embed)
                    motion_pos_embed = self.pos_proj(motion_pos_embed)
                    motion_embed = motion_embed + motion_pos_embed
                motion_embeds.append(motion_embed)    
        
        input_embed_list, attn_list, target_list = [], [], []
        max_seq_len = 0
        object_list_intervals = []

        activity_losses = torch.zeros(len(questions), device=device)
        contact_losses = torch.zeros(len(questions), device=device)
        position_losses = torch.zeros(len(questions), device=device)
        for i, question in enumerate(questions):
            prompt = f"{question} {self.role[1]}: "
            prompt_embed = self.get_text_emb(prompt, device=device).squeeze(0)
            
            if scene_feat is not None:
                obj_num = obj_nums[i]
                scene_embed = proj_object_embed[i][:obj_num]
                scene_pos_embed = scene_pos_embeds[i][:obj_num]
                scene_loc = scene_locs[i][:obj_num, :3]
            if motion_tokens is not None:
                motion_embed = motion_embeds[i]
                motion_len = len(motion_embed)
                if motion_trajs is not None:
                    motion_loc = motion_trajs[i, :motion_len, 0, :3].unsqueeze(0) #[1, len_traj, 3]

            if task == "scene-motion":
                # scene-motion feature fusion
                if self.scene_motion_fuse:
                    scene_fuse_embed, motion_fuse_embed = self.fusion_module(scene_embed.unsqueeze(0), motion_embed.unsqueeze(0))
                    scene_embed = self.object_post_proj(0.1 * scene_fuse_embed + scene_embed).squeeze(0)
                    motion_embed = self.motion_post_proj(0.1 * motion_fuse_embed + motion_embed).squeeze(0)
                    if self.add_scene_pos_emb:
                        scene_pos_embed = self.pos_post_proj(scene_pos_embed.unsqueeze(0)).squeeze(0)
                        scene_embed = scene_embed + scene_pos_embed
                    if self.add_motion_pos_emb:
                        motion_pos_embed = self.pos_post_proj(motion_pos_embeds[i].unsqueeze(0)).squeeze(0)
                        motion_embed = motion_embed + motion_pos_embed
                else:
                    scene_fuse_embed = scene_embed
                    motion_fuse_embed = motion_embed

                # scene-motion interaction subtask
                if self.activity_w > 0:
                    activity_preds, activity_loss = self.activity_head(scene_fuse_embed, motion_fuse_embed, scene_loc, motion_loc, activity_labels[i])
                    activity_losses[i] = activity_loss
                if self.contact_w > 0:
                    contact_label = contact_labels[i:i+1, :motion_len, :, :obj_num]
                    contact_mask = contact_masks[i:i+1, :motion_len, :, :obj_num]
                    contact_loss = self.contact_head(scene_fuse_embed, motion_fuse_embed, contact_label, contact_mask)
                    contact_losses[i] = contact_loss
                if self.position_w > 0:
                    position_label = position_labels[i:i+1, :motion_len, :obj_num]
                    position_weight = position_weights[i:i+1, :motion_len, :obj_num]
                    position_loss = self.position_head(scene_fuse_embed, motion_fuse_embed, position_label, position_weight)
                    position_losses[i] = position_loss

            elif task == "scene" and self.scene_motion_fuse:
                scene_embed = self.object_post_proj(scene_embed).squeeze(0)
                if self.add_scene_pos_emb:
                    scene_pos_embed = self.pos_post_proj(scene_pos_embed.unsqueeze(0)).squeeze(0)
                    scene_embed = scene_embed + scene_pos_embed
            elif task == "motion" and self.scene_motion_fuse:
                motion_embed = self.motion_post_proj(motion_embed).squeeze(0)

            if scene_feat is not None:
                object_list_embed = self.get_object_list_embed(
                    scene_embed, 
                    None, 
                    None, 
                    scene_mask[i][:obj_num],
                    assigned_ids[i][:obj_num]
                )

            fixed_pre_embeds = self.pre_embeds[task]
            for i in range(len(fixed_pre_embeds)):
                fixed_pre_embeds[i] = fixed_pre_embeds[i].to(device)

            if task == "scene-motion":
                pre_embed = torch.cat([fixed_pre_embeds[0], object_list_embed, 
                    fixed_pre_embeds[1], motion_embed, fixed_pre_embeds[2]])
            elif task == "scene":
                pre_embed = torch.cat([fixed_pre_embeds[0], object_list_embed, fixed_pre_embeds[1]])
            elif task == "motion":
                pre_embed = torch.cat([fixed_pre_embeds[0], motion_embed, fixed_pre_embeds[1]])
            elif task == "language":
                pre_embed = fixed_pre_embeds[0]
            else:
                raise NotImplementedError
            
            wrapped_embed = torch.cat([pre_embed, prompt_embed], dim=0)
            wrapped_attn = torch.ones(wrapped_embed.size()[:-1], dtype=torch.long).to(wrapped_embed.device)
            empty_target = (
                torch.ones(wrapped_attn.shape[0], dtype=torch.long).to(device).fill_(-100)
            )

            answer = answers[i] + self.end_sym
            to_regress_token = self.llama_tokenizer(answer, return_tensors="pt", add_special_tokens=False).to(device)
            answer_target = to_regress_token.input_ids.masked_fill(
                to_regress_token.input_ids == self.llama_tokenizer.pad_token_id, -100
            ).squeeze(0)
            to_regress_embed = self.get_text_emb(answer, device=device).squeeze(0)

            target = torch.cat([empty_target, answer_target], dim=0)
            input_embed = torch.cat([wrapped_embed, to_regress_embed], dim=0)
            attn = torch.cat([wrapped_attn, to_regress_token.attention_mask[0]], dim=0)
            input_embed_list.append(input_embed)
            attn_list.append(attn)
            target_list.append(target)
            max_seq_len = max(max_seq_len, target.shape[0])
        
        max_seq_len = min(768, max_seq_len)

        def pad_and_trim(tensor_list, max_len, batch_first=True, padding_value=0):
            padded = pad_sequence(tensor_list, batch_first=batch_first, padding_value=padding_value)
            if padded.shape[1] > max_len:
                return padded[:, :max_len]
            return padded
        
        input_embeds = pad_and_trim(input_embed_list, max_seq_len, batch_first=True, padding_value=0).to(device)
        targets = pad_and_trim(target_list, max_seq_len, batch_first=True, padding_value=-100).to(device)
        attention_mask = pad_and_trim(attn_list, max_seq_len, batch_first=True, padding_value=0).to(device)
        if self.bidirection:
            input_dtype = input_embeds.dtype
            causal_mask = torch.ones((max_seq_len, max_seq_len), dtype=input_dtype, device=device)
            causal_mask = torch.tril(causal_mask, diagonal=0)
            causal_mask = causal_mask[None, None, :, :].expand(input_embeds.shape[0], 1, -1, -1).clone()
            padding_mask = causal_mask[..., :].eq(1.0) * attention_mask[:, None, None, :].eq(0.0)
            causal_mask[..., :] = causal_mask[..., :].masked_fill(padding_mask, 0.0)
            for i in range(causal_mask.shape[0]):
                st, ed = object_list_intervals[i]
                causal_mask[i, :, st:ed, st:ed] = 1.0
            attention_mask = causal_mask

        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
                # label_weights=label_weights
            )
        
        # total loss calculation
        llm_loss = outputs.loss
        activity_loss = activity_losses.mean()
        contact_loss = contact_losses.mean()
        position_loss = position_losses.mean()
        loss = llm_loss + self.activity_w * activity_loss + self.contact_w * contact_loss + self.position_w * position_loss

        if task in ["motion", "scene-motion"]:
            motion_norm = torch.tensor([x.norm(dim=-1).mean() for x in motion_embeds]).mean().detach().cpu()
        else:
            motion_norm = 0.0

        return dict(
            loss=loss,
            llm_loss=llm_loss,
            act_loss=activity_loss,
            cont_loss=contact_loss,
            pos_loss=position_loss,
            obj_norm=proj_object_embed.norm(dim=-1).mean().detach().cpu() if proj_object_embed is not None else 0.,
            motion_norm=motion_norm,
            max_seq_len=max_seq_len
        )

    def evaluate(self,
            questions,
            ref_answers,
            scene_feat=None,
            scene_img_feat=None,
            scene_locs=None,
            scene_mask=None,
            obj_nums=None, 
            assigned_ids=None,
            motion_tokens=None,
            motion_trajs=None,
            scene_id=None,
            scene_motion_id=None,
            is_eval=True,
            type_infos=None,
            device='cuda:0',
            **kwargs
        ):
        ## define task
        if scene_feat is not None and motion_tokens is not None:
            task = "scene-motion"
        elif scene_feat is not None:
            task = "scene"
        elif motion_tokens is not None:
            task = "motion"
        else:
            task = "language"
        
        batch_size = len(questions)

        # prepare scene embeddings
        proj_object_embed = None
        scene_pos_embeds = None
        if scene_feat is not None:
            object_embed, object_img_embed = self.encode_object_feat(scene_feat, scene_img_feat, scene_locs)
            proj_object_embed = self.object_proj(object_embed)
            if self.add_scene_pos_emb:
                # get spatial coords
                scene_mins, scene_maxs = self.get_min_max_coord(scene_locs[:, :, :3], scene_mask) # [B, 3]
                # get temporal coords
                B, _ = scene_mins.shape
                t_mins = torch.zeros((B, 1), dtype=scene_mins.dtype, device=scene_mins.device)
                t_maxs = torch.zeros((B, 1), dtype=scene_maxs.dtype, device=scene_maxs.device)
                if motion_tokens is not None:
                    for i, motion_token in enumerate(motion_tokens):
                        motion_token_strs = motion_token.split('>')[:-1]
                        motion_len = len(motion_token_strs)
                        t_maxs[i].fill_(motion_len)
                else:
                    t_maxs.fill_(10.0)
                # combine spatial and temporal coords
                mins = torch.cat([scene_mins, t_mins], dim=1)
                maxs = torch.cat([scene_maxs, t_maxs], dim=1)
                
                # calculate scene position embeddings
                scene_pos_embeds = self.pos_embedding(scene_locs[:, :, :3], input_range=[mins, maxs]) / 10
                proj_pos_embeds = self.pos_proj(scene_pos_embeds)
                proj_object_embed = proj_object_embed + proj_pos_embeds
        
        ## prepare motion embeddings
        motion_pos_embeds = []
        if motion_tokens is not None:
            motion_embeds = []

            for i, motion_token in enumerate(motion_tokens):
                motion_token_strs = motion_token.split('>')[:-1]
                motion_len = len(motion_token_strs)
                motion_token_ids = torch.tensor([int(x[2:]) for x in motion_token_strs], dtype=torch.long)
                motion_embed = self.motion_pretrain_embeds[motion_token_ids].to(device)
                # normalization
                motion_embed = torch.nn.functional.normalize(motion_embed, dim=-1)
                motion_embed = self.motion_proj(motion_embed)
                if self.add_motion_pos_emb and motion_trajs is not None:
                    motion_traj = motion_trajs[i]
                    motion_traj = motion_traj[:motion_len, 0, :3].unsqueeze(0) #[1, len_traj, 3]
                    motion_mask = torch.ones_like(motion_traj[:, :, 0], dtype=torch.bool)

                    # get spatial coords
                    if not self.add_scene_pos_emb:
                        scene_min, scene_max = self.get_min_max_coord(motion_traj, motion_mask)
                    else:
                        scene_min, scene_max = scene_mins[i:i+1, :], scene_maxs[i:i+1, :]
                    # get temporal coords
                    t_mins = torch.full((1, 1), 0, dtype=scene_min.dtype, device=scene_mins.device)
                    t_maxs = torch.full((1, 1), motion_len, dtype=scene_max.dtype, device=scene_maxs.device)
                    mins = torch.cat([scene_min, t_mins], dim=1)
                    maxs = torch.cat([scene_max, t_maxs], dim=1)

                    # calculate position embeddings
                    motion_pos_embed = self.pos_embedding(motion_traj, input_range=[mins, maxs]) / 10
                    motion_pos_embed = motion_pos_embed.squeeze(0)
                    motion_pos_embeds.append(motion_pos_embed)
                    motion_pos_embed = self.pos_proj(motion_pos_embed)
                    motion_embed = motion_embed + motion_pos_embed
                motion_embeds.append(motion_embed) 

        output_texts = []
        for i in range(batch_size):
            tmp_prompt = f"{questions[i]} {self.role[1]}: "
            prompt_embed = self.get_text_emb(tmp_prompt, device=device)
            if scene_feat is not None:
                obj_num = obj_nums[i]
                scene_embed = proj_object_embed[i][:obj_num]
                scene_pos_embed = scene_pos_embeds[i][:obj_num]
            if motion_tokens is not None:
                motion_embed = motion_embeds[i]
                motion_len = len(motion_embed)

            # scene-motion feature fusion
            if self.scene_motion_fuse:
                if task == "scene-motion":
                    scene_fuse_embed, motion_fuse_embed = self.fusion_module(scene_embed.unsqueeze(0), motion_embed.unsqueeze(0))
                    scene_embed = self.object_post_proj(0.1 * scene_fuse_embed + scene_embed).squeeze(0)
                    motion_embed = self.motion_post_proj(0.1 * motion_fuse_embed + motion_embed).squeeze(0)
                    if self.add_scene_pos_emb:
                        scene_pos_embed = self.pos_post_proj(scene_pos_embed.unsqueeze(0)).squeeze(0)
                        scene_embed = scene_embed + scene_pos_embed
                    if self.add_motion_pos_emb:
                        motion_pos_embed = self.pos_post_proj(motion_pos_embeds[i].unsqueeze(0)).squeeze(0)
                        motion_embed = motion_embed + motion_pos_embed
            
                elif task == "scene":
                    scene_embed = self.object_post_proj(scene_embed).squeeze(0)
                    if self.add_scene_pos_emb:
                        scene_pos_embed = self.pos_post_proj(scene_pos_embed.unsqueeze(0)).squeeze(0)
                        scene_embed = scene_embed + scene_pos_embed
                elif task == "motion":
                    motion_embed = self.motion_post_proj(motion_embed).squeeze(0)
            
            if scene_feat is not None:
                object_list_embed = self.get_object_list_embed(
                    scene_embed, 
                    None, 
                    None, 
                    scene_mask[i][:obj_num],
                    assigned_ids[i][:obj_num]
                )
                object_list_embed = object_list_embed.unsqueeze(0)
            if motion_tokens is not None:
                motion_embed = motion_embed.unsqueeze(0)

            fixed_pre_embeds = self.pre_embeds[task]
            for i in range(len(fixed_pre_embeds)):
                fixed_pre_embeds[i] = fixed_pre_embeds[i].to(device)

            if task == "scene-motion":
                pre_embed = torch.cat([
                    fixed_pre_embeds[0].unsqueeze(0), 
                    object_list_embed, 
                    fixed_pre_embeds[1].unsqueeze(0), 
                    motion_embed, 
                    fixed_pre_embeds[2].unsqueeze(0)], 
                    dim=1)
            elif task == "scene":
                pre_embed = torch.cat([
                    fixed_pre_embeds[0].unsqueeze(0), 
                    object_list_embed, 
                    fixed_pre_embeds[1].unsqueeze(0)], 
                    dim=1)
            elif task == "motion":
                pre_embed = torch.cat([
                    fixed_pre_embeds[0].unsqueeze(0), 
                    motion_embed, 
                    fixed_pre_embeds[1].unsqueeze(0)], 
                    dim=1)
            elif task == "language":
                pre_embed = fixed_pre_embeds[0].unsqueeze(0)
            else:
                raise NotImplementedError

            wrapped_embed = torch.cat([pre_embed, prompt_embed], dim=1)
            attention_mask=None
            
            with self.maybe_autocast():
                outputs = self.llama_model.generate(
                    inputs_embeds=wrapped_embed,
                    max_new_tokens=self.max_txt_len,
                    # stopping_criteria=stopping_criteria,
                    num_beams=5,
                    # do_sample=True,
                    min_length=1,
                    # top_p=0.9,
                    repetition_penalty=3.0,
                    length_penalty=1,
                    temperature=1.0,
                    customized_mask=attention_mask
                )
            output_token = outputs[0]
            output_text = self.llama_tokenizer.decode(output_token)
            output_text = output_text.split(self.end_sym)[0]
            output_text = output_text.replace('  ', ' ').replace(' .', '.').strip()
            output_texts.append(output_text)
        return output_texts

    def forward(self, **kwargs):
        is_eval = kwargs.get("is_eval", False)
        if not is_eval:
            return self.forward_train(**kwargs)
        else:
            return self.evaluate(**kwargs)

    def _get_text_len(self, text):
        return self.llama_tokenizer(text, return_tensors="pt").input_ids.shape[1]

    def maybe_autocast(self, dtype=torch.bfloat16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    @property
    def device(self):
        return list(self.parameters())[0].device
