import torch
from torch.nn.utils.rnn import pad_sequence

def scene_train_collate_fn(batch):
    scene_feats, scene_img_feats, scene_masks, scene_locs, obj_nums, assigned_ids, captions, questions = zip(*batch)
    batch_scene_feat = pad_sequence(scene_feats, batch_first=True)
    batch_scene_img_feat = pad_sequence(scene_img_feats, batch_first=True)
    batch_scene_mask = pad_sequence(scene_masks, batch_first=True).to(torch.bool)
    batch_scene_locs = pad_sequence(scene_locs, batch_first=True)
    batch_obj_nums = pad_sequence(obj_nums, batch_first=True)
    batch_assigned_ids = pad_sequence(assigned_ids, batch_first=True)
    # batch_detach_mask = torch.ones_like(batch_scene_mask, dtype=torch.bool)
    # for i in range(batch_detach_mask.shape[0]):
    #     batch_detach_mask[i][:detach_masks[i].shape[0]] = detach_masks[i]
    return {
        "scene_feat": batch_scene_feat,
        "scene_img_feat": batch_scene_img_feat,
        "scene_locs": batch_scene_locs,
        "scene_mask": batch_scene_mask,
        "assigned_ids": batch_assigned_ids,
        # "detach_mask": batch_detach_mask,
        "obj_nums": batch_obj_nums,
        "answers": captions,
        "questions": questions
        # "ref_captions": ref_captions,
        # "ids": index
    }

def scene_val_collate_fn(batch):
    scene_feats, scene_img_feats, scene_masks, scene_locs, obj_nums, assigned_ids, questions, ref_answers, scene_ids, qids, pred_ids, type_infos = zip(*batch)
    batch_scene_feat = pad_sequence(scene_feats, batch_first=True)
    batch_scene_img_feat = pad_sequence(scene_img_feats, batch_first=True)
    batch_scene_mask = pad_sequence(scene_masks, batch_first=True).to(torch.bool)
    batch_scene_locs = pad_sequence(scene_locs, batch_first=True)
    batch_obj_nums = pad_sequence(obj_nums, batch_first=True)
    batch_assigned_ids = pad_sequence(assigned_ids, batch_first=True)
    pred_ids = torch.tensor(pred_ids)
    return {
        "scene_feat": batch_scene_feat,
        "scene_img_feat": batch_scene_img_feat,
        "scene_locs": batch_scene_locs,
        "scene_mask": batch_scene_mask,
        "assigned_ids": batch_assigned_ids,
        "obj_nums": batch_obj_nums,
        "questions": questions,
        "ref_answers": ref_answers,
        "scene_id": scene_ids,
        "qid": qids,
        "pred_ids": pred_ids,
        "type_infos": type_infos
        # "ids": index
    }

def scene_motion_train_collate_fn(batch):
    scene_feats, scene_img_feats, scene_masks, scene_locs, scene_obj_labels, \
        obj_nums, assigned_ids, motion_tokens, motion_trajs, \
        activity_labels, contact_labels, contact_masks, position_labels, position_weights, \
        answers, questions = zip(*batch)
    batch_scene_feat = pad_sequence(scene_feats, batch_first=True)
    batch_scene_img_feat = pad_sequence(scene_img_feats, batch_first=True)
    batch_scene_mask = pad_sequence(scene_masks, batch_first=True).to(torch.bool)
    batch_scene_locs = pad_sequence(scene_locs, batch_first=True)
    batch_obj_nums = pad_sequence(obj_nums, batch_first=True)
    batch_assigned_ids = pad_sequence(assigned_ids, batch_first=True)
    # batch_motion_tokens = pad_sequence(motion_tokens, batch_first=True)
    batch_motion_trajs = pad_sequence(motion_trajs, batch_first=True)
    batch_activity_labels = pad_sequence(activity_labels, batch_first=True)
    batch_contact_labels = pad_sequence(contact_labels, batch_first=True)
    batch_contact_masks = pad_sequence(contact_masks, batch_first=True)
    batch_position_labels = pad_sequence(position_labels, batch_first=True)
    batch_position_weights = pad_sequence(position_weights, batch_first=True)
    # batch_detach_mask = torch.ones_like(batch_scene_mask, dtype=torch.bool)
    # for i in range(batch_detach_mask.shape[0]):
    #     batch_detach_mask[i][:detach_masks[i].shape[0]] = detach_masks[i]
    return {
        "scene_feat": batch_scene_feat,
        "scene_img_feat": batch_scene_img_feat,
        "scene_locs": batch_scene_locs,
        "scene_mask": batch_scene_mask,
        "obj_nums": batch_obj_nums,
        "assigned_ids": batch_assigned_ids,
        # "detach_mask": batch_detach_mask,
        "motion_tokens": motion_tokens,
        "motion_trajs": batch_motion_trajs,
        "activity_labels": batch_activity_labels,
        "contact_labels": batch_contact_labels,
        "contact_masks": batch_contact_masks,
        "position_labels": batch_position_labels,
        "position_weights": batch_position_weights,
        "answers": answers,
        "questions": questions,
        # "ref_captions": ref_captions,
        # "ids": index
    }

def scene_motion_val_collate_fn(batch):
    scene_feats, scene_img_feats, scene_masks, scene_locs, obj_nums, assigned_ids, \
        motion_tokens, motion_trajs, questions, ref_answers, scene_ids, scene_motion_ids, qids, pred_ids, type_infos = zip(*batch)
    batch_scene_feat = pad_sequence(scene_feats, batch_first=True)
    batch_scene_img_feat = pad_sequence(scene_img_feats, batch_first=True)
    batch_scene_mask = pad_sequence(scene_masks, batch_first=True).to(torch.bool)
    batch_scene_locs = pad_sequence(scene_locs, batch_first=True)
    batch_obj_nums = pad_sequence(obj_nums, batch_first=True)
    batch_assigned_ids = pad_sequence(assigned_ids, batch_first=True)
    batch_motion_trajs = pad_sequence(motion_trajs, batch_first=True)
    pred_ids = torch.tensor(pred_ids)
    return {
        "scene_feat": batch_scene_feat,
        "scene_img_feat": batch_scene_img_feat,
        "scene_locs": batch_scene_locs,
        "scene_mask": batch_scene_mask,
        "obj_nums": batch_obj_nums,
        "assigned_ids": batch_assigned_ids,
        "motion_tokens": motion_tokens,
        "motion_trajs": batch_motion_trajs,
        "questions": questions,
        "ref_answers": ref_answers,
        "scene_id": scene_ids,
        "scene_motion_id": scene_motion_ids,
        "qid": qids,
        "pred_ids": pred_ids,
        "type_infos": type_infos
        # "ids": index
    }

def motion_train_collate_fn(batch):
    motion_tokens, answers, questions = zip(*batch)
    return {
        "motion_tokens": motion_tokens,
        "answers": answers,
        "questions": questions,
        # "ref_captions": ref_captions,
        # "ids": index
    }

def motion_val_collate_fn(batch):
    motion_tokens, questions, ref_answers, motion_ids, qids, pred_ids, type_infos = zip(*batch)
    pred_ids = torch.tensor(pred_ids)
    return {
        "motion_tokens": motion_tokens,
        "questions": questions,
        "ref_answers": ref_answers,
        "scene_motion_id": motion_ids,
        "qid": qids,
        "pred_ids": pred_ids,
        "type_infos": type_infos
        # "ids": index
    }

def language_train_collate_fn(batch):
    answers, questions = zip(*batch)
    return {
        "answers": answers,
        "questions": questions,
        # "ref_captions": ref_captions,
        # "ids": index
    }

def language_val_collate_fn(batch):
    questions, ref_answers, qids, pred_ids, type_infos = zip(*batch)
    pred_ids = torch.tensor(pred_ids)
    return {
        "questions": questions,
        "ref_answers": ref_answers,
        "qid": qids,
        "pred_ids": pred_ids,
        "type_infos": type_infos
        # "ids": index
    }