from openai import OpenAI
import json
import glob
import os
import os.path as osp
from collections import defaultdict
import random
from tqdm import tqdm
import argparse
import numpy as np
import pyrootutils
pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)

from api_keys import GPT_API_KEY, QWEN_API_KEY
from vl_generation.templates import *
from vl_generation.generate_funcs import *
from vl_generation.prompts import *

task2tools = {
    "caption": (
        generate_cap_allinfo, 
        ALL_INFO_INPUT_TEMPLATE, 
        None, 
        DENSE_CAPTION_PROMPT, 
        'gpt_prox_cap_incontext.json'),
    "activity": (
        generate_qa_action, 
        ACTION_INPUT_TEMPLATE, 
        QA_OUTPUT_TEMPLATE,
        QA_ACTION_PROMPT.format(ACTIVITY),
        'qa_activity.json'),
    "hoi": (
        generate_qa_allinfo,
        ALL_INFO_INPUT_TEMPLATE,
        QA_OUTPUT_TEMPLATE,
        QA_ALLINFO_PROMPT.format(HOI),
        'qa_hoi.json'),
    "hoi_interaction": (
        generate_qa_action,
        ACTION_INPUT_TEMPLATE, 
        QA_OUTPUT_TEMPLATE,
        QA_ACTION_PROMPT.format(HOI_INTERACTION),
        'qa_hoi_interaction.json'),
    "hoi_object": (
        generate_qa_allinfo,
        ALL_INFO_INPUT_TEMPLATE,
        QA_OUTPUT_TEMPLATE,
        QA_ALLINFO_PROMPT.format(HOI_OBJECT),
        'qa_hoi_object.json'),
    "hoi_part": (
        generate_qa_allinfo,
        ALL_INFO_INPUT_TEMPLATE,
        QA_OUTPUT_TEMPLATE,
        QA_ALLINFO_PROMPT.format(HOI_PART),
        'qa_hoi_part.json'),
    "loc": (
        generate_qa_allinfo,
        ALL_INFO_INPUT_TEMPLATE,
        QA_OUTPUT_TEMPLATE,
        QA_ALLINFO_PROMPT.format(LOC),
        'qa_loc.json'),
    "loc_object": (
        generate_qa_allinfo,
        ALL_INFO_INPUT_TEMPLATE,
        QA_OUTPUT_TEMPLATE,
        QA_ALLINFO_PROMPT.format(LOC_OBJECT),
        'qa_loc_object.json'),
    "loc_orient": (
        generate_qa_allinfo,
        ALL_INFO_INPUT_TEMPLATE,
        QA_OUTPUT_TEMPLATE,
        QA_ALLINFO_PROMPT.format(LOC_ORIENT),
        'qa_loc_orient.json'),
    "loc_position": (
        generate_qa_allinfo,
        ALL_INFO_INPUT_TEMPLATE,
        QA_OUTPUT_TEMPLATE,
        QA_ALLINFO_PROMPT.format(LOC_POSITION),
        'qa_loc_position.json'),
    "pred": (
        generate_pred_allinfo,
        PRED_INPUT_TEMPLATE,
        QA_OUTPUT_TEMPLATE,
        PRED_PROMPT.format(PRED),
        'pred.json'),
    "pred_intent": (
        generate_pred_allinfo,
        PRED_INPUT_TEMPLATE,
        QA_OUTPUT_TEMPLATE,
        PRED_PROMPT.format(PRED_INTENT),
        'qa_pred_intent.json'),
    "pred_movement": (
        generate_pred_allinfo,
        PRED_INPUT_TEMPLATE,
        QA_OUTPUT_TEMPLATE,
        PRED_PROMPT.format(PRED_MOVEMENT),
        'qa_pred_movement.json'),
    "planning": (
        generate_qa_allinfo,
        ALL_INFO_INPUT_TEMPLATE,
        QA_OUTPUT_TEMPLATE,
        QA_ALLINFO_PROMPT.format(PLANNING),
        'planning.json'),
    "dialogue": (
        generate_dialogue_allinfo,
        ALL_INFO_INPUT_TEMPLATE,
        DIALOGUE_OUTPUT_TEMPLATE,
        QA_ALLINFO_PROMPT.format(DIALOGUE),
        'dialogue.json'),
    "open": (
        generate_qa_allinfo,
        ALL_INFO_INPUT_TEMPLATE,
        QA_OUTPUT_TEMPLATE,
        QA_ALLINFO_PROMPT.format(OPEN),
        'gpt_prox_qa_incontext.json')
}

scene_ref_files = [
    'scene_ref_multiobj.json',
    'scene_ref_pairwise.json',
    'scene_ref_star.json'
]

# gpt_model = 'qwen-plus'
# my_api_key = QWEN_API_KEY
# client = OpenAI(
#     api_key=my_api_key,
#     base_url='https://dashscope.aliyuncs.com/compatible-mode/v1'
# )

gpt_model = 'gpt-4o-mini'
my_api_key = GPT_API_KEY
client = OpenAI(
    api_key=my_api_key,
)

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True, help='root of all annotations')
    parser.add_argument('--output', type=str, default='output', help='directory of outputing the final file')
    parser.add_argument('--task', type=str, help='type of specific task to generate')
    parser.add_argument('--sample_n', type=int, default=5, help='number of keyframes to sample')
    parser.add_argument('--w_dist', action='store_true', help='use distance information of human positions')
    parser.add_argument('--w_far', action='store_true', help='include positions of far objects')
    parser.add_argument('--n_ref', type=int, default=3, help='number of references for each object')
    parser.add_argument('--all_ref', action='store_true', help='include references of all objects')
    parser.add_argument('--pre_n', type=int, default=3, help='length for current motion (the rest are used as predictions)')

    # start and end index for partial generation of the whole dataset
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=100000)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = argparser()

    motion_dir = osp.join(args.root, 'motions')
    scene_dir = osp.join(args.root, 'scenes')
    sceneref_dir = osp.join(args.root, 'scenerefer')
    pose_dir = osp.join(args.root, 'posescripts')
    metadata_file = osp.join(args.root, 'metadata.json')

    ## load scene reference expressions
    scene2ref = defaultdict(list)
    for reffile in scene_ref_files:
        reffile = osp.join(sceneref_dir, reffile)
        if reffile.endswith('.json'):
            with open(reffile, 'r') as f:
                refdata = json.load(f)
        elif reffile.endswith('.jsonl'):
            refdata = []
            with open(reffile, 'r') as f:
                for line in f:
                    refdata.append(json.loads(line))
        
        for rd in refdata:
            scene_id = rd["scan_id"]
            scene2ref[scene_id].append(rd)

    ## load scene-motion annotations
    all_motions_list = sorted(os.listdir(motion_dir))[args.start:args.end:500]
    all_motion_caps = []

    gen_func, input_template, output_template, system_prompt, in_context_file = \
        task2tools[args.task]
    ## load in-context examples
    with open(f"vl_generation/in_context_examples/{in_context_file}", 'r') as f:
        in_context_examples = json.load(f)

    output_file = osp.join(args.output, f'trumans_{args.task}.jsonl')
    os.makedirs(args.output, exist_ok=True)
    with open(output_file, 'a+') as fo:
        for ind, motion_id in tqdm(enumerate(all_motions_list), total=len(all_motions_list)):
            ## load motion annotations
            motion_folder = osp.join(motion_dir, motion_id)
            with open(osp.join(motion_folder, 'keyframe_annotations.json'), 'r') as fp:
                motion_dat = json.load(fp)

            human_scene_contacts = {k: v["contacts"] for k, v in motion_dat.items()}
            human_positions = {k: v["pos"] for k, v in motion_dat.items()}
            if not args.w_dist:
                for k in human_positions:
                    human_positions[k] = [x[:2] + x[3:] for x in human_positions[k]]
            if not args.w_far:
                for k in human_positions:
                    human_positions[k] = [x for x in human_positions[k] if x[0] != 'far']
            elif random.random() < 0.5:
                for k in human_positions:
                    human_positions[k] = [x for x in human_positions[k] if x[0] != 'far']


            ## load pose annotations
            full_motion_id = motion_id.split('_')[1]
            pose_file = osp.join(pose_dir, full_motion_id + '.json')
            with open(pose_file, 'r') as fp:
                pose_dat = json.load(fp)

            ## load action annotations
            with open(metadata_file, 'r') as fp:
                metadata = json.load(fp)
            motionindex2ann = {f"{v['scene']}_{v['motion'][:-4]}": v["action"] for v in metadata}
            action_dat = motionindex2ann[motion_id]

            ## sample keyframes
            motion_frame_ids = list(motion_dat.keys())

            ## aggregate all motion annotations
            motion_annotations = {}
            motion_annotations['action'] = action_dat
            if args.task in ['pred', 'pred_intent', 'pred_movement']:
                motion_annotations['current key moments'] = []
                motion_annotations['future key moments'] = []
            else:
                motion_annotations['key moments'] = []
            interact_obj_ids = set()
            for i, k in enumerate(motion_frame_ids):
                keydat = {
                    'pose': pose_dat[k],
                    'human-scene contacts': human_scene_contacts[k],
                    'human-scene spatial relations': human_positions[k]
                }
                if args.task in ['pred', 'pred_intent', 'pred_movement']:
                    if i < len(motion_frame_ids) * 0.6:
                        motion_annotations['current key moments'].append(keydat)
                    else:
                        motion_annotations['future key moments'].append(keydat)
                else:
                    motion_annotations['key moments'].append(keydat)
                for c in human_scene_contacts[k]:
                    interact_obj_ids.add(c[-1])
                for p in human_positions[k]:
                    interact_obj_ids.add(p[-1])
            if args.task in ['pred', 'pred_intent', 'pred_movement'] and len(motion_annotations['future key moments']) == 0:
                continue

            ## load scene annotations
            scene_id = motion_id.split('_')[0]
            scene_folder = osp.join(scene_dir, scene_id)
            with open(osp.join(scene_folder, 'scene_objects.json'), 'r') as f:
                scene_object_semantics = json.load(f)
            
            scene_refs = defaultdict(list)
            ref_list = scene2ref[scene_id]
            for ref in ref_list:
                scene_refs[ref['target_id']].append(ref['utterance'])
            
            ## aggregate scene annotations
            if args.all_ref:
                object_id_list = list(scene_object_semantics.keys())
            else:
                object_id_list = list(interact_obj_ids)
            
            scene_annotations = {}
            scene_annotations['object information'] = {}
            for obj_id in object_id_list:
                obj_category = scene_object_semantics[str(obj_id)]
                obj_ref = scene_refs[obj_id]
                if len(obj_ref) > args.n_ref:
                    obj_ref = random.sample(obj_ref, args.n_ref)
                scene_annotations['object information'][obj_id] = {
                    'category': obj_category,
                    'referral': obj_ref
                }
            

            ## generate motion-in-scene caption
            try:
                output = gen_func(
                    motion_anns=motion_annotations,
                    scene_anns=scene_annotations,
                    in_context_examples=in_context_examples,
                    input_template=input_template,
                    output_template=output_template,
                    system_prompt=system_prompt,
                    client=client,
                    gpt_model=gpt_model
                )
            except Exception as e:
                print(f"{motion_id} has error occured when generating: {e}. Skip generation.")
                continue

            out_data = {
                'task': args.task,
                'index': ind,
                'data_id': motion_id,
                'scene_id': scene_id,
                'motion_id': full_motion_id,
                'text': output,
            }

            ## save generated caption
            line = json.dumps(out_data)
            fo.write(line + '\n')