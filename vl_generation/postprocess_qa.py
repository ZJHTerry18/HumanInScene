#################
# Convert the output jsonl into separate questions
#################
import json
import os
import random
random.seed(42)

src_dir = "output_humanise_v1_pred"
tgt_dir = "output_humanise_v1_pred_qa"
os.makedirs(tgt_dir, exist_ok=True)

caption_instruction_template = "/home/zhaojiaohe/codes/Human-in-scene-LLM/prompts/detailed_scenemotion_description.txt"
caption_instructions = []
with open(caption_instruction_template, 'r') as f:
    for line in f:
        caption_instructions.append(line.strip())

src_jsonls = os.listdir(src_dir)
for src_file in src_jsonls:
    src_datas = []
    with open(os.path.join(src_dir, src_file), 'r') as f:
        for line in f:
            src_datas.append(json.loads(line))
    
    tgt_datas = []
    index = 0
    for sd in src_datas:
        texts = sd["text"]
        task = sd["task"]
        if sd["task"] == "dialogue":
            td = dict()
            td["task"] = sd["task"]
            td["index"] = index
            td["data_id"] = sd["data_id"]
            td["scene_id"] = sd["scene_id"]
            td["motion_id"] = sd["motion_id"]
            td["qa"] = texts

            tgt_datas.append(td)
            index += 1
        elif sd["task"] == "caption":
            td = dict()
            td["task"] = sd["task"]
            td["index"] = index
            td["data_id"] = sd["data_id"]
            td["scene_id"] = sd["scene_id"]
            td["motion_id"] = sd["motion_id"]
            td["qa"] = [
                {
                    "question": random.choice(caption_instructions),
                    "answer": texts,
                }
            ]

            tgt_datas.append(td)
            index += 1
        else:
            for t in texts:
                td = dict()
                td["task"] = sd["task"]
                td["index"] = index
                td["data_id"] = sd["data_id"]
                td["scene_id"] = sd["scene_id"]
                td["motion_id"] = sd["motion_id"]
                td["qa"] = [t]

                tgt_datas.append(td)
                index += 1
    
    tgt_path = os.path.join(tgt_dir, src_file[:-1])
    with open(tgt_path, 'w') as f:
        json.dump(tgt_datas, f)
    
    print(f"Task: {task}, Question number: {len(tgt_datas)}")