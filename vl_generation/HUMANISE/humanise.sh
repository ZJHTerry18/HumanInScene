python vl_generation/HUMANISE/gpt_humanise.py \
    --root /home/zhaojiaohe/data/zhaojiahe/HISLLM-humanise \
    --output vl_generation/output \
    --task caption 

python vl_generation/HUMANISE/gpt_humanise.py \
    --root /home/zhaojiaohe/data/zhaojiahe/HISLLM-humanise \
    --output vl_generation/output \
    --task activity

python vl_generation/HUMANISE/gpt_humanise.py \
    --root /home/zhaojiaohe/data/zhaojiahe/HISLLM-humanise \
    --output vl_generation/output \
    --task hoi

python vl_generation/HUMANISE/gpt_humanise.py \
    --root /home/zhaojiaohe/data/zhaojiahe/HISLLM-humanise \
    --output vl_generation/output \
    --task hoi_interaction

python vl_generation/HUMANISE/gpt_humanise.py \
    --root /home/zhaojiaohe/data/zhaojiahe/HISLLM-humanise \
    --output vl_generation/output \
    --task hoi_object

python vl_generation/HUMANISE/gpt_humanise.py \
    --root /home/zhaojiaohe/data/zhaojiahe/HISLLM-humanise \
    --output vl_generation/output \
    --task hoi_part

python vl_generation/HUMANISE/gpt_humanise.py \
    --root /home/zhaojiaohe/data/zhaojiahe/HISLLM-humanise \
    --output vl_generation/output \
    --task loc --w_far --all_ref

python vl_generation/HUMANISE/gpt_humanise.py \
    --root /home/zhaojiaohe/data/zhaojiahe/HISLLM-humanise \
    --output vl_generation/output \
    --task loc_object --w_far --all_ref

python vl_generation/HUMANISE/gpt_humanise.py \
    --root /home/zhaojiaohe/data/zhaojiahe/HISLLM-humanise \
    --output vl_generation/output \
    --task loc_orient --w_far --all_ref

python vl_generation/HUMANISE/gpt_humanise.py \
    --root /home/zhaojiaohe/data/zhaojiahe/HISLLM-humanise \
    --output vl_generation/output \
    --task loc_position --all_ref

python vl_generation/HUMANISE/gpt_humanise.py \
    --root /home/zhaojiaohe/data/zhaojiahe/HISLLM-humanise \
    --output vl_generation/output \
    --task pred

python vl_generation/HUMANISE/gpt_humanise.py \
    --root /home/zhaojiaohe/data/zhaojiahe/HISLLM-humanise \
    --output vl_generation/output_humanise_v1_qa \
    --task pred_intent

python vl_generation/HUMANISE/gpt_humanise.py \
    --root /home/zhaojiaohe/data/zhaojiahe/HISLLM-humanise \
    --output vl_generation/output_humanise_v1_qa \
    --task pred_movement

python vl_generation/HUMANISE/gpt_humanise.py \
    --root /home/zhaojiaohe/data/zhaojiahe/HISLLM-humanise \
    --output vl_generation/output \
    --task planning

python vl_generation/HUMANISE/gpt_humanise.py \
    --root /home/zhaojiaohe/data/zhaojiahe/HISLLM-humanise \
    --output vl_generation/output \
    --task dialogue

python vl_generation/HUMANISE/gpt_humanise.py \
    --root /home/zhaojiaohe/data/zhaojiahe/HISLLM-humanise \
    --output vl_generation/output \
    --task open --w_far --all_ref