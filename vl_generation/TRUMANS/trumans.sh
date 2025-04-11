python vl_generation/TRUMANS/gpt_trumans.py \
    --root /home/zhaojiaohe/data/zhaojiahe/HISLLM-trumans \
    --output vl_generation/output \
    --task caption 

python vl_generation/TRUMANS/gpt_trumans.py \
    --root /home/zhaojiaohe/data/zhaojiahe/HISLLM-trumans \
    --output vl_generation/output \
    --task activity

python vl_generation/TRUMANS/gpt_trumans.py \
    --root /home/zhaojiaohe/data/zhaojiahe/HISLLM-trumans \
    --output vl_generation/output \
    --task hoi_interaction

python vl_generation/TRUMANS/gpt_trumans.py \
    --root /home/zhaojiaohe/data/zhaojiahe/HISLLM-trumans \
    --output vl_generation/output \
    --task hoi_object

python vl_generation/TRUMANS/gpt_trumans.py \
    --root /home/zhaojiaohe/data/zhaojiahe/HISLLM-trumans \
    --output vl_generation/output \
    --task hoi_part

python vl_generation/TRUMANS/gpt_trumans.py \
    --root /home/zhaojiaohe/data/zhaojiahe/HISLLM-trumans \
    --output vl_generation/output \
    --task loc --w_far --all_ref

python vl_generation/TRUMANS/gpt_trumans.py \
    --root /home/zhaojiaohe/data/zhaojiahe/HISLLM-trumans \
    --output vl_generation/output \
    --task loc_object --w_far --all_ref

python vl_generation/TRUMANS/gpt_trumans.py \
    --root /home/zhaojiaohe/data/zhaojiahe/HISLLM-trumans \
    --output vl_generation/output \
    --task loc_orient --w_far --all_ref

python vl_generation/TRUMANS/gpt_trumans.py \
    --root /home/zhaojiaohe/data/zhaojiahe/HISLLM-trumans \
    --output vl_generation/output \
    --task loc_position --all_ref

python vl_generation/TRUMANS/gpt_trumans.py \
    --root /home/zhaojiaohe/data/zhaojiahe/HISLLM-trumans \
    --output vl_generation/output \
    --task pred

python vl_generation/TRUMANS/gpt_trumans.py \
    --root /home/zhaojiaohe/data/zhaojiahe/HISLLM-trumans \
    --output vl_generation/output \
    --task pred_intent

python vl_generation/TRUMANS/gpt_trumans.py \
    --root /home/zhaojiaohe/data/zhaojiahe/HISLLM-trumans \
    --output vl_generation/output \
    --task pred_movement

python vl_generation/TRUMANS/gpt_trumans.py \
    --root /home/zhaojiaohe/data/zhaojiahe/HISLLM-trumans \
    --output vl_generation/output \
    --task dialogue

python vl_generation/TRUMANS/gpt_trumans.py \
    --root /home/zhaojiaohe/data/zhaojiahe/HISLLM-trumans \
    --output vl_generation/output \
    --task open --w_far --all_ref