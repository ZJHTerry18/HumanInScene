which_python=$(which python)
export PYTHONPATH=${PYTHONPATH}:${which_python}:.
echo "PYTHONPATH: ${PYTHONPATH}"

export MASTER_PORT=$((54002 + $RANDOM % 10000))
export MASTER_ADDR=localhost

train_tag="humanise"
val_tag="humanise#hisbench"

evaluate=False
resume=False
debug=False
if [ $debug = "True" ]; then
    enable_wandb=False
    gpu_num=1
    do_save=False
    other_info="debug"
else
    enable_wandb=False
    gpu_num=4
    do_save=True
    other_info="scratch"
fi

tag="${train_tag}__${val_tag}__${other_info}"

if [[ $evaluate = "True" || $resume = "True" ]]; then
    OUTPUT_DIR=outputs/"20250213_162627_humanise__humanise#hisbench__scratch" # self-define
else
    OUTPUT_DIR=outputs/"$(date +"%Y%m%d_%H%M%S")"_"$tag"
    mkdir -p ${OUTPUT_DIR}
fi
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=${gpu_num} --master_port=${MASTER_PORT}  tasks/train.py \
    "configs/human_scene_it.yaml" \
    output_dir "$OUTPUT_DIR" \
    debug "$debug" \
    evaluate "$evaluate" \
    gpu_num "$gpu_num"