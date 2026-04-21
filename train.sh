export OMP_NUM_THREADS=8
# 设置显卡序号
GPU_IDS=0
Include=localhost:${GPU_IDS}  # deepspeed用的节点:卡数
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
# Stage 2
LORA_NAME=1
LangModelPara=7  # 0.5b, 7b
PROMPT_VERSION="qwen_1_5"
RUN_NAME="path-to-lora"
PREV_STAGE_CHECKPOINT="path-to-base-model"
OUTPUT_DIR="./checkpoints/onevision/${RUN_NAME}"
master_port=2950${GPU_IDS}
EPOCH=1 # 训练轮数

data_path="fsc_train_convs_merged.json"

echo "PREV_STAGE_CHECKPOINT: ${PREV_STAGE_CHECKPOINT}"
echo "MID_RUN_NAME: ${RUN_NAME}"
echo "OUTPUT_DIR: ${OUTPUT_DIR}"


# 训练lora权重
ACCELERATE_CPU_AFFINITY=1 deepspeed --include $Include --master_port=$master_port  LLaVA-NeXT/llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed LLaVA-NeXT/scripts/zero3.json \
    --model_name_or_path $PREV_STAGE_CHECKPOINT \
    --version $PROMPT_VERSION \
    --data_path $data_path \
    --image_folder data/FSC/images_384_VarV2 \
    --video_folder data/FSC/images_384_VarV2 \
    --mm_vision_tower_lr 2e-6 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres_max_9 \
    --image_grid_pinpoints  "(1x1),...,(6x6)" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name $RUN_NAME \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs $EPOCH \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --frames_upbound 32

exit 0;  
