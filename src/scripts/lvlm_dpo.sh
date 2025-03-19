#!/bin/bash
#SBATCH --job-name=DPO
#SBATCH --partition=HGX,DGX
#SBATCH --account=research
#SBATCH --qos=lv1
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --output=./slurm_logs/dpo-llama3.1-sharevideo.out
#SBATCH --error=./slurm_logs/dpo-llama3.1-sharevideo.error.out


export OMP_NUM_THREADS=4
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
# export NCCL_IB_HCA=${ARNOLD_RDMA_DEVICE}
# export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO

export NUM_GPUS=4
MASTER_PORT=$(expr $RANDOM + 1000)
export PORT=$MASTER_PORT

export PYTHONPATH=$(pwd)
echo $PYTHONPATH

LLM_VERSION="checkpoints/Meta-Llama-3.1-8B-Instruct"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="checkpoints/clip-vit-large-patch14-336"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

############### Pretrain ################

# Stage 2
PROMPT_VERSION="llava_llama_3"

BASE_RUN_NAME="LongVA-7B-Llama31"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

MID_RUN_NAME="LongVA-7B-Llama31-DPO"
echo "MID_RUN_NAME: ${MID_RUN_NAME}"

CKPT_PATH="checkpoints/${BASE_RUN_NAME}" # this could also be the previous stage checkpoint

module add cuda11.8

#torchrun --nproc_per_node="${ARNOLD_WORKER_GPU}" --nnodes="${ARNOLD_WORKER_NUM}" --node_rank="${ARNOLD_ID}" --master_addr="${METIS_WORKER_0_HOST}" --master_port="${port_in_cmd}" \
# ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="${ARNOLD_WORKER_GPU}" --nnodes="${ARNOLD_WORKER_NUM}" --node_rank="${ARNOLD_ID}" --master_addr="${METIS_WORKER_0_HOST}" --master_port="${port_in_cmd}" \
ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="${NUM_GPUS}" --master_port="${PORT}" \
    longva/train/train_dpo.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path ${CKPT_PATH} \
    --version $PROMPT_VERSION \
    --dpo_alpha 1.0 --beta 0.1 --gamma 0 \
    --data_path="inputs/texts/sft_dpo_17k.jsonl" \
    --image_folder /data/llava_data \
    --video_folder inputs/videos/shareVideoGPTV/ \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --mm_spatial_pool_stride 2 \
    --mm_resampler_type "spatial_pool" \
    --mm_spatial_pool_out_channels 1024 \
    --group_by_modality_length True \
    --image_aspect_ratio anyres \
    --image_grid_pinpoints "[(336, 672), (672, 336), (672, 672), (1008, 336), (336, 1008)]" \
    --mm_patch_merge_type unires \
    --bf16 True \
    --run_name $MID_RUN_NAME \
    --output_dir "checkpoints/${MID_RUN_NAME}" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 3000 \
    --save_total_limit 1 \
    --learning_rate 5e-7 \
    --weight_decay 0. \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "linear" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    # --attn_implementation sdpa