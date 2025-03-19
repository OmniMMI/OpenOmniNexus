#!/bin/bash
#SBATCH --job-name=preproc
#SBATCH --partition=HGX,DGX
#SBATCH --account=research
#SBATCH --qos=lv0a
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --output=tts.out
#SBATCH --error=tts.error.out

export PYTHONPATH=third_party/Matcha-TTS
module load cuda11.8

# 批次数量
num_batches=5


gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

# Python 脚本路径
python_script="./process_cosyvoice.py"

python3 $python_script --batch_id 1 --num_batch $num_batches
python3 $python_script --batch_id 2 --num_batch $num_batches
python3 $python_script --batch_id 3 --num_batch $num_batches
python3 $python_script --batch_id 4 --num_batch $num_batches
python3 $python_script --batch_id 5 --num_batch $num_batches