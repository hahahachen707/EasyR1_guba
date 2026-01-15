#!/bin/bash

set -x

# MODEL_PATH=Qwen/Qwen3-VL-4B-Instruct  # replace it with your local file path
MODEL_PATH=/home/tione/notebook/workspace/xiaoyangchen/work/EasyR1/checkpoints/llama_factory/qwen3vl_lora_sft

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=/home/tione/notebook/workspace/xiaoyangchen/work/data/aha/aha_train_verl.jsonl \
    data.val_files=/home/tione/notebook/workspace/xiaoyangchen/work/data/aha/aha_eval_verl.jsonl \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=qwen3_vl_4b_aha_grpo \
    worker.reward.reward_function=./examples/reward_function/reward_aha.py:compute_score \
    trainer.n_gpus_per_node=4
