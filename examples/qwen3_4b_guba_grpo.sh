#!/bin/bash

set -x

MODEL_PATH=Qwen/Qwen3-VL-4B-Instruct  # replace it with your local file path

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.max_prompt_length=4096 \
    data.max_response_length=4096 \
    data.train_files=/home/tione/notebook/workspace/xiaoyangchen/work/data/guba/guba_train_verl.jsonl \
    data.val_files=/home/tione/notebook/workspace/xiaoyangchen/work/data/guba/guba_eval_verl.jsonl \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=qwen3_4b_instruct_guba_grpo \
    worker.reward.reward_function=./examples/reward_function/reward_guba.py:compute_score \
    trainer.n_gpus_per_node=4
