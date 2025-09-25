#!/bin/bash


export VLLM_ATTENTION_BACKEND=XFORMERS
export HYDRA_FULL_ERROR=1
export project_name=""
export RAY_memory_monitor_refresh_ms=0
export experiment_name=""
export PET_NODE_RANK=0
export MODEL_PATH=""
export OUTPUT=''
export EVAL_LOG_PATH=''
mkdir -p $OUTPUT
mkdir -p $EVAL_LOG_PATH

    
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=./data/train.parquet \
    data.val_files=./data/dev.parquet \
    data.train_batch_size=32 \
    data.max_prompt_length=30767 \
    data.max_response_length=2000 \
    +data.max_model_len=32768 \
    +data.data_writing_path=oss://yzz123456/temp/data_debug/ \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.65 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
	actor_rollout_ref.rollout.max_model_len=32768 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.use_dynamic_bsz=true \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=12288 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=4 \
    actor_rollout_ref.rollout.temperature=1.0 \
    critic.optim.lr=1e-5 \
    critic.model.path=${MODEL_PATH} \
    critic.ppo_micro_batch_size_per_gpu=2 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.val_before_train=false \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=1 \
    trainer.test_freq=1 \
    trainer.validation_data_dir=${EVAL_LOG_PATH} \
    trainer.default_local_dir=${OUTPUT} \
    agent_grpo.n=16 \
    max_turns=10 \
    search_engine=online_search \
    codeact_env_disabled=true \
    trainer.total_epochs=1 2>&1 | tee ./${project_name}_${experiment_name}.log

