#!/usr/bin/env bash
set -euo pipefail
set -x

# LiteGUI-RL is a ClawGUI-RL/verl recipe, not a separate trainer.
cd "$(dirname "$0")"

# ============ Paths: override through environment variables ============
MODEL_PATH=${MODEL_PATH:-Tongyi-MAI/MAI-UI-2B}
DATA_ROOT=${DATA_ROOT:-$HOME/data/mw_online_rl}
DATA_DIR=$DATA_ROOT/visual
GEOMETRY3K_DIR=${GEOMETRY3K_DIR:-$HOME/data/geometry3k}
SERVER_FILE=${SERVER_FILE:-../env_server/mobileworld_server.txt}
TASK_FILE=${TASK_FILE:-../env_server/mobileworld_tasks.xlsx}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-$HOME/checkpoints/litegui_rl_2b}
STATS_PATH=${STATS_PATH:-$CHECKPOINT_DIR/difficulty_stats.json}
DIFFICULTY_FILE=${DIFFICULTY_FILE:-task_difficulty.json}

# ============ Low-cost 2B recipe ============
N_GPUS=${N_GPUS:-2}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-2}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-1}
GROUP_SIZE=${GROUP_SIZE:-4}
MAX_STEPS=${MAX_STEPS:-15}
HISTORY_LENGTH=${HISTORY_LENGTH:-3}
TOTAL_CURRICULUM_EPOCHS=${TOTAL_CURRICULUM_EPOCHS:-2}
HARD_TASK_NUM=${HARD_TASK_NUM:-15}
LORA_RANK=${LORA_RANK:-16}

# The server file must contain at least TRAIN_BATCH_SIZE * GROUP_SIZE
# independent Android backends (2 * 4 = 8 by default).
python ../data_preprocess/mw_onlinerl.py \
    --curriculum \
    --curriculum_mode interleave \
    --hard_task_num "$HARD_TASK_NUM" \
    --exclude_google \
    --batch_size "$TRAIN_BATCH_SIZE" \
    --total_epochs "$TOTAL_CURRICULUM_EPOCHS" \
    --local_dir "$DATA_ROOT" \
    --task_file "$TASK_FILE" \
    --data_source "$GEOMETRY3K_DIR"

HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.use_kl_in_reward=False \
    algorithm.filter_groups.enable=True \
    algorithm.filter_groups.max_num_gen_batches=5 \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/test.parquet" \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    data.val_batch_size="$VAL_BATCH_SIZE" \
    data.shuffle=False \
    data.max_prompt_length=12000 \
    data.max_response_length=384 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.image_key=images \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.lora_rank="$LORA_RANK" \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.target_modules='[q_proj,k_proj,v_proj,o_proj]' \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    env.env_name=MobileWorld \
    env.model_type=mai_ui \
    env.server_file="$SERVER_FILE" \
    env.seed=7 \
    env.history_length="$HISTORY_LENGTH" \
    env.max_steps="$MAX_STEPS" \
    env.rollout.n="$GROUP_SIZE" \
    env.resources_per_worker.num_cpus=0.10 \
    env.step_reward_judge=False \
    env.litegui.enable=True \
    env.litegui.outcome_weight=1.0 \
    env.litegui.state_change_weight=0.20 \
    env.litegui.efficiency_weight=0.20 \
    env.litegui.invalid_action_penalty=0.05 \
    env.litegui.repeat_action_penalty=0.03 \
    env.litegui.state_change_threshold=0.015 \
    env.litegui.max_steps="$MAX_STEPS" \
    env.litegui.difficulty_file="$DIFFICULTY_FILE" \
    env.litegui.stats_path="$STATS_PATH" \
    env.litegui.target_success=0.5 \
    env.litegui.min_difficulty_weight=0.75 \
    env.litegui.max_difficulty_weight=1.5 \
    trainer.critic_warmup=0 \
    trainer.default_local_dir="$CHECKPOINT_DIR" \
    trainer.logger="['console','swanlab']" \
    trainer.project_name=litegui_rl \
    trainer.experiment_name=litegui_maiui_2b_grpo \
    trainer.n_gpus_per_node="$N_GPUS" \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.val_before_train=False \
    "$@"
