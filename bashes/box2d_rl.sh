# Mark the start of the job
startTime=$(date +%s)
set -x

export VLLM_ATTENTION_BACKEND=XFORMERS

N_GPUS=8
TP=1
MODEL_NAME="Qwen/Qwen2.5-3B-Instruct" # Please set to your finetuned model path
ADV="grpo"
RUN_NAME="Qwen2.5-3Bfintune_adv@${ADV}-formatr-fix-realfix"
SAVE_MODEL_DIR="model_cache/$RUN_NAME"
# RUN_NAME="Qwen2.5-3B_countdown-adv@${ADV}_prompt@tinyzero"

box_train=data_cache/box/box1-small-train
box_test=data_cache/box/box1-small-test


TRAIN_DATA_DIR=$box_train
TEST_DATA_DIR="['$box_test']"
STEP_BATCH_SIZE=128
# STEP_BATCH_SIZE=16
BATCH_SIZE=2
FW_BS=$((BATCH_SIZE * 2))
N_ROLLOUT=1
DATA_CLASS="box"
# PROMPT_FILE="prompts/box-prompt-v1/dmas-think-qwen-stepplan.py"
PROMPT_FILE="prompts/box-prompt-v2/dmas-think-qwen-stepplan.py"
TRAINER_MODE="box"
# 
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=$ADV \
    data.train_files="$TRAIN_DATA_DIR" \
    data.val_files="$TEST_DATA_DIR" \
    data.train_batch_size=$STEP_BATCH_SIZE \
    data.max_prompt_length=4096 \
    data.max_response_length=5120 \
    data.data_class=$DATA_CLASS \
    data.prompt_file=$PROMPT_FILE \
    actor_rollout_ref.model.path=$MODEL_NAME \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$BATCH_SIZE \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.000 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$FW_BS\
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TP \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
    actor_rollout_ref.rollout.n=$N_ROLLOUT \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$FW_BS \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=False \
    critic.model.path=$MODEL_NAME \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=$BATCH_SIZE \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    algorithm.kl_ctrl.kl_coef=0.000 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='robot-rl' \
    trainer.experiment_name=${RUN_NAME} \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 \
    trainer.remove_previous_ckpt_in_save=True \
    trainer.use_custom=True \
    trainer.val_before_train=False \
    trainer.trainer_mode=$TRAINER_MODE \
    $@
    # trainer.max_steps=40 \

