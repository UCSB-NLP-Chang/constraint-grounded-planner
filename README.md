# Collision-and Reachability-Aware Multi-Robot Control with Grounded LLM Planners

This is the implementation for the paper [Collision-and Reachability-Aware Multi-Robot Control with Grounded LLM Planners](https://arxiv.org/abs/2505.20573).

## Requirements
```bash
conda create -n rob python=3.10 -y
pip install -e .
```
We also include the reference conda environments we used in `conda_env.txt`.

## Usage
We include three core implementations in this repository, including data generation, RL training and evaluation. For SFT training, we refer readers to [LlamaFactory](https://github.com/hiyouga/LLaMA-Factory).

### Data Generation
```bash
# Box2D environment
python scripts/search_ray.py 
# Box3D environment
python scripts/search_3d_ray.py
```

The data we involved in the experiments for the paper is included in this [Huggingface](https://huggingface.co/collections/Shiyu-Lab/constraint-grounded-multi-robot-llm-planner).

### RL Training
```bash
N_ROLLOUT=8
N_GPUS=8
TRAINER_MODE="box" # "box" or "box3d"
ADV="grpo"
MODEL_NAME="" # path to the SFT finetuned model
TRAIN_DATA_DIR="" # path to the training data
TEST_DATA_DIR="" # path to the test data

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
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.use_custom=True \
    trainer.val_before_train=False \
    trainer.trainer_mode=$TRAINER_MODE \
    $@
```

### Evaluation
```bash
TEST_DATA="" # path to the evaluation data
ENV="" # box or box3d
PROMPT="" # path to the prompt file
llm="" # path to the LLM

python scripts/run_box.py \
    data=$TEST_DATA \
    env_name=$ENV \
    prompt_file=$PROMPT \
    llm_name=$llm \
    runner=$RUNNER \
    $@
```

We also provide example bash scripts in the `bash/` folder for further reference.

### Citation
If you find this code useful for your research, please consider citing our paper:
```bibtex
@article{ji2025collision,
  title={Collision-and Reachability-Aware Multi-Robot Control with Grounded LLM Planners},
  author={Ji, Jiabao and Chen, Yongchao and Zhang, Yang and Kompella, Ramana Rao and Fan, Chuchu and Liu, Gaowen and Chang, Shiyu},
  journal={arXiv preprint arXiv:2505.20573},
  year={2025}
}
```