# Environment activate
. /mnt/data/zouanni.zan/miniconda3/etc/profile.d/conda.sh;
conda activate appworld;

# Set Appworld environment service
cd /mnt/data/zouanni.zan/service_0815/EnvService
bash env_sandbox/appworld.sh &

# Set ExperienceMaker service
# conda activate em;
# cd /mnt/data/zouanni.zan/service_0815/ExperienceMaker
# experiencemaker \
#   http_service.host="127.0.0.1" \
#   http_service.port=8001 \
#   llm.default.model_name=qwen-max-2025-01-25 \
#   embedding_model.default.model_name=text-embedding-v4 \
#   vector_store.default.backend=local_file \
#   op.rerank_experience_op.params.enable_llm_rerank=false \
#   op.experience_validation_op.params.validation_threshold=0.3 \
#   op.experience_deduplication_op.params.similarity_threshold=0.3 \
#   http_service.limit_concurrency=256 \
#   thread_pool.max_workers=256 &
# sleep 30
# # Load vector store in ExperienceMaker from local file
# curl -X POST "http://localhost:8001/vector_store" \
#   -H "Content-Type: application/json" \
#   -d '{
#     "workspace_id": "default",
#     "action": "load",
#     "path": "./step_experiences/qwen2.5-14b_fixed"
#   }'
# sleep 10

# Set gpu parameters
num_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Number of GPUS: $num_gpus"
cd /mnt/data/zouanni.zan/codes/BeyondAgent;
conda activate verl;

set -x
export HYDRA_FULL_ERROR=1
export SWANLAB_API_KEY="xxx"
export DASHSCOPE_API_KEY="sk-xxx"
ulimit -n 65535

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/config"
env_url=http://localhost:8000
em_url=http://localhost:8001
current_time=$(date "+%Y%m%d_%H%M%S")
experiment_name="exp_${current_time}_debug_het_mergemain"


val_rollout_expmode="woexp"
train_rollout_expmode="woexp"
rollout_expratio=0.0
train_sample_expmode="keep"
train_sample_keepratio=1.0
clip_ratio_high=0.28
off_cliprange_high=0.4

python3 -m beyondagent.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='beyond_agent_dataflow' \
    env_service.env_url=$env_url \
    experience_maker.base_url=$em_url \
    experience_maker.enable_context_generator=False \
    experience_maker.enable_summarizer=False \
    experience_maker.workspace_id="default" \
    experience_maker.updated_freq=0 \
    actor_rollout_ref.rollout.val_kwargs.n=8 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.9 \
    actor_rollout_ref.rollout.enable_request_id=False \
    actor_rollout_ref.rollout.sparse=True \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.off_cliprange_high=${off_cliprange_high} \
    hybrid_experience_training.val_rollout_expmode=${val_rollout_expmode} \
    hybrid_experience_training.train_rollout_expmode=${train_rollout_expmode} \
    hybrid_experience_training.train_sample_expmode=${train_sample_expmode} \
    hybrid_experience_training.rollout_expratio=${rollout_expratio} \
    hybrid_experience_training.train_sample_keepratio=${train_sample_keepratio} \
    actor_rollout_ref.rollout.sparse=True \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=32 \
    data.max_prompt_length=4096 \
    data.max_response_length=20480 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.rollout.prompt_length=20480 \
    actor_rollout_ref.rollout.response_length=4096 \
    actor_rollout_ref.rollout.max_model_len=20480 \
    actor_rollout_ref.rollout.temperature=0.9 \
    actor_rollout_ref.model.path=/mnt/data/zouanni.zan/models/Qwen2.5-14B-Instruct \
    actor_rollout_ref.rollout.use_qwen3=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.n_gpus_per_node=$num_gpus \
    trainer.critic_warmup=0 \
    trainer.logger=['console','swanlab'] \
    trainer.project_name='ba_debug_anni_25-14b' \
    trainer.experiment_name=$experiment_name \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=10 \
    trainer.total_epochs=40 \
    trainer.val_before_train=False \
    trainer.validation_data_dir="experiments/${experiment_name}/validation_log" \
    trainer.rollout_data_dir="experiments/${experiment_name}/rollout_log" \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=20480 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=20480 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=20480 \
    critic.ppo_max_token_len_per_gpu=20480 \
    critic.forward_max_token_len_per_gpu=20480 \
    data.train_files=null \
    data.val_files=null \
    $@

sleep 1h