#!/bin/bash
#
# FLOUR on MUSE — Llama-2-7b, Books (or News), multi-GPU (DDP).
#
# Uses the early-layer FLOUR variant (trainer=FLOUR_DDP_early: hook layer 7,
# edit layers 3-7 mlp.down_proj). Editing early layers disrupts the
# representations downstream MIA probes rely on, moving privleak toward the
# retrain reference where late-layer edits leave it stuck.
#
# The forget/retain data and the unlearning target are the public MUSE-bench
# artifacts (muse-bench/MUSE-${DATA_SPLIT} and muse-bench/MUSE-${DATA_SPLIT}_target),
# wired in configs/experiment/unlearn/muse/default.yaml — nothing local required.
#
# privleak needs the retain-reference eval logs (RETAIN_LOGS). Point it at the
# MUSE retain/retrain reference eval JSON (fetched by `python setup_data.py --eval`).
# If missing, training/eval still run but privleak is reported as unavailable.
#
# Knobs (override via env):
#   DATA_SPLIT (Books|News, default Books)  STEERING_COEFF  GAMMA  ALPHA
#   LR  EPOCHS  ENCODER_EPOCHS  TRAIN_GPUS  EVAL_GPU  RETAIN_LOGS  RUN_LABEL
#
# Usage:
#   bash scripts/flour_muse.sh
#   DATA_SPLIT=News TRAIN_GPUS=0,1 bash scripts/flour_muse.sh

set -e

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

# Optional: load HF_TOKEN for the gated Llama-2 weights (./.env or HF cache).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$SCRIPT_DIR/../.env" ] && { set -a; source "$SCRIPT_DIR/../.env"; set +a; }
if [ -z "${HF_TOKEN:-}" ]; then
    HF_TOKEN_FILE="${HF_HOME:-$HOME/.cache/huggingface}/token"
    [ -f "$HF_TOKEN_FILE" ] && export HF_TOKEN=$(cat "$HF_TOKEN_FILE")
fi

model=Llama-2-7b-hf
data_split="${DATA_SPLIT:-Books}"
trainer=FLOUR_DDP_early

per_device_train_batch_size="${BATCH:-2}"
gradient_accumulation_steps="${GRAD_ACCUM:-1}"
TRAIN_GPUS="${TRAIN_GPUS:-0,1}"
EVAL_GPU="${EVAL_GPU:-1}"
ACCEL_CFG=configs/accelerate/ddp_2gpu_config.yaml

STEERING_COEFF="${STEERING_COEFF:-30}"
GAMMA="${GAMMA:-1.0}"
ALPHA="${ALPHA:-1.0}"
LR="${LR:-1e-4}"
EPOCHS="${EPOCHS:-12}"
ENCODER_EPOCHS="${ENCODER_EPOCHS:-2}"
RUN_LABEL="${RUN_LABEL:-repro}"

TAG="L7_sc${STEERING_COEFF}_g${GAMMA}_a${ALPHA}_lr${LR}_ep${EPOCHS}"
task_name=muse_${model}_${data_split}_FLOUR_${TAG}_${RUN_LABEL}
save_dir=saves/unlearn/${task_name}

RETAIN_LOGS="${RETAIN_LOGS:-saves/eval/muse_${model}_${data_split}_retrain/MUSE_EVAL.json}"
[ -f "$RETAIN_LOGS" ] || echo "[warn] retain logs missing at $RETAIN_LOGS (privleak will be unavailable)"

echo "============================================="
echo "Run: $task_name"
echo "  hook layer 7, edit layers 3-7 (from trainer=FLOUR_DDP_early)"
echo "  steering_coeff=${STEERING_COEFF} gamma=${GAMMA} alpha=${ALPHA} lr=${LR} epochs=${EPOCHS} encoder_epochs=${ENCODER_EPOCHS}"
echo "============================================="

# ---- 1. unlearn ----
if [ -f "$save_dir/last/model.safetensors" ] || [ -f "$save_dir/last/model.safetensors.index.json" ]; then
    echo "[done] training already exists at $save_dir — delete to redo"
else
    CUDA_VISIBLE_DEVICES=$TRAIN_GPUS accelerate launch \
        --config_file $ACCEL_CFG \
        --main_process_port $MASTER_PORT \
        src/train.py --config-name=unlearn.yaml \
        experiment=unlearn/muse/default.yaml \
        model=${model} \
        data_split=${data_split} \
        trainer=${trainer} \
        task_name=${task_name} \
        retain_logs_path=${RETAIN_LOGS} \
        trainer.args.per_device_train_batch_size=${per_device_train_batch_size} \
        trainer.args.gradient_accumulation_steps=${gradient_accumulation_steps} \
        trainer.args.ddp_find_unused_parameters=true \
        trainer.args.gradient_checkpointing=true \
        trainer.args.save_strategy=epoch \
        +trainer.args.save_total_limit=10 \
        trainer.args.save_only_model=true \
        trainer.args.num_train_epochs=${EPOCHS} \
        trainer.args.learning_rate=${LR} \
        trainer.args.logging_steps=10 \
        trainer.args.eval_strategy=no \
        trainer.method_args.steering_coeff=${STEERING_COEFF} \
        trainer.method_args.gamma=${GAMMA} \
        trainer.method_args.alpha=${ALPHA} \
        trainer.method_args.encoder_epochs=${ENCODER_EPOCHS}
fi

# ---- 2. eval final checkpoint ----
if [ -f "$save_dir/evals/MUSE_SUMMARY.json" ]; then
    echo "[done] eval already exists at $save_dir/evals/"
else
    echo "--- Eval $task_name (last/) on GPU $EVAL_GPU ---"
    CUDA_VISIBLE_DEVICES=$EVAL_GPU python src/eval.py \
        experiment=eval/muse/default.yaml \
        data_split=${data_split} \
        task_name=${task_name} \
        model=${model} \
        model.model_args.pretrained_model_name_or_path=${save_dir}/last \
        paths.output_dir=${save_dir}/evals \
        retain_logs_path=${RETAIN_LOGS}
fi

echo "=========================================="
echo "Done: $save_dir/evals/MUSE_SUMMARY.json"
echo "To find the best epoch, eval every checkpoint:"
echo "  bash scripts/eval_muse_checkpoints.sh $save_dir $data_split $model $EVAL_GPU"
echo "=========================================="
