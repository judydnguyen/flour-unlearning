#!/bin/bash
#
# Evaluate every checkpoint-* subdir of a MUSE unlearn run, so you can pick the
# best epoch. Per-checkpoint evals land in ${save_dir}/evals_checkpoint-{step}/
# so the final-model eval at ${save_dir}/evals/ is preserved.
#
# Usage:
#   bash scripts/eval_muse_checkpoints.sh <save_dir> <data_split> [model] [eval_gpu]
# Example:
#   bash scripts/eval_muse_checkpoints.sh \
#       saves/unlearn/muse_Llama-2-7b-hf_Books_FLOUR_L7_sc30_g1.0_a1.0_lr1e-4_ep12_repro Books

set -e

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SAVE_DIR=${1:?save_dir required}
DATA_SPLIT=${2:?data_split required (News|Books)}
MODEL=${3:-Llama-2-7b-hf}
EVAL_GPU=${4:-0}

RETAIN_LOGS="${RETAIN_LOGS:-saves/eval/muse_${MODEL}_${DATA_SPLIT}_retrain/MUSE_EVAL.json}"
[ -f "$RETAIN_LOGS" ] || echo "[warn] retain logs missing at $RETAIN_LOGS (privleak will be unavailable)"

# Some network filesystems SIGBUS when mmap-ing safetensors under memory
# pressure. Stage each checkpoint to local /tmp before loading; clean up on exit.
STAGE_ROOT="${STAGE_ROOT:-/tmp/eval_stage_$$}"
mkdir -p "$STAGE_ROOT"
trap 'rm -rf "$STAGE_ROOT"' EXIT

shopt -s nullglob
ckpts=("$SAVE_DIR"/checkpoint-*)
if [ ${#ckpts[@]} -eq 0 ]; then
    echo "[err] no checkpoint-* dirs in $SAVE_DIR"
    exit 1
fi

for ckpt in "${ckpts[@]}"; do
    step=$(basename "$ckpt" | sed 's/checkpoint-//')
    out_dir="$SAVE_DIR/evals_checkpoint-${step}"

    if [ -f "$out_dir/MUSE_SUMMARY.json" ]; then
        echo "[done] checkpoint-${step}: $out_dir/MUSE_SUMMARY.json"
        continue
    fi
    if [ ! -f "$ckpt/model.safetensors" ] && [ ! -f "$ckpt/model.safetensors.index.json" ]; then
        echo "[skip] checkpoint-${step}: no model weights"
        continue
    fi

    staged="$STAGE_ROOT/checkpoint-${step}"
    rm -rf "$staged"; mkdir -p "$staged"
    echo "[stage] copying $(du -sh "$ckpt" | cut -f1) -> $staged"
    cp -r "$ckpt"/* "$staged"/

    echo "=== Eval checkpoint-${step} ==="
    CUDA_VISIBLE_DEVICES=$EVAL_GPU python src/eval.py \
        experiment=eval/muse/default.yaml \
        data_split=${DATA_SPLIT} \
        task_name=$(basename "$SAVE_DIR")_ckpt${step} \
        model=${MODEL} \
        model.model_args.pretrained_model_name_or_path=${staged} \
        paths.output_dir=${out_dir} \
        retain_logs_path=${RETAIN_LOGS}

    rm -rf "$staged"
done

echo "=== All checkpoint evals done. Compare: ==="
echo "  for f in $SAVE_DIR/evals_checkpoint-*/MUSE_SUMMARY.json; do echo \$f; cat \$f; echo; done"
