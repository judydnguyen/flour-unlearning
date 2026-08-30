#!/bin/bash
#
# FLOUR on TOFU — Llama-3.2-1B-Instruct, forget01 / forget05 / forget10.
#
# FLOUR is a two-phase, encoder-steered, RMU-style unlearning method:
#   Phase 1 (encoder): learn a per-sample steering direction r that points the
#     forget activations toward a stable, on-manifold answer-ablated "anchor".
#       loss = anchor_weight * (1 - cos(r, anchor))
#            + simulated_forget_weight * MSE(h_forget, h_forget_ref + coeff * r)
#     The encoder is warmed up for ENCODER_WARMUP_STEPS so it starts Phase 1
#     already aligned with the anchor direction.
#   Phase 2 (model): freeze the encoder and steer the model's layer-${LAYER}
#     activations toward (h_forget_ref + coeff * r) while preserving retain
#     activations (EMBED_DIFF retain loss).
#
# Workflow: train with eval OFF (the per-epoch MIA suite is the bottleneck),
# then evaluate checkpoints post-hoc with scripts/eval_calibrated.sh, which
# early-stops at the privleak zero-crossing and promotes the best-HM checkpoint
# to <run>/best/.
#
# Per-split recipe used in the paper (layer 3, gamma 0.5, alpha 2.0):
#   forget01: steering_coeff=8    forget05: steering_coeff=6    forget10: steering_coeff=6
#
# Knobs (override via env):
#   GPUS SPLITS LAYER GAMMA ALPHA ENCODER_EPOCHS ANCHOR_WEIGHT
#   SIMULATED_FORGET_WEIGHT ENCODER_WARMUP_STEPS SAVE_STEPS TASK_SUFFIX
#   COEFF_F01 COEFF_F05 COEFF_F10   per-split steering coefficients
#   DRY_RUN=1   print the plan, run nothing
#   OVERWRITE=1 ignore an existing best/ and retrain
#
# Usage:
#   bash scripts/flour_tofu_1b.sh                            # all splits, GPU 0
#   SPLITS="forget01" GPUS=1 bash scripts/flour_tofu_1b.sh   # forget01 only

set -e

PYTHON="${PYTHON:-python}"
export MASTER_PORT=$($PYTHON -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

MODEL="Llama-3.2-1B-Instruct"
MODEL_PATH="open-unlearning/tofu_${MODEL}_full"
GPUS="${GPUS:-0}"
DRY_RUN="${DRY_RUN:-0}"
OVERWRITE="${OVERWRITE:-0}"
SPLITS="${SPLITS:-forget01 forget05 forget10}"

LAYER="${LAYER:-3}"
GAMMA="${GAMMA:-0.5}"
ALPHA="${ALPHA:-2.0}"
ENCODER_EPOCHS="${ENCODER_EPOCHS:-6}"
ANCHOR_WEIGHT="${ANCHOR_WEIGHT:-1.0}"
SIMULATED_FORGET_WEIGHT="${SIMULATED_FORGET_WEIGHT:-1.0}"
ENCODER_WARMUP_STEPS="${ENCODER_WARMUP_STEPS:-30}"
SAVE_STEPS="${SAVE_STEPS:-10}"
TASK_SUFFIX="${TASK_SUFFIX:-}"
STOP_MARGIN="${STOP_MARGIN:-15}"

# Per-split steering coefficients (paper settings).
COEFF_F01="${COEFF_F01:-8}"
COEFF_F05="${COEFF_F05:-6}"
COEFF_F10="${COEFF_F10:-6}"

echo "Master Port: $MASTER_PORT"

run_scenario() {
    local FORGET_SPLIT=$1
    local RETAIN_SPLIT=$2
    local HOLDOUT_SPLIT=$3
    local BATCH=$4
    local WARMUP=$5
    local COEFF=$6

    local TASK=tofu_${MODEL}_${FORGET_SPLIT}_FLOUR_L${LAYER}c${COEFF}g${GAMMA}${TASK_SUFFIX}

    echo "=========================================="
    echo "  ${FORGET_SPLIT} / ${RETAIN_SPLIT} / ${HOLDOUT_SPLIT}"
    echo "  task: ${TASK}"
    echo "  layer=${LAYER} coeff=${COEFF} gamma=${GAMMA} alpha=${ALPHA}"
    echo "  encoder_epochs=${ENCODER_EPOCHS} warmup_steps=${ENCODER_WARMUP_STEPS}"
    echo "  batch=${BATCH} forget_warmup=${WARMUP} save_steps=${SAVE_STEPS}"
    echo "=========================================="

    if [ "$OVERWRITE" = "1" ]; then
        echo "[overwrite] retraining from scratch"
    elif [ -f "saves/unlearn/${TASK}/best/evals/TOFU_EVAL.json" ]; then
        echo "[skip] already complete at saves/unlearn/${TASK}/best"
        return
    fi
    local SKIP_TRAIN=0
    if [ "$OVERWRITE" != "1" ] && [ -d "saves/unlearn/${TASK}/best" ]; then
        echo "[resume] best/ exists, eval missing — eval only"
        SKIP_TRAIN=1
    fi
    if [ "$DRY_RUN" = "1" ]; then
        [ "$SKIP_TRAIN" = "1" ] && echo "  (dry run — would eval only)" || echo "  (dry run — would train + eval)"
        return
    fi

    if [ "$SKIP_TRAIN" != "1" ]; then
        CUDA_VISIBLE_DEVICES=$GPUS $PYTHON src/train.py --config-name=unlearn.yaml \
            experiment=unlearn/tofu/default \
            trainer=FLOUR \
            task_name=${TASK} \
            model=${MODEL} \
            forget_split=${FORGET_SPLIT} \
            retain_split=${RETAIN_SPLIT} \
            model.model_args.pretrained_model_name_or_path=${MODEL_PATH} \
            model.tokenizer_args.pretrained_model_name_or_path=${MODEL_PATH} \
            retain_logs_path=saves/eval/tofu_${MODEL}_${RETAIN_SPLIT}/TOFU_EVAL.json \
            trainer.args.per_device_train_batch_size=${BATCH} \
            trainer.args.gradient_accumulation_steps=1 \
            trainer.args.num_train_epochs=12 \
            trainer.args.learning_rate=2e-5 \
            +trainer.args.max_grad_norm=0.5 \
            trainer.args.logging_steps=10 \
            trainer.args.eval_strategy=no \
            trainer.args.save_strategy=steps \
            +trainer.args.save_steps=${SAVE_STEPS} \
            +trainer.args.save_total_limit=100 \
            trainer.method_args.module_regex="model\.layers\.${LAYER}" \
            trainer.method_args.encoder_epochs=${ENCODER_EPOCHS} \
            trainer.method_args.steering_coeff=${COEFF} \
            trainer.method_args.latent_dim=256 \
            trainer.method_args.anchor_weight=${ANCHOR_WEIGHT} \
            trainer.method_args.simulated_forget_weight=${SIMULATED_FORGET_WEIGHT} \
            trainer.method_args.encoder_warmup_steps=${ENCODER_WARMUP_STEPS} \
            trainer.method_args.forget_warmup_steps=${WARMUP} \
            trainer.method_args.gamma=${GAMMA} \
            trainer.method_args.alpha=${ALPHA} \
            trainer.method_args.retain_loss_type=EMBED_DIFF
    fi

    echo "  [eval] post-hoc early-stop calibration sweep"
    GPU=$GPUS STOP_MARGIN=$STOP_MARGIN MODEL=$MODEL PYTHON=$PYTHON \
        bash scripts/eval_calibrated.sh \
        "saves/unlearn/${TASK}" "${FORGET_SPLIT}" "${RETAIN_SPLIT}" "${HOLDOUT_SPLIT}"

    echo "Done: saves/unlearn/${TASK}"
}

echo "================================================================"
echo "  FLOUR on TOFU (Llama-3.2-1B-Instruct)"
echo "  splits: ${SPLITS}   GPU: ${GPUS}   dry_run: ${DRY_RUN}"
echo "  layer=${LAYER}  gamma=${GAMMA}  coeff f01/05/10=${COEFF_F01}/${COEFF_F05}/${COEFF_F10}"
echo "================================================================"

for split in $SPLITS; do
    case "$split" in
        forget01) run_scenario forget01 retain99 holdout01  4 10 "$COEFF_F01" ;;
        forget05) run_scenario forget05 retain95 holdout05  8 30 "$COEFF_F05" ;;
        forget10) run_scenario forget10 retain90 holdout10 16 30 "$COEFF_F10" ;;
        *) echo "[warn] unknown split '$split'" ;;
    esac
done

echo "================================================================"
echo "  Done. Best checkpoint per run: saves/unlearn/<task>/best"
echo "  Metrics:                       saves/unlearn/<task>/best/evals/TOFU_SUMMARY.json"
echo "================================================================"
