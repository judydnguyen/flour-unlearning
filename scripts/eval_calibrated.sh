#!/bin/bash
#
# Post-hoc TOFU eval of FLOUR checkpoints with EARLY STOP at the privleak
# zero-crossing. During Phase 2, privleak rises roughly monotonically
# (very negative -> positive), so the calibrated checkpoint (privleak ~ 0, peak
# joint HM score) is the first one near zero. We evaluate checkpoints in
# increasing step order and stop once privleak exceeds +STOP_MARGIN, skipping the
# over-trained tail. The best-HM checkpoint is then copied to <run_dir>/best/.
#
# The joint score is the harmonic mean of three components in [0,1]:
#   utility (model_utility), forgetting (1 - forget_Q_A_Prob),
#   privacy (1 - |privleak|/100).
#
# Usage:
#   bash scripts/eval_calibrated.sh <run_dir> <forget_split> <retain_split> <holdout_split>
#   GPU=1 STOP_MARGIN=15 bash scripts/eval_calibrated.sh saves/unlearn/<task> forget01 retain99 holdout01
#
# Env:
#   MODEL         default Llama-3.2-1B-Instruct
#   GPU           default 0
#   STOP_MARGIN   stop once privleak > this (default 15)
#   STEP_STRIDE   evaluate every Nth checkpoint (default 1 = all)
#   PYTHON        python interpreter (default: python)

set -e

PYTHON="${PYTHON:-python}"
RUN_DIR="$1"
FORGET_SPLIT="${2:-forget01}"
RETAIN_SPLIT="${3:-retain99}"
HOLDOUT_SPLIT="${4:-holdout01}"
MODEL="${MODEL:-Llama-3.2-1B-Instruct}"
GPU="${GPU:-0}"
STOP_MARGIN="${STOP_MARGIN:-15}"
STEP_STRIDE="${STEP_STRIDE:-1}"

if [ -z "$RUN_DIR" ] || [ ! -d "$RUN_DIR" ]; then
    echo "usage: bash scripts/eval_calibrated.sh <run_dir> [forget_split] [retain_split] [holdout_split]"
    exit 1
fi

# Collect checkpoint steps that have model weights, sorted ascending.
STEPS=$(for d in "$RUN_DIR"/checkpoint-*; do
    [ -f "$d/model.safetensors" ] || continue
    echo "$d" | grep -oE 'checkpoint-[0-9]+' | grep -oE '[0-9]+'
done | sort -n | awk -v s="$STEP_STRIDE" 'NR % s == 1 || s == 1')

if [ -z "$STEPS" ]; then
    echo "[error] no checkpoints with model.safetensors under $RUN_DIR"
    exit 1
fi

echo "================================================================"
echo "  Calibrated eval (early stop at privleak > +${STOP_MARGIN})"
echo "  run: $(basename $RUN_DIR)"
echo "  splits: ${FORGET_SPLIT} / ${RETAIN_SPLIT} / ${HOLDOUT_SPLIT}"
echo "  steps to consider: $(echo $STEPS | tr '\n' ' ')"
echo "================================================================"

BEST_STEP=""
BEST_HM="-1"
for STEP in $STEPS; do
    CKPT="$RUN_DIR/checkpoint-$STEP"
    SUM="$CKPT/evals/TOFU_SUMMARY.json"

    # Reuse an existing full eval if present (must have mia_loss + privleak).
    if [ -f "$SUM" ] && $PYTHON -c "import json,sys; d=json.load(open('$SUM')); sys.exit(0 if 'mia_loss' in d and 'privleak' in d else 1)" 2>/dev/null; then
        echo "[reuse] ckpt-$STEP already evaluated"
    else
        echo "[eval]  ckpt-$STEP"
        CUDA_VISIBLE_DEVICES=$GPU $PYTHON src/eval.py --config-name=eval.yaml \
            experiment=eval/tofu/default \
            eval=tofu_full_privacy \
            eval.tofu.overwrite=true \
            forget_split=${FORGET_SPLIT} \
            holdout_split=${HOLDOUT_SPLIT} \
            model=${MODEL} \
            task_name=$(basename ${RUN_DIR})_ckpt${STEP}_eval \
            model.model_args.pretrained_model_name_or_path=$CKPT \
            model.tokenizer_args.pretrained_model_name_or_path=$CKPT \
            paths.output_dir=$CKPT/evals \
            retain_logs_path=saves/eval/tofu_${MODEL}_${RETAIN_SPLIT}/TOFU_EVAL.json \
            > /dev/null 2>&1
    fi

    # Read privleak + joint HM, decide whether to stop.
    read PRIV HM < <($PYTHON -c "
import json
d = json.load(open('$SUM'))
util = d.get('model_utility'); fp = d.get('forget_Q_A_Prob'); pl = d.get('privleak')
if util is None or fp is None or pl is None:
    print('nan nan')
else:
    hm = 3/(1/max(util,1e-9) + 1/max(1-fp,1e-9) + 1/max(1-abs(pl)/100,1e-9))
    print(f'{pl:.4f} {hm:.4f}')
")
    echo "        privleak=${PRIV}  HM=${HM}"

    if $PYTHON -c "import sys; sys.exit(0 if float('$HM') > float('$BEST_HM') else 1)" 2>/dev/null; then
        BEST_HM="$HM"
        BEST_STEP="$STEP"
    fi

    if $PYTHON -c "import sys; sys.exit(0 if float('$PRIV') > float('$STOP_MARGIN') else 1)" 2>/dev/null; then
        echo "        privleak past +${STOP_MARGIN} — stopping early (over-trained beyond here)"
        break
    fi
done

echo "================================================================"
echo "  Best: checkpoint-${BEST_STEP}  HM=${BEST_HM}"
echo "================================================================"

# Promote the best checkpoint to best/.
if [ -n "$BEST_STEP" ]; then
    BEST_SRC="$RUN_DIR/checkpoint-$BEST_STEP"
    BEST_DST="$RUN_DIR/best"
    rm -rf "$BEST_DST"
    cp -r "$BEST_SRC" "$BEST_DST"
    $PYTHON -c "
import json
d = json.load(open('$BEST_SRC/evals/TOFU_SUMMARY.json'))
util = d['model_utility']; fp = d['forget_Q_A_Prob']; pl = d['privleak']
uc = max(0.0, min(1.0, util)); fc = max(0.0, min(1.0, 1-fp)); pc = max(0.0, 1-min(abs(pl),100)/100)
hm = 3/(1/max(uc,1e-10) + 1/max(fc,1e-10) + 1/max(pc,1e-10))
json.dump({'step': $BEST_STEP, 'model_utility': util, 'forget_Q_A_Prob': fp, 'privleak': pl,
           'util_component': uc, 'forget_component': fc, 'priv_component': pc, 'joint_score': hm},
          open('$BEST_DST/best_step.json','w'), indent=2)
print('wrote', '$BEST_DST/best_step.json')
"
    echo "Promoted checkpoint-${BEST_STEP} -> ${BEST_DST}"
fi
