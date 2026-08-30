# Reproducing the FLOUR results

This document lists the exact recipes the launch scripts use and explains which
metrics to read. Every command assumes you have run `python setup_data.py --eval`
so the retain/retrain reference logs are present under `saves/eval/`.

## TOFU recipe

Common to all TOFU runs: hook **layer 3**, encoder `latent_dim=256`,
`encoder_epochs=6`, `encoder_warmup_steps=30`, `anchor_mode=blank`,
`simulated_forget_weight=1.0`, `retain_loss_type=EMBED_DIFF`, `max_grad_norm=0.5`,
`num_train_epochs=12`. Checkpoints are selected post-hoc by
`scripts/eval_calibrated.sh` (early stop at the privleak zero-crossing, pick the
peak joint-score checkpoint).

| Model | Split | steering_coeff | gamma | alpha | lr | batch |
|---|---|---|---|---|---|---|
| Llama-3.2-1B-Instruct | forget01 | 8  | 0.5 | 2.0 | 2e-5 | 4  |
| Llama-3.2-1B-Instruct | forget05 | 6  | 0.5 | 2.0 | 2e-5 | 8  |
| Llama-3.2-1B-Instruct | forget10 | 6  | 0.5 | 2.0 | 2e-5 | 16 |
| Llama-3.2-3B-Instruct | forget01 | 10 | 1.0 | 1.0 | 5e-5 | 2  |
| Llama-3.2-3B-Instruct | forget05 | 10 | 1.0 | 1.0 | 5e-5 | 4  |
| Llama-3.2-3B-Instruct | forget10 | 10 | 1.0 | 1.0 | 5e-5 | 8  |

These are the script defaults, so:

```bash
bash scripts/flour_tofu_1b.sh
bash scripts/flour_tofu_3b.sh
```

reproduce all six runs. Override any value via env, e.g.
`COEFF_F01=8 SPLITS=forget01 bash scripts/flour_tofu_1b.sh`.

### TOFU metrics to read

Final metrics are in `saves/unlearn/<task>/best/evals/TOFU_SUMMARY.json`. Read
forgetting, utility, and privacy **together** — FLOUR's claim is a better joint
trade-off, not a single scalar:

- **forgetting**: `forget_quality`, `forget_Q_A_Prob`, `forget_Q_A_ROUGE`,
  `forget_Truth_Ratio`
- **utility**: `model_utility`
- **privacy**: `privleak` (closer to 0 is better) and the MIA attacks
  (`mia_loss`, `mia_min_k`, `mia_min_k_plus_plus`, `mia_zlib`)

`forget_quality` and `privleak` are defined relative to the retain reference at
`retain_logs_path` — comparisons are only meaningful against the matching
reference (`retain99`/`retain95`/`retain90` for `forget01`/`05`/`10`). The
`best_step.json` file records the selected step and the per-component scores used
for checkpoint selection (utility, `1 - forget_Q_A_Prob`, `1 - |privleak|/100`).

## MUSE recipe

Llama-2-7b, early-layer FLOUR (`trainer=FLOUR_DDP_early`: hook layer 7, edit
layers 3–7 `mlp.down_proj`), 2-GPU DDP. Defaults: `steering_coeff=30`,
`gamma=1.0`, `alpha=1.0`, `lr=1e-4`, `num_train_epochs=12`, `encoder_epochs=2`.

```bash
bash scripts/flour_muse.sh                 # Books
DATA_SPLIT=News bash scripts/flour_muse.sh # News
bash scripts/eval_muse_checkpoints.sh saves/unlearn/<task> Books   # per-epoch eval
```

### MUSE metrics to read

Final metrics in `saves/unlearn/<task>/evals/MUSE_SUMMARY.json` (or the
per-checkpoint `evals_checkpoint-*/`). Read forgetting and retention together:

- **forgetting**: `forget_knowmem_ROUGE`, `forget_verbmem_ROUGE`
- **retention**: `retain_knowmem_ROUGE` (should stay near the target model)
- **privacy**: `privleak` (interpret only against the retain reference logs)

A large forget-ROUGE drop is not a success if `retain_knowmem_ROUGE` collapses.

## Notes

- The scripts skip any run whose `best/` (TOFU) or `last/` (MUSE) already exists;
  use `OVERWRITE=1` (TOFU) or delete the run directory to force a rerun.
- Absolute numbers depend on the target model, the reference logs, and hardware
  numerics. Always compare a method against the baselines run with the *same*
  reference logs in the same environment.
