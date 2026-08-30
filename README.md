# FLOUR: Mixing the Right Ingredients for LLM Unlearning in Latent Space

> **Accepted to Findings of EMNLP 2026.**
> This repository accompanies our paper. It is built on top of the
> open-source [OpenUnlearning](https://github.com/locuslab/open-unlearning)
> framework (MIT licensed); we add our method, **FLOUR**, and the configs and
> scripts needed to reproduce our main results.

---

## Overview

**FLOUR** is a representation-level unlearning method. Instead of pushing forget
examples toward a fixed random vector (as in RMU) or maximizing their loss (as in
gradient-ascent methods), FLOUR learns a *per-sample steering direction* with a
small encoder and steers forget activations toward a **stable, on-manifold
answer-ablated anchor**. This avoids the representation collapse that degrades
model utility and produces incoherent generations, while still removing the
target knowledge.

It is a two-phase method:

- **Phase 1 (encoder).** Freeze the model and train a small residual-MLP encoder
  to produce a steering direction `r` for each forget example:

  ```
  loss = anchor_weight * (1 - cos(r, anchor))
       + simulated_forget_weight * MSE(h_forget, h_forget_ref + coeff * r)
  ```

  The anchor is an on-manifold, answer-ablated activation (see `anchor_mode`).
  The encoder is warmed up for `encoder_warmup_steps` so Phase 1 starts already
  aligned with the anchor direction.

- **Phase 2 (model).** Freeze the encoder and update the model so that the
  layer-`L` activations of forget examples move to `h_forget_ref + coeff * r`,
  while retain activations are preserved (`EMBED_DIFF` retain loss). The
  forget/retain balance is controlled by `gamma`/`alpha`.

The implementation is in [`src/trainer/unlearn/flour.py`](src/trainer/unlearn/flour.py)
(single-GPU) and [`src/trainer/unlearn/flour_parallel.py`](src/trainer/unlearn/flour_parallel.py)
(DDP / multi-GPU). Configs live in
[`configs/trainer/FLOUR.yaml`](configs/trainer/FLOUR.yaml),
[`FLOUR_DDP.yaml`](configs/trainer/FLOUR_DDP.yaml), and
[`FLOUR_DDP_early.yaml`](configs/trainer/FLOUR_DDP_early.yaml).

We evaluate on **TOFU** and **MUSE** using the standard OpenUnlearning metrics
(forget quality, model utility, privleak / MIA, ROUGE, etc.).

---

## Installation

```bash
conda create -n flour python=3.11
conda activate flour
pip install .                      # or: pip install .[lm-eval] for WMDP/lm-eval
pip install --no-build-isolation flash-attn==2.6.3
```

Key pinned dependencies (see [`requirements.txt`](requirements.txt)): `torch==2.4.1`,
`transformers==4.45.1`, `accelerate==0.34.2`, `deepspeed==0.15.4`,
`datasets==3.0.1`, `hydra-core==1.3`.

## Data and reference-log setup

Evaluation metrics such as TOFU `forget_quality` and MUSE/TOFU `privleak` are
computed *relative to a retain/retrain reference model*. Download the reference
eval logs (and supporting data) before evaluating:

```bash
python setup_data.py --eval     # populates saves/eval/ with reference logs
# python setup_data.py --help   # other datasets (e.g. WMDP)
```

Target models are pulled from the Hugging Face Hub on first use:
TOFU targets from `open-unlearning/tofu_<model>_full`, MUSE targets/data from
`muse-bench/MUSE-<split>`. The gated Llama-2 weights used for MUSE require a
Hugging Face token (`export HF_TOKEN=...`).

---

## Reproducing the main results

All scripts are env-configurable and skip runs that are already complete. They
write to `saves/unlearn/<task_name>/`.

### TOFU (Llama-3.2-1B and 3B), forget01 / forget05 / forget10

```bash
bash scripts/flour_tofu_1b.sh        # Llama-3.2-1B-Instruct, all three splits
bash scripts/flour_tofu_3b.sh        # Llama-3.2-3B-Instruct, all three splits

# single split / specific GPU:
SPLITS="forget01" GPUS=0 bash scripts/flour_tofu_1b.sh
```

These train with evaluation off (the per-epoch MIA suite is the bottleneck),
then run [`scripts/eval_calibrated.sh`](scripts/eval_calibrated.sh), which sweeps
checkpoints, early-stops at the privleak zero-crossing, and promotes the
best-joint-score checkpoint to `saves/unlearn/<task>/best/`. Final metrics:

```
saves/unlearn/<task>/best/evals/TOFU_SUMMARY.json
saves/unlearn/<task>/best/best_step.json     # selected step + component scores
```

### MUSE (Llama-2-7b), Books / News

```bash
bash scripts/flour_muse.sh                       # Books (default), 2-GPU DDP
DATA_SPLIT=News bash scripts/flour_muse.sh

# evaluate every epoch checkpoint to pick the best:
bash scripts/eval_muse_checkpoints.sh saves/unlearn/<task> Books
```

See [`docs/repro.md`](docs/repro.md) for the per-split hyperparameters and the
metrics to read.

---

## Baselines

The OpenUnlearning baselines used for comparison are included and run through the
same entrypoints — just swap the `trainer=` override:

```bash
python src/train.py --config-name=unlearn.yaml experiment=unlearn/tofu/default \
  forget_split=forget10 retain_split=retain90 trainer=NPO \
  model=Llama-3.2-1B-Instruct \
  model.model_args.pretrained_model_name_or_path=open-unlearning/tofu_Llama-3.2-1B-Instruct_full \
  task_name=tofu_1B_forget10_NPO
```

Available baseline trainers: `GradAscent`, `GradDiff`, `NPO`, `SimNPO`, `DPO`,
`RMU`, `UNDIAL`, `CEU`, `SatImp`, `WGA`, `PDU`
(see [`configs/trainer/`](configs/trainer/)).

---

## FLOUR hyperparameters

Set on the command line as `trainer.method_args.<name>=<value>` or in
[`configs/trainer/FLOUR.yaml`](configs/trainer/FLOUR.yaml).

| Parameter | Meaning | Default |
|---|---|---|
| `module_regex` | residual-stream layer hooked for steering | `model.layers.7` |
| `trainable_params_regex` | params updated in Phase 2 (`.*` = full model) | `.*` |
| `latent_dim`, `encoder_layers` | steering-encoder size | `256`, `2` |
| `encoder_epochs` | Phase-1 (encoder-only) epochs | `2` |
| `encoder_lr` | encoder learning rate | `1e-3` |
| `encoder_warmup_steps` | pre-align `r` with the anchor before Phase 1 | `30` |
| `anchor_weight` | weight on the anchor alignment term | `1.0` |
| `anchor_mode` | how the anchor is built: `blank` / `refusal` / `corrupt` / `delta` | `blank` |
| `ablation_token` | token that replaces the answer span in `blank`/`delta` | `unk` |
| `simulated_forget_weight` | weight on the Phase-1 simulated-forget MSE | `1.0` |
| `steering_coeff` | Phase-2 activation-steering scale | `20` |
| `forget_warmup_steps` | ramp the forget loss in over the first N steps | `0` |
| `gamma`, `alpha` | forget / retain loss weights | `1.0`, `1.0` |
| `retain_loss_type` | retain loss (`EMBED_DIFF` = activation-level) | `EMBED_DIFF` |

---

## Tests

A unit test for the FLOUR retain-loss / encoder path (no GPU required):

```bash
CUDA_VISIBLE_DEVICES= pytest tests/test_flour_retain.py
```

---

## Repository layout

```
src/train.py                         finetuning / unlearning entrypoint
src/eval.py                          benchmark evaluation entrypoint
src/trainer/unlearn/flour.py         FLOUR trainer (single-GPU)
src/trainer/unlearn/flour_parallel.py FLOUR trainer (DDP / multi-GPU)
configs/trainer/FLOUR*.yaml          FLOUR method configs
configs/experiment/{unlearn,eval}/   TOFU / MUSE / WMDP experiment presets
configs/eval/                        benchmarks + metric definitions
scripts/flour_tofu_1b.sh, flour_tofu_3b.sh, flour_muse.sh
scripts/eval_calibrated.sh, eval_muse_checkpoints.sh
docs/repro.md                        reproduction details + metrics to read
```

---

## Acknowledgements and citations

This code builds on the **OpenUnlearning** framework and the **TOFU** and
**MUSE** benchmarks. Please cite them:

```bibtex
@article{openunlearning2025,
  title={{OpenUnlearning}: Accelerating {LLM} Unlearning via Unified Benchmarking of Methods and Metrics},
  author={Dorna, Vineeth and Mekala, Anmol and Zhao, Wenlong and McCallum, Andrew and Lipton, Zachary C and Kolter, J Zico and Maini, Pratyush},
  journal={arXiv preprint arXiv:2506.12618},
  year={2025},
  url={https://arxiv.org/abs/2506.12618}
}
@inproceedings{maini2024tofu,
  title={{TOFU}: A Task of Fictitious Unlearning for {LLMs}},
  author={Maini, Pratyush and Feng, Zhili and Schwarzschild, Avi and Lipton, Zachary Chase and Kolter, J Zico},
  booktitle={First Conference on Language Modeling},
  year={2024}
}
@article{shi2024muse,
  title={{MUSE}: Machine Unlearning Six-Way Evaluation for Language Models},
  author={Shi, Weijia and Lee, Jaechan and Huang, Yangsibo and Malladi, Sadhika and Zhao, Jieyu and Holtzman, Ari and Liu, Daogao and Zettlemoyer, Luke and Smith, Noah A. and Zhang, Chiyuan},
  journal={arXiv preprint arXiv:2407.06460},
  year={2024}
}
```

FLOUR is accepted to **Findings of EMNLP 2026**; a BibTeX entry will be added
once the camera-ready version is available.

## License

Released under the MIT License (see [`LICENSE`](LICENSE)), inherited from the
OpenUnlearning framework this work builds on.
