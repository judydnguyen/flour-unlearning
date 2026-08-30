# Modified from https://github.com/huggingface/transformers/blob/v4.45.1/src/transformers/trainer.py

from typing import Dict, List, Optional, Union

import glob
import json
import os
import re
import shutil
import logging
from transformers import Trainer
from torch.utils.data import Dataset
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from typing import Any

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MPL = True
except ImportError:
    _MPL = False

logger = logging.getLogger(__name__)

_PLOT_METRICS = [
    ("model_utility",      "Model Utility",     "tab:blue"),
    ("forget_truth_ratio", "Forget Truth Ratio", "tab:orange"),
    ("forget_Q_A_Prob",    "Forget Q/A Prob",    "tab:red"),
    # ("forget_quality",     "Forget Quality",     "tab:green"),
    ("extraction_strength", "Extraction Strength", "tab:purple"),
    ("privleak",          "Privacy Leakage",    "tab:brown"),
]


def _refresh_metrics_plot(run_dir: str) -> None:
    """Scan all checkpoint-*/evals/TOFU_SUMMARY.json and save metrics_plot.png.

    Metrics are normalized to 0-100 relative to checkpoint-0 (pre-unlearning baseline).
    """
    if not _MPL:
        return
    paths = glob.glob(os.path.join(run_dir, "checkpoint-*", "evals", "TOFU_SUMMARY.json"))
    if not paths:
        return

    records = []
    for p in paths:
        m = re.search(r"checkpoint-(\d+)", p)
        if not m:
            continue
        with open(p) as f:
            data = json.load(f)
        records.append((int(m.group(1)), data))
    records.sort(key=lambda x: x[0])

    # Use checkpoint-0 as the baseline for normalization (pre-unlearning values = 100)
    baseline = records[0][1] if records[0][0] == 0 else {}

    steps = [r[0] for r in records]
    fig, ax = plt.subplots(figsize=(9, 4))
    for key, label, color in _PLOT_METRICS:
        vals = [r[1].get(key) for r in records]
        if not any(v is not None for v in vals):
            continue
        base = baseline.get(key)
        if base and base != 0:
            vals = [v / base * 100 if v is not None else None for v in vals]
        ax.plot(steps, vals, label=label, color=color, marker="o", markersize=3)

    ax.set_xlabel("Step")
    ax.set_ylabel("Score (% of pre-unlearning baseline)")
    ax.axhline(100, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_title("TOFU Metrics Over Training (normalized to baseline)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "metrics_plot_by_step.png"), dpi=120)
    plt.close(fig)


class FinetuneTrainer(Trainer):
    def __init__(self, evaluators=None, template_args=None, save_checkpoint_at_eval=False, *args, **kwargs):
        self.evaluators = evaluators
        self.template_args = template_args
        self.save_checkpoint_at_eval = save_checkpoint_at_eval
        self._best_forget_quality = -float("inf")
        self._custom_eval_skipped_warned = False
        # transformers renamed `tokenizer` → `processing_class` in 4.46. Older
        # 4.4x releases (e.g. 4.45.1, which sometimes ships in this env) reject
        # `processing_class`; newer ones reject `tokenizer`. Inspect the actual
        # HF Trainer signature once and forward under whichever name it accepts.
        import inspect
        _hf_trainer_params = inspect.signature(super().__init__).parameters
        if "processing_class" in _hf_trainer_params:
            if "tokenizer" in kwargs and "processing_class" not in kwargs:
                kwargs["processing_class"] = kwargs.pop("tokenizer")
        else:
            # Older transformers — keep `tokenizer`, drop `processing_class` if set.
            if "processing_class" in kwargs and "tokenizer" not in kwargs:
                kwargs["tokenizer"] = kwargs.pop("processing_class")
            kwargs.pop("processing_class", None)
        # transformers 5.x rejects eval_strategy != no when eval_dataset is
        # None. This repo uses custom evaluators (see self.evaluators) that
        # do not require an HF eval_dataset. Flip the strategy to "no" for
        # the duration of HF validation, then restore it so our overridden
        # evaluate() still fires at the configured intervals.
        _trainer_args = kwargs.get("args")
        _saved_strategy = None
        if (
            evaluators is not None
            and kwargs.get("eval_dataset") is None
            and _trainer_args is not None
        ):
            _saved_strategy = getattr(_trainer_args, "eval_strategy", None)
            if _saved_strategy not in (None, "no"):
                _trainer_args.eval_strategy = "no"
        super().__init__(*args, **kwargs)
        if _saved_strategy is not None and _saved_strategy not in (None, "no"):
            self.args.eval_strategy = _saved_strategy

    def __getattr__(self, name):
        # Many places in this repo (evaluators, etc.) read
        # `self.tokenizer`. transformers 5.x removed that attribute and
        # only exposes `self.processing_class`. Fall back transparently so
        # we don't have to rewrite every callsite.
        if name == "tokenizer":
            try:
                return object.__getattribute__(self, "processing_class")
            except AttributeError:
                pass
        raise AttributeError(name)

    def evaluate(
        self,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        trial: Dict[str, Any] = None,
    ) -> Dict[str, float]:
        # Run a custom evaluator and save results
        if self.evaluators:
            if self.accelerator.num_processes > 1:
                if not self._custom_eval_skipped_warned:
                    logger.warning(
                        "Custom evaluator requires a single accelerator process; "
                        "skipping in-training eval (num_processes=%d). "
                        "Run src/eval.py after training for full metrics.",
                        self.accelerator.num_processes,
                    )
                    self._custom_eval_skipped_warned = True
                return {}
            if self.accelerator.is_local_main_process:
                run_dir = self._get_output_dir(trial=trial)
                checkpoint_folder = (
                    f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
                )
                output_dir = os.path.join(run_dir, checkpoint_folder, "evals")
                os.makedirs(output_dir, exist_ok=True)
                eval_metrics = {}
                for _, evaluator in self.evaluators.items():
                    eval_args = {
                        "output_dir": output_dir,
                        "template_args": self.template_args,
                        "model": self.model,
                        "tokenizer": self.tokenizer,
                    }
                    eval_metrics.update(evaluator.evaluate(**eval_args))
                self.log(eval_metrics)
                _refresh_metrics_plot(run_dir)
                if self.save_checkpoint_at_eval:
                    ckpt_model_dir = os.path.join(run_dir, checkpoint_folder)
                    self.save_model(ckpt_model_dir)
                    logger.info(
                        "Saved model checkpoint at step %d → %s",
                        self.state.global_step, ckpt_model_dir,
                    )
                mu = eval_metrics.get("model_utility")
                fq = eval_metrics.get("forget_Q_A_Prob")
                pl = eval_metrics.get("privleak")
                if mu is not None and fq is not None and pl is not None:
                    util_c = max(0.0, min(1.0, mu))
                    forget_c = max(0.0, min(1.0, 1.0 - fq))
                    priv_c = max(0.0, 1.0 - min(abs(pl), 100.0) / 100.0)
                    parts = [max(util_c, 1e-10), max(forget_c, 1e-10), max(priv_c, 1e-10)]
                    joint_score = 3.0 / sum(1.0 / p for p in parts)
                else:
                    joint_score = None
                if joint_score is not None and joint_score > self._best_forget_quality:
                    self._best_forget_quality = joint_score
                    best_dir = os.path.join(run_dir, "best")
                    # Remove stale evals before saving so they can't outlive the model
                    stale_evals = os.path.join(best_dir, "evals")
                    if os.path.exists(stale_evals):
                        shutil.rmtree(stale_evals)
                    self.save_model(best_dir)
                    with open(os.path.join(best_dir, "best_step.json"), "w") as _f:
                        json.dump({
                            "step": self.state.global_step,
                            "model_utility": mu,
                            "forget_Q_A_Prob": fq,
                            "privleak": pl,
                            "util_component": util_c,
                            "forget_component": forget_c,
                            "priv_component": priv_c,
                            "joint_score": joint_score,
                        }, _f, indent=2)
                    logger.info(
                        "New best joint_score=%.4f (util=%.4f, forget=%.4f, priv=%.4f) "
                        "[mu=%.4f, forget_Q_A_Prob=%.4f, privleak=%.4f] at step %d → saved to %s",
                        joint_score, util_c, forget_c, priv_c,
                        mu, fq, pl, self.state.global_step, best_dir,
                    )
                return eval_metrics

        if eval_dataset is None:
            return {}
        # Run the default HF Trainer evaluate method when eval dataset is provided
        return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

    def train(self, *args, **kwargs):
        result = super().train(*args, **kwargs)
        if self.accelerator.is_local_main_process:
            run_dir = self._get_output_dir(trial=None)
            last_dir = os.path.join(run_dir, "last")
            # Remove stale evals before saving so they can't outlive the model
            stale_evals = os.path.join(last_dir, "evals")
            if os.path.exists(stale_evals):
                shutil.rmtree(stale_evals)
            self.save_model(last_dir)
            with open(os.path.join(last_dir, "last_step.json"), "w") as _f:
                json.dump({"step": self.state.global_step}, _f, indent=2)
            logger.info(
                "Final model at step %d → saved to %s",
                self.state.global_step, last_dir,
            )
        return result
