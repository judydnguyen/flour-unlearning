"""
Attack implementations.
"""

import torch
from transformers import AutoModelForCausalLM

from evals.metrics.base import unlearning_metric
from evals.metrics.mia.loss import LOSSAttack
from evals.metrics.mia.min_k import MinKProbAttack
from evals.metrics.mia.min_k_plus_plus import MinKPlusPlusAttack
from evals.metrics.mia.gradnorm import GradNormAttack
from evals.metrics.mia.zlib import ZLIBAttack
from evals.metrics.mia.reference import ReferenceAttack

from evals.metrics.mia.utils import mia_auc
import logging

logger = logging.getLogger("metrics")

## NOTE: all MIA attack statistics are signed as required in order to show the
# same trends as loss (higher the score on an example, less likely the membership)


@unlearning_metric(name="mia_loss")
def mia_loss(model, **kwargs):
    return mia_auc(
        LOSSAttack,
        model,
        data=kwargs["data"],
        collator=kwargs["collators"],
        batch_size=kwargs["batch_size"],
    )


@unlearning_metric(name="mia_min_k")
def mia_min_k(model, **kwargs):
    return mia_auc(
        MinKProbAttack,
        model,
        data=kwargs["data"],
        collator=kwargs["collators"],
        batch_size=kwargs["batch_size"],
        k=kwargs["k"],
    )


@unlearning_metric(name="mia_min_k_plus_plus")
def mia_min_k_plus_plus(model, **kwargs):
    return mia_auc(
        MinKPlusPlusAttack,
        model,
        data=kwargs["data"],
        collator=kwargs["collators"],
        batch_size=kwargs["batch_size"],
        k=kwargs["k"],
    )


@unlearning_metric(name="mia_gradnorm")
def mia_gradnorm(model, **kwargs):
    return mia_auc(
        GradNormAttack,
        model,
        data=kwargs["data"],
        collator=kwargs["collators"],
        batch_size=kwargs["batch_size"],
        p=kwargs["p"],
    )


@unlearning_metric(name="mia_zlib")
def mia_zlib(model, **kwargs):
    return mia_auc(
        ZLIBAttack,
        model,
        data=kwargs["data"],
        collator=kwargs["collators"],
        batch_size=kwargs["batch_size"],
        tokenizer=kwargs.get("tokenizer"),
    )


@unlearning_metric(name="mia_reference")
def mia_reference(model, **kwargs):
    if "reference_model_path" not in kwargs:
        raise ValueError("Reference model must be provided in kwargs")
    logger.info(f"Loading reference model from {kwargs['reference_model_path']}")
    reference_model = AutoModelForCausalLM.from_pretrained(
        kwargs["reference_model_path"],
        torch_dtype=model.dtype,
        device_map={"": model.device},
    )
    # Guard against vocab mismatch: feeding tokens with IDs beyond the reference
    # model's vocab triggers an async CUDA indexSelect assert in the embedding
    # layer (e.g. Llama-3.2 vocab 128k tokens routed into a Llama-2 7B reference
    # with vocab 32k). Skip with a warning so the rest of the eval suite still
    # completes; user can supply a compatible reference_model_path to enable it.
    model_vocab = getattr(getattr(model, "config", None), "vocab_size", None)
    ref_vocab = getattr(reference_model.config, "vocab_size", None)
    if model_vocab is not None and ref_vocab is not None and ref_vocab < model_vocab:
        logger.warning(
            f"Skipping mia_reference: reference model vocab_size={ref_vocab} is "
            f"smaller than target model vocab_size={model_vocab}. The reference "
            f"checkpoint at '{kwargs['reference_model_path']}' is incompatible "
            f"with this tokenizer; pass a same-architecture reference via "
            f"`reference_model_path` or drop mia_reference from the eval list."
        )
        del reference_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {"auc": float("nan"), "agg_value": float("nan"), "skipped": True}
    return mia_auc(
        ReferenceAttack,
        model,
        data=kwargs["data"],
        collator=kwargs["collators"],
        batch_size=kwargs["batch_size"],
        reference_model=reference_model,
    )
