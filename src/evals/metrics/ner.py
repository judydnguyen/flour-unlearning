import logging
from typing import Dict, List, Set
from torch.utils.data import DataLoader
import numpy as np
from omegaconf import OmegaConf

from evals.metrics.base import unlearning_metric
from evals.metrics.utils import run_batchwise_evals, stop_sequences_criteria
from data.utils import IGNORE_INDEX

logger = logging.getLogger("evaluator")

# Entity types used for forget evaluation
ENTITY_TYPES = frozenset(
    {"PERSON", "GPE", "LOC", "ORG", "DATE", "NORP", "WORK_OF_ART", "EVENT"}
)


def _load_spacy():
    try:
        import spacy
    except ImportError:
        raise ImportError(
            "spaCy is required for entity_forget metric. "
            "Install with: pip install spacy && python -m spacy download en_core_web_sm"
        )
    for model_name in ("en_core_web_sm", "en_core_web_md", "en_core_web_lg"):
        try:
            return spacy.load(model_name)
        except OSError:
            continue
    raise OSError(
        "No spaCy English model found. Run: python -m spacy download en_core_web_sm"
    )


def _extract_entities(nlp, text: str) -> Dict[str, List[str]]:
    """Extract and normalize entities of target types from text."""
    doc = nlp(text)
    entities: Dict[str, Set[str]] = {etype: set() for etype in ENTITY_TYPES}
    for ent in doc.ents:
        if ent.label_ in ENTITY_TYPES:
            normalized = ent.text.lower().strip()
            if normalized:
                entities[ent.label_].add(normalized)
    # Only return types that have at least one entity
    return {etype: sorted(ents) for etype, ents in entities.items() if ents}


def _compute_entity_scores(
    gt_entities: Dict[str, List[str]], gen_entities: Dict[str, List[str]]
) -> Dict:
    """Compute precision, recall, f1, and entity_forget_score from entity dicts."""
    gt_set: Set = set()
    for etype, ents in gt_entities.items():
        for e in ents:
            gt_set.add((etype, e))

    gen_set: Set = set()
    for etype, ents in gen_entities.items():
        for e in ents:
            gen_set.add((etype, e))

    matched = gt_set & gen_set
    missed = gt_set - gen_set
    hallucinated = gen_set - gt_set

    gt_total = len(gt_set)
    gen_total = len(gen_set)
    matched_total = len(matched)

    precision = matched_total / gen_total if gen_total > 0 else 0.0
    recall = matched_total / gt_total if gt_total > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    entity_forget_score = 1.0 - recall

    return {
        "gt_entities": gt_entities,
        "gen_entities": gen_entities,
        "matched": [list(e) for e in sorted(matched)],
        "missed": [list(e) for e in sorted(missed)],
        "hallucinated": [list(e) for e in sorted(hallucinated)],
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "entity_forget_score": entity_forget_score,
    }


def _eval_entity_forget_batch(model, tokenizer, batch, generation_args, nlp):
    """Generate text and compute entity-level forget scores vs ground truth."""
    batch = {k: v.to(model.device) for k, v in batch.items()}
    input_ids = batch["input_ids"]
    labels = batch["labels"]

    input_texts = tokenizer.batch_decode(
        input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    tokens = [label[label != IGNORE_INDEX] for label in labels]
    full_texts = tokenizer.batch_decode(
        tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    ground_truths = [
        full_text.replace(input_text, "").strip()
        for input_text, full_text in zip(input_texts, full_texts)
    ]

    attention_mask = batch["attention_mask"]
    generation_args = OmegaConf.to_container(generation_args, resolve=True)
    stopwords = generation_args.pop("stopwords", None)
    if stopwords is not None:
        sc = stop_sequences_criteria(
            tokenizer, stopwords, input_ids.shape[1], input_ids.shape[0]
        )
        generation_args["stopping_criteria"] = sc

    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        **generation_args,
        pad_token_id=tokenizer.eos_token_id,
    )
    gen_texts = tokenizer.batch_decode(
        output[:, input_ids.shape[-1] :],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    # Cut off at stopwords (mirrors eval_text_similarity)
    all_stopwords = [tokenizer.decode([tokenizer.eos_token_id])] + (
        stopwords if stopwords else []
    )
    for i in range(len(gen_texts)):
        raw = gen_texts[i]
        for word in all_stopwords:
            if word and word in raw:
                raw = raw.split(word)[0]
        gen_texts[i] = raw.strip()

    scores = []
    for gt_text, gen_text, input_text in zip(ground_truths, gen_texts, input_texts):
        gt_entities = _extract_entities(nlp, gt_text)
        gen_entities = _extract_entities(nlp, gen_text)
        result = _compute_entity_scores(gt_entities, gen_entities)
        result["input"] = input_text
        result["ground_truth"] = gt_text
        result["generation"] = gen_text
        scores.append(result)

    return scores


@unlearning_metric(name="entity_forget")
def entity_forget(model, **kwargs):
    """Compute entity-level forget score: fraction of GT named entities not reproduced
    by the model.  entity_forget_score = 1 - recall.  Higher = better forgetting.

    Extracted entity types: PERSON, GPE, LOC, ORG, DATE, NORP, WORK_OF_ART, EVENT.
    Entities are lowercased and deduplicated before comparison.
    """
    tokenizer = kwargs["tokenizer"]
    data = kwargs["data"]
    collator = kwargs["collators"]
    batch_size = kwargs["batch_size"]
    generation_args = kwargs["generation_args"]

    nlp = _load_spacy()
    dataloader = DataLoader(data, batch_size=batch_size, collate_fn=collator)

    fun_args = {"tokenizer": tokenizer, "generation_args": generation_args, "nlp": nlp}
    scores_by_index = run_batchwise_evals(
        model,
        dataloader,
        _eval_entity_forget_batch,
        fun_args,
        "Calculating entity forget score",
    )

    forget_scores = np.array(
        [
            evals["entity_forget_score"]
            for evals in scores_by_index.values()
            if evals.get("entity_forget_score") is not None
        ]
    )
    agg = float(np.mean(forget_scores)) if len(forget_scores) > 0 else 0.0
    return {"agg_value": agg, "value_by_index": scores_by_index}
