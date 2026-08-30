"""
Torch 2.4 + transformers 5.8 compatibility shim.

transformers/integrations/moe.py registers a ``torch.library.custom_op``
whose annotations are PEP-563 strings; torch 2.4's ``infer_schema``
does not resolve string annotations (fixed in torch >=2.5), so any
import path that touches ``transformers.modeling_utils`` crashes.

Importing this module patches ``torch._library.infer_schema.infer_schema``
(and the alias ``torch._custom_op.impl.infer_schema``) so it pre-resolves
string annotations via ``typing.get_type_hints`` before validating
parameter types. Must be imported BEFORE the first ``import transformers``.

Remove this module once the env upgrades torch >=2.5 (or downgrades
transformers <5).
"""

import typing

import torch._custom_op.impl as _impl_mod
import torch._library.infer_schema as _is_mod

_orig_infer = _is_mod.infer_schema


def _safe_infer(prototype_function, mutates_args=()):
    try:
        hints = typing.get_type_hints(prototype_function)
        for k, v in hints.items():
            prototype_function.__annotations__[k] = v
    except Exception:
        pass
    return _orig_infer(prototype_function, mutates_args)


_is_mod.infer_schema = _safe_infer
_impl_mod.infer_schema = _safe_infer


# ── transformers 5.8 renames: lm_eval still references old names ────────────
# In transformers 5.8, AutoModelForVision2Seq → AutoModelForImageTextToText.
# lm_eval 0.4.8 still imports the old name at module top, which crashes the
# import of any evaluator that pulls in lm_eval. Alias the old name onto the
# new class via a lazy attribute injection that runs the first time anything
# touches `transformers`.

def _install_transformers_aliases():
    """
    Inject old->new attribute aliases for transformers 5.8 renames so that
    older deps (e.g., lm_eval 0.4.8) keep working. We patch
    ``_LazyModule.__getattr__`` because setting via ``setattr`` does not
    survive transformers' lazy-loading machinery on every access path.

    Self-disables on transformers 4.x where the alias target doesn't exist
    (the old names are still the real ones), so the shim is safe to leave in
    place across env swaps.
    """
    try:
        import transformers
        from transformers.utils import import_utils as _iu
    except Exception:
        return

    # On transformers 4.x the alias target doesn't exist; leaving the shim
    # active would redirect lookups to a missing attribute and crash imports.
    if not hasattr(transformers, "AutoModelForImageTextToText"):
        return

    aliases = {
        # transformers 5.x removed AutoModelForVision2Seq; lm_eval still uses it
        "AutoModelForVision2Seq": "AutoModelForImageTextToText",
    }

    _orig_getattr = _iu._LazyModule.__getattr__

    def _aliased_getattr(self, name):
        target_name = aliases.get(name)
        if target_name is not None:
            return _orig_getattr(self, target_name)
        return _orig_getattr(self, name)

    _iu._LazyModule.__getattr__ = _aliased_getattr


_install_transformers_aliases()
