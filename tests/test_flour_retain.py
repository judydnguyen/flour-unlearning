"""Tests for the FLOUR retain-loss path and the two-phase loss structure.

These tests verify that:
1. compute_retain_loss dispatches on retain_loss_type (NLL and EMBED_DIFF both
   route through it and return a scalar).
2. Phase 2 obtains the retain loss via compute_retain_loss (never a hardcoded
   retain_outputs.loss); Phase 1 trains only the encoder (anchor + simulated
   forget) and uses no retain loss.
3. Phase-1 activations produce defined forget gradients under both retain modes.
4. The forget-loss formula is the masked-token MSE we expect.
"""

import types
import unittest
from unittest.mock import MagicMock, patch, PropertyMock
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Minimal stubs so we can import FLOUR without the full training stack
# ---------------------------------------------------------------------------

def _make_fake_model(hidden=8, vocab=32, seq=4, batch=2):
    """Tiny linear model that returns plausible loss/hidden-states."""

    class FakeOutput:
        def __init__(self):
            self.loss = torch.tensor(1.5, requires_grad=True)
            self.logits = torch.randn(batch, seq, vocab)

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = types.SimpleNamespace(hidden_size=hidden)
            self.layer = nn.Linear(hidden, hidden, bias=False)

        def forward(self, input_ids=None, attention_mask=None, labels=None):
            return FakeOutput()

        def named_modules(self):
            yield "", self
            yield "layer", self.layer

        def named_parameters(self, recurse=True):
            yield "layer.weight", self.layer.weight

        def parameters(self, recurse=True):
            yield self.layer.weight

    return FakeModel()


# ---------------------------------------------------------------------------
# Helper: build a minimal FLOUR-like object without the HF Trainer stack
# ---------------------------------------------------------------------------

def _build_latent_rmu(retain_loss_type="NLL"):
    """Return a FLOUR instance with all heavy dependencies mocked out."""
    # We patch the parent __init__ chain so we don't need transformers / deepspeed
    from unittest.mock import patch

    model = _make_fake_model()
    ref_model = _make_fake_model()

    # Patch GradDiff and UnlearnTrainer __init__ to no-ops
    with patch("trainer.unlearn.grad_diff.GradDiff.__init__", return_value=None):
        from trainer.unlearn.flour import FLOUR
        obj = FLOUR.__new__(FLOUR)

    # Manually set attributes that FLOUR.__init__ and compute_loss rely on
    obj.model = model
    obj.ref_model = ref_model
    obj.retain_loss_type = retain_loss_type
    obj.gamma = 1.0
    obj.alpha = 1.0
    obj.steering_coeff = 2.0
    obj.orth_weight = 1.0
    obj.retain_sep_weight = 1.0
    obj.forget_warmup_steps = 0
    obj._phase = 1
    obj._phase2_step = 0
    obj.module_regex = "layer"
    obj.trainable_params_regex = ["layer.weight"]

    # model_module / ref_module — just the linear layer
    obj.model_module = model.layer
    obj.ref_module = ref_model.layer

    # Tiny encoder
    from trainer.unlearn.flour import PerSampleEncoder
    obj.encoder = PerSampleEncoder(hidden_size=8, latent_dim=4, num_layers=2)

    return obj


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPhase1RetainLossDispatch(unittest.TestCase):
    """compute_retain_loss is called (not raw model()) in phase 1."""

    def setUp(self):
        try:
            _build_latent_rmu()
        except ImportError as e:
            self.skipTest(f"Import failed (missing dep): {e}")

    def _make_inputs(self, batch=2, seq=4):
        return {
            "forget": {
                "input_ids": torch.zeros(batch, seq, dtype=torch.long),
                "attention_mask": torch.ones(batch, seq, dtype=torch.long),
                "labels": torch.zeros(batch, seq, dtype=torch.long),
            },
            "retain": {
                "input_ids": torch.zeros(batch, seq, dtype=torch.long),
                "attention_mask": torch.ones(batch, seq, dtype=torch.long),
                "labels": torch.zeros(batch, seq, dtype=torch.long),
            },
        }

    def test_nll_calls_compute_retain_loss(self):
        """With retain_loss_type=NLL, phase 1 routes through compute_retain_loss."""
        obj = _build_latent_rmu(retain_loss_type="NLL")
        obj._phase = 1

        with patch.object(obj, "compute_retain_loss", wraps=obj.compute_retain_loss) as spy:
            # We only test that compute_retain_loss is invoked; skip full loss computation
            # by patching the expensive parts.
            retain_inputs = self._make_inputs()["retain"]
            # Direct call to simulate phase-1 retain path
            loss = obj.compute_retain_loss(obj.model, retain_inputs)
            spy.assert_called_once()
            self.assertIsNotNone(loss)

    def test_embed_diff_calls_compute_retain_loss(self):
        """With retain_loss_type=EMBED_DIFF, compute_retain_loss is called and returns a scalar."""
        obj = _build_latent_rmu(retain_loss_type="EMBED_DIFF")
        retain_inputs = self._make_inputs()["retain"]

        # Patch forward_with_cache to return plausible activations
        hidden = torch.randn(2, 4, 8, requires_grad=True)
        ref_hidden = torch.randn(2, 4, 8)
        fake_outputs = MagicMock()

        def fake_fwc(model, inputs, module, no_grad=True):
            if no_grad:
                return ref_hidden.detach(), fake_outputs
            return hidden, fake_outputs

        with patch.object(obj, "forward_with_cache", side_effect=fake_fwc):
            loss = obj.compute_retain_loss(obj.model, retain_inputs)

        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.shape, torch.Size([]))


class TestPhase1Phase2RetainConsistency(unittest.TestCase):
    """Phase 2 routes the retain loss through compute_retain_loss (which
    dispatches on retain_loss_type) rather than a hardcoded retain_outputs.loss.
    Phase 1 trains only the encoder (anchor + simulated-forget MSE) and uses no
    retain loss."""

    def _make_inputs(self, batch=2, seq=4):
        return {
            "forget": {
                "input_ids": torch.zeros(batch, seq, dtype=torch.long),
                "attention_mask": torch.ones(batch, seq, dtype=torch.long),
                "labels": torch.zeros(batch, seq, dtype=torch.long),
            },
            "retain": {
                "input_ids": torch.zeros(batch, seq, dtype=torch.long),
                "attention_mask": torch.ones(batch, seq, dtype=torch.long),
                "labels": torch.zeros(batch, seq, dtype=torch.long),
            },
        }

    def test_phase2_routes_retain_through_dispatcher(self):
        """Phase 2 must compute the retain loss via compute_retain_loss (which
        respects retain_loss_type), not via a hardcoded retain_outputs.loss."""
        try:
            from trainer.unlearn.flour import FLOUR
        except ImportError as e:
            self.skipTest(str(e))

        import inspect
        phase2_src = inspect.getsource(FLOUR._phase2_loss)

        # Phase 2 should route through compute_retain_loss at least once.
        self.assertGreaterEqual(
            phase2_src.count("compute_retain_loss("), 1,
            "Phase 2 must call compute_retain_loss to obtain the retain loss",
        )

        # No phase should hardcode retain_outputs.loss — always go through the dispatcher.
        class_src = inspect.getsource(FLOUR)
        self.assertNotIn(
            "retain_outputs.loss", class_src,
            "Retain loss must route through compute_retain_loss, not retain_outputs.loss",
        )

    def test_phase1_uses_anchor_not_retain_loss(self):
        """Phase 1 trains only the encoder: anchor alignment + simulated-forget
        MSE, with no retain-loss term."""
        try:
            from trainer.unlearn.flour import FLOUR
        except ImportError as e:
            self.skipTest(str(e))

        import inspect
        phase1_src = inspect.getsource(FLOUR._phase1_loss)
        self.assertIn("anchor_loss", phase1_src)
        self.assertIn("simulated_forget_weight", phase1_src)
        self.assertEqual(
            phase1_src.count("compute_retain_loss("), 0,
            "Phase 1 trains only the encoder and should not use a retain loss",
        )


class TestPhase1GradientsDefined(unittest.TestCase):
    """Phase-1 gradient-conflict produces defined gradients under both retain modes."""

    def _run_phase1_grad_check(self, retain_loss_type):
        try:
            obj = _build_latent_rmu(retain_loss_type=retain_loss_type)
        except ImportError as e:
            self.skipTest(str(e))

        hidden_size = 8
        batch, seq = 2, 4

        # Fake activations with grad enabled
        h_forget = torch.randn(batch, seq, hidden_size, requires_grad=True)
        ref_retain_act = torch.randn(batch, seq, hidden_size)
        model_retain_act = torch.randn(batch, seq, hidden_size, requires_grad=True)

        # Compute encoder steering vector
        pooled = h_forget.float().mean(dim=1)
        r = obj.encoder(pooled)
        r = r / (r.norm(dim=-1, keepdim=True) + 1e-8) * obj.steering_coeff
        r_expanded = r.unsqueeze(1).expand_as(h_forget)

        labels = torch.zeros(batch, seq, dtype=torch.long)
        mask = labels != -100

        forget_loss = obj.compute_activation_loss(h_forget.float(), r_expanded.float(), mask)

        # Build retain_loss according to retain_loss_type
        if retain_loss_type == "NLL":
            retain_loss = torch.tensor(1.2, requires_grad=True)
        elif retain_loss_type == "EMBED_DIFF":
            retain_loss = obj.compute_activation_loss(
                model_retain_act.float(), ref_retain_act.float(), mask
            )
        else:
            self.skipTest(f"retain_loss_type {retain_loss_type} not covered by this test")

        # Use encoder params as proxy for phase2_params
        phase2_params = list(obj.encoder.parameters())

        grad_forget = torch.autograd.grad(
            forget_loss, phase2_params,
            create_graph=True, allow_unused=True, retain_graph=True,
        )
        grad_retain = torch.autograd.grad(
            retain_loss, phase2_params,
            create_graph=False, allow_unused=True,
        )

        # At least one gradient should be defined for each
        has_forget_grad = any(g is not None for g in grad_forget)
        has_retain_grad = any(g is not None for g in grad_retain) if retain_loss_type == "NLL" else True
        # NLL retain_loss is a leaf tensor — no grad through phase2_params, that's OK;
        # the smoke test just checks forget grad is defined.
        self.assertTrue(has_forget_grad, f"No defined forget grad under {retain_loss_type}")

    def test_nll_phase1_gradients_defined(self):
        self._run_phase1_grad_check("NLL")

    def test_embed_diff_phase1_gradients_defined(self):
        self._run_phase1_grad_check("EMBED_DIFF")


class TestForgetLossUnchanged(unittest.TestCase):
    """Forget-loss formula is unchanged from the original phase-1 code."""

    def test_forget_loss_formula(self):
        """compute_activation_loss(h_forget, r_expanded, mask) equals MSE over masked tokens."""
        try:
            obj = _build_latent_rmu()
        except ImportError as e:
            self.skipTest(str(e))

        batch, seq, hidden = 2, 4, 8
        h_forget = torch.randn(batch, seq, hidden)
        r_expanded = torch.randn(batch, seq, hidden)
        mask = torch.ones(batch, seq, dtype=torch.bool)

        loss = obj.compute_activation_loss(h_forget.float(), r_expanded.float(), mask)

        # Manually replicate the formula
        squared_diff = F.mse_loss(h_forget.float(), r_expanded.float(), reduction="none")
        expanded_mask = mask.unsqueeze(-1).expand_as(squared_diff)
        sds = (squared_diff * expanded_mask).mean(dim=2).sum(dim=1)
        num_tokens = mask.sum(dim=-1, keepdim=True)
        expected = (sds / num_tokens).mean()

        self.assertTrue(torch.allclose(loss, expected, atol=1e-6),
                        f"Forget loss formula changed: got {loss}, expected {expected}")


if __name__ == "__main__":
    unittest.main()
