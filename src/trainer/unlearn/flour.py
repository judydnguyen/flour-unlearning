import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import deepspeed
from trainer.unlearn.grad_diff import GradDiff

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MPL_OK = True
except Exception:
    _MPL_OK = False


# Phase 1 metric groups for the diagnostic plot. Each row is one subplot.
# Keys must match what _phase1_loss logs via self.log({...}).
_PHASE1_PLOT_GROUPS = [
    ("Loss components", [
        ("phase1/anchor_loss",     "anchor_loss = 1 − cos(r, dontknow)", "tab:blue"),
        ("phase1/forget_loss_sim", "forget MSE (Phase-2 simulated)",     "tab:orange"),
    ]),
    ("Encoder alignment with dontknow", [
        ("phase1/cos_r_dontknow", "cos(r, dontknow)", "tab:green"),
    ]),
    ("Phase-2 displacement at init", [
        ("phase1/displacement_h_to_target", "||target − h_forget||", "tab:gray"),
    ]),
    ("Encoder direction magnitude", [
        ("phase1/r_norm", "||r|| (pre-normalize)", "tab:purple"),
    ]),
]


def _refresh_phase1_plot(run_dir, log_history):
    """Plot Phase 1 diagnostics → {run_dir}/phase1_losses.png.

    Reads only entries that contain phase1/* keys; skips Phase 2 / eval rows.
    Safe to call repeatedly — write is cheap relative to a train step.
    """
    if not _MPL_OK or not log_history or not run_dir:
        return
    rows = [h for h in log_history if any(k.startswith("phase1/") for k in h)]
    if not rows:
        return
    xs = [h.get("epoch", i) for i, h in enumerate(rows)]
    n = len(_PHASE1_PLOT_GROUPS)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.4 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, (title, series) in zip(axes, _PHASE1_PLOT_GROUPS):
        plotted_any = False
        for key, label, color in series:
            vals = [h.get(key) for h in rows]
            if not any(v is not None for v in vals):
                continue
            ax.plot(xs, vals, label=label, color=color, marker=".", markersize=3, linewidth=1)
            plotted_any = True
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3)
        if plotted_any:
            ax.legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("Epoch (Phase 1)")
    fig.suptitle(f"Phase 1 training trace — {os.path.basename(run_dir)}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(run_dir, "phase1_losses.png"), dpi=120)
    plt.close(fig)


class PerSampleEncoder(nn.Module):
    """Residual MLP encoder that produces a steering direction.

    forward: r = alpha * h_pooled + (1 - alpha) * delta(h_pooled)
    Last layer of delta is zero-initialized so r = alpha * h_pooled at init,
    a meaningful (non-zero, non-NaN) starting direction.
    """
    def __init__(self, hidden_size: int, latent_dim: int = 256, num_layers: int = 2, alpha: float = 0.2):
        super().__init__()
        assert num_layers >= 2, "num_layers must be at least 2"
        self.alpha = alpha
        layers = [nn.Linear(hidden_size, latent_dim), nn.GELU()]
        for _ in range(num_layers - 2):
            layers += [nn.Linear(latent_dim, latent_dim), nn.GELU()]
        layers.append(nn.Linear(latent_dim, hidden_size))
        self.net = nn.Sequential(*layers)

        # In __init__, REMOVE the nn.init.zeros_ lines, OR replace with:
        nn.init.normal_(self.net[-1].weight, std=0.02)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h_pooled: torch.Tensor) -> torch.Tensor:
        delta = self.net(h_pooled)
        return self.alpha * h_pooled + (1 - self.alpha) * delta
        # return delta


class FLOUR(GradDiff):
    def __init__(
        self,
        module_regex=r"model\.layers\.7",
        trainable_params_regex=(r"model\.layers\.(5|6|7)\.mlp\.down_proj\.weight",),
        steering_coeff=20,
        latent_dim=256,
        encoder_epochs=2,
        encoder_layers=2,
        encoder_alpha=0.5,
        orth_weight=1.0,              # deprecated: grad_conflict term removed; kept for config compat
        anchor_weight=1.0,
        anchor_mode="blank",          # blank | refusal | corrupt | delta — what the anchor steers toward
        ablation_token="unk",         # token that replaces the answer span in blank/delta modes: unk|pad|mask|eos|bos|<int>|<literal>
        refusal_text=" I don't know. I cannot recall that information.",
        simulated_forget_weight=1.0,  # weight for Phase-2-simulated forget MSE in Phase-1 loss
        encoder_warmup_steps=0,       # pre-Phase-1 steps aligning r with dontknow
        encoder_lr=1e-3,
        forget_warmup_steps=0,
        coeff_warmup_steps=0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if self.ref_model is None:
            self.ref_model = self._prepare_ref_model(self.model)

        self.trainable_params_regex = list(trainable_params_regex)
        self.module_regex = module_regex
        self.model_module = self._get_matching_module(self.model, self.module_regex)
        self.ref_module = self._get_matching_module(self.ref_model, self.module_regex)
        self.steering_coeff = steering_coeff
        self.encoder_epochs = encoder_epochs
        self.orth_weight = orth_weight  # unused; kept so existing configs don't fail
        self.anchor_weight = anchor_weight
        self.anchor_mode = anchor_mode
        self.ablation_token = ablation_token
        self.refusal_text = refusal_text
        self._refusal_ids = None
        self.simulated_forget_weight = simulated_forget_weight
        self.encoder_warmup_steps = encoder_warmup_steps
        self.encoder_lr = encoder_lr
        self.forget_warmup_steps = forget_warmup_steps
        self.coeff_warmup_steps = coeff_warmup_steps
        self._phase = 1
        self._phase2_step = 0

        hidden_size = self.model.config.hidden_size
        self.encoder = PerSampleEncoder(hidden_size, latent_dim, encoder_layers, encoder_alpha)
        self.encoder.to(next(self.model.parameters()).device)

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _warmup_encoder(self, num_steps):
        """Pre-align encoder direction r with the dontknow direction.

        Runs `num_steps` of gradient descent on (1 - cos(r, dontknow))
        using forget batches from the trainer's dataset. Only encoder
        params are updated — model is forwarded in no_grad.
        Purpose: start Phase 1 with r already on the right manifold,
        so the joint loss converges instead of drifting (the issue we
        saw where anchor_loss climbed 0.01 → 0.30 during Phase 1).
        """
        from torch.utils.data import DataLoader
        if self.train_dataset is None or num_steps <= 0:
            return
        print(f"[encoder-warmup] aligning r with dontknow, {num_steps} steps", flush=True)
        self.encoder.train()
        opt = torch.optim.AdamW(self.encoder.parameters(), lr=self.encoder_lr)
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            collate_fn=self.data_collator,
            shuffle=True,
        )
        device = next(self.encoder.parameters()).device

        step = 0
        last_loss = None
        for batch in loader:
            if step >= num_steps:
                break
            forget_inputs = {
                k: batch["forget"][k].to(device)
                for k in ("input_ids", "attention_mask", "labels")
            }
            with torch.no_grad():
                h_forget, _ = self.forward_with_cache(
                    self.model, forget_inputs, self.model_module, no_grad=True
                )
            pooled = self._masked_mean(h_forget.float(), forget_inputs["attention_mask"])
            r = self.encoder(pooled)
            r_normed = r / (r.norm(dim=-1, keepdim=True) + 1e-8)
            dontknow = self._get_dontknow_activations(forget_inputs)
            anchor_loss = 1 - (r_normed * dontknow).sum(dim=-1).mean()
            opt.zero_grad()
            anchor_loss.backward()
            opt.step()
            last_loss = anchor_loss.item()
            if step % 10 == 0:
                print(f"[encoder-warmup] step {step:3d}/{num_steps}: 1 - cos(r, dontknow) = {last_loss:.4f}", flush=True)
            step += 1
        print(f"[encoder-warmup] done. final 1 - cos(r, dontknow) = {last_loss:.4f}", flush=True)

    def _resolve_ablation_token_id(self):
        """Token id used to overwrite the answer span in blank/delta anchor modes.

        Configurable via `ablation_token` (default "unk", the original behavior).
        Accepts a named special token (unk|pad|mask|eos|bos), an int id, or any
        literal string (first sub-token used). Falls back to unk→pad if the
        requested special token is undefined for this tokenizer.
        """
        tok = getattr(self, "ablation_token", "unk")
        if isinstance(tok, int):
            return tok
        name = str(tok).lower()
        named = {
            "unk": self.tokenizer.unk_token_id,
            "pad": self.tokenizer.pad_token_id,
            "mask": getattr(self.tokenizer, "mask_token_id", None),
            "eos": self.tokenizer.eos_token_id,
            "bos": self.tokenizer.bos_token_id,
        }
        if name in named:
            tid = named[name]
        else:
            ids = self.tokenizer(str(tok), add_special_tokens=False).input_ids
            tid = ids[0] if ids else None
        if tid is None:
            tid = self.tokenizer.unk_token_id or self.tokenizer.pad_token_id
        return tid

    def _get_dontknow_activations(self, forget_inputs):
        """Build a 'proxy' forget sequence and return its mean-pooled, L2-normalized
        layer-L activation from the frozen reference model = the anchor direction.

        anchor_mode controls what the proxy is (ablation):
          blank   — answer tokens → [UNK]/pad        (original; answer-ablated anchor)
          refusal — answer tokens → a refusal phrase (genuine 'don't-know' target)
          corrupt — question/prompt tokens → random  (input-corruption target)
          delta   — answer-ablation *difference* direction (see below)

        delta returns normalize(mean_answer(h_blank) − mean_answer(h_full)),
        pooled over answer tokens. Unlike the absolute modes above, this is the
        *change* the ablation induces — verified (scripts/diag_null_direction.py)
        to be large, ~1-D across examples, and ~orthogonal to the blank anchor.
        """
        input_ids = forget_inputs["input_ids"].clone()
        labels = forget_inputs["labels"]
        attn = forget_inputs["attention_mask"]
        answer_mask = labels != -100
        mode = getattr(self, "anchor_mode", "blank")

        if mode == "delta":
            # Answer-ablation *difference* (null) direction. Pool over answer
            # tokens, where the change actually lives. input_ids is still the
            # un-ablated clone here, so it doubles as h_full's input.
            blank_ids = input_ids.clone()
            blank_ids[answer_mask] = self._resolve_ablation_token_id()
            with torch.no_grad():
                h_blank, _ = self.forward_with_cache(
                    self.ref_model, {"input_ids": blank_ids, "attention_mask": attn},
                    self.ref_module, no_grad=True,
                )
                h_full, _ = self.forward_with_cache(
                    self.ref_model, {"input_ids": input_ids, "attention_mask": attn},
                    self.ref_module, no_grad=True,
                )
            delta = self._masked_mean(h_blank.float(), answer_mask) - self._masked_mean(
                h_full.float(), answer_mask
            )
            return F.normalize(delta, dim=-1)

        if mode == "blank":
            input_ids[answer_mask] = self._resolve_ablation_token_id()
        elif mode == "refusal":
            if self._refusal_ids is None:
                self._refusal_ids = torch.tensor(
                    self.tokenizer(self.refusal_text, add_special_tokens=False).input_ids,
                    device=input_ids.device, dtype=input_ids.dtype)
            rl = len(self._refusal_ids)
            for i in range(input_ids.shape[0]):
                pos = answer_mask[i].nonzero(as_tuple=True)[0]
                if len(pos) == 0:
                    continue
                input_ids[i, pos] = self._refusal_ids[torch.arange(len(pos), device=pos.device) % rl]
        elif mode == "corrupt":
            special = {self.tokenizer.bos_token_id, self.tokenizer.eos_token_id,
                       self.tokenizer.pad_token_id, self.tokenizer.unk_token_id}
            special.discard(None)
            prompt_mask = (~answer_mask) & attn.bool()      # question/prompt, real tokens
            for sid in special:
                prompt_mask &= (input_ids != sid)
            n = int(prompt_mask.sum())
            if n > 0:
                input_ids[prompt_mask] = torch.randint(
                    0, self.model.config.vocab_size, (n,), device=input_ids.device)
        else:
            raise ValueError(f"unknown anchor_mode={mode!r} (use blank|refusal|corrupt)")

        proxy_inputs = {"input_ids": input_ids, "attention_mask": attn}
        with torch.no_grad():
            h, _ = self.forward_with_cache(
                self.ref_model, proxy_inputs, self.ref_module, no_grad=True
            )
        return F.normalize(self._masked_mean(h.float(), attn), dim=-1)

    def _get_matching_module(self, model, module_regex):
        if isinstance(model, deepspeed.DeepSpeedEngine):
            model = model.module
        matched = {n: m for n, m in model.named_modules() if re.fullmatch(module_regex, n)}
        if len(matched) != 1:
            raise ValueError(f"Expected 1 module match for {module_regex}, got {len(matched)}")
        return next(iter(matched.values()))

    def _freeze_all_params(self, model, requires_grad=False):
        for p in model.parameters():
            p.requires_grad = requires_grad

    def _set_trainable_params(self, model, trainable_params_regex, requires_grad=True):
        for name, param in model.named_parameters():
            if any(re.fullmatch(pat, name) for pat in trainable_params_regex):
                param.requires_grad = requires_grad

    def _get_phase2_params(self):
        return [
            p for n, p in self.model.named_parameters()
            if any(re.fullmatch(pat, n) for pat in self.trainable_params_regex)
            and p.requires_grad
        ]

    def forward_with_cache(self, model, inputs, module, no_grad=True):
        cache = []
        def hook(_m, _i, output):
            cache.append(output[0] if isinstance(output, tuple) else output)
        handle = module.register_forward_hook(hook)
        try:
            with torch.set_grad_enabled(not no_grad):
                outputs = model(**inputs)
        finally:
            handle.remove()
        return cache[0], outputs

    def compute_activation_loss(self, activation1, activation2, mask):
        squared_diff = F.mse_loss(activation1, activation2, reduction="none")
        expanded_mask = mask.unsqueeze(-1).expand_as(squared_diff)
        squared_diff_sum = (squared_diff * expanded_mask).mean(dim=2).sum(dim=1)
        num_tokens = mask.sum(dim=-1, keepdim=True)
        return (squared_diff_sum / num_tokens).mean()

    def _masked_mean(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Mean-pool h:(B,T,H) over real (non-pad) tokens given mask:(B,T).

        The collator pads dynamically to the longest sequence in each batch, so
        pooling over all positions (h.mean(dim=1)) averages in a batch-dependent
        number of pad activations — which made the encoder direction and the
        dontknow anchor shift with batch composition. Masking removes that
        dependence so the anchor is a stable function of the example alone.
        """
        m = mask.unsqueeze(-1).to(h.dtype)
        return (h * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)

    # ------------------------------------------------------------------ #
    #  Two-phase training                                                  #
    # ------------------------------------------------------------------ #

    def evaluate(self, *args, **kwargs):
        if self._phase == 1:
            return {}
        return super().evaluate(*args, **kwargs)

    def train(self, resume_from_checkpoint=None, **kwargs):
        original_epochs = self.args.num_train_epochs

        self._phase = 1
        # Phase 1 only updates the encoder — keep the entire model frozen.
        # (Skipping the trainable_params_regex unfreeze here saves the
        # gradient buffer on all matched layers; with regex=".*" that's
        # the whole 1B model, which is what OOM'd phase 1 on forget10.)
        self._freeze_all_params(self.model, requires_grad=False)

        if self.encoder_warmup_steps > 0:
            self._warmup_encoder(self.encoder_warmup_steps)

        self.args.num_train_epochs = self.encoder_epochs
        super().train(resume_from_checkpoint=resume_from_checkpoint, **kwargs)

        if self.args.output_dir:
            try:
                _refresh_phase1_plot(self.args.output_dir, self.state.log_history)
                print(f"[phase1] wrote {self.args.output_dir}/phase1_losses.png", flush=True)
            except Exception as e:
                print(f"[phase1] plot failed: {e}", flush=True)

        self._phase = 2
        self._freeze_all_params(self.encoder, requires_grad=False)
        self.encoder.eval()
        self.optimizer = None
        self.lr_scheduler = None
        self.args.num_train_epochs = original_epochs - self.encoder_epochs
        super().train(**kwargs)

        self.args.num_train_epochs = original_epochs

    def create_optimizer(self):
        if self._phase == 1:
            # Phase 1: only encoder is trainable. Model fully frozen — no
            # gradient buffer required on any model parameter.
            self._freeze_all_params(self.model, requires_grad=False)
            optimizer_cls, _ = self.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(
                self.encoder.parameters(),
                lr=self.encoder_lr,
            )
        else:
            self._freeze_all_params(self.encoder, requires_grad=False)
            self._freeze_all_params(self.model, requires_grad=False)
            self._set_trainable_params(self.model, self.trainable_params_regex, True)
            super().create_optimizer()

    # ------------------------------------------------------------------ #
    #  Steering vector — single source of truth, used by BOTH phases     #
    # ------------------------------------------------------------------ #

    def _ramped_steering_coeff(self) -> float:
        if self.coeff_warmup_steps <= 0:
            return self.steering_coeff
        return self.steering_coeff * max(0.0, 1.0 - self._phase2_step / self.coeff_warmup_steps)

    def _compute_steering(self, h_forget: torch.Tensor, steering_coeff: float, mask: torch.Tensor):
        """Returns (r_normed, r_scaled_unsqueezed).

        r_normed: (B, H), unit-norm encoder direction. Carries gradient through encoder.
        r_scaled_unsqueezed: (B, 1, H), ready to broadcast across sequence positions.

        `mask` is the (B, T) attention mask; pooling is masked so the encoder
        input lives in the same space as the masked anchor and neither drifts
        with batch padding.
        """
        pooled = self._masked_mean(h_forget.float(), mask)
        r = self.encoder(pooled)
        r_normed = r / (r.norm(dim=-1, keepdim=True) + 1e-8)
        r_scaled = r_normed.unsqueeze(1).to(h_forget.dtype) * steering_coeff
        return r_normed, r_scaled

    def _build_steering_target(self, h_forget_ref: torch.Tensor, r_scaled: torch.Tensor) -> torch.Tensor:
        """The target Phase 2 actually pulls toward, and Phase 1 simulates against.

        target = h_forget_ref + steering_coeff * r_normed
        """
        return h_forget_ref + r_scaled

    # ------------------------------------------------------------------ #
    #  Loss computation                                                    #
    # ------------------------------------------------------------------ #

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # transformers 4.46+ passes num_items_in_batch into compute_loss; we
        # don't use it here (loss is already mean-reduced), so swallow it via
        # **kwargs — tolerate extra trainer kwargs passed via the registry.
        forget_inputs = {k: inputs["forget"][k] for k in ("input_ids", "attention_mask", "labels")}
        retain_inputs = {k: inputs["retain"][k] for k in ("input_ids", "attention_mask", "labels")}
        mask = forget_inputs["labels"] != -100

        if self._phase == 1:
            return self._phase1_loss(forget_inputs, retain_inputs, mask)

        loss, forget_outputs = self._phase2_loss(model, forget_inputs, retain_inputs, mask)
        return (loss, forget_outputs) if return_outputs else loss

    # ---- Phase 1 ----------------------------------------------------- #

    def _phase1_loss(self, forget_inputs, retain_inputs, mask):
        # Phase 1 trains ONLY the encoder; the model is frozen. We don't
        # need gradient through the model — gradient flows to the encoder
        # via `target` (which depends on r). Forwarding with no_grad drops
        # the saved-activation memory for the entire forget pass.
        with torch.no_grad():
            h_forget, _ = self.forward_with_cache(
                self.model, forget_inputs, self.model_module, no_grad=True
            )
            h_forget_ref, _ = self.forward_with_cache(
                self.ref_model, forget_inputs, self.ref_module, no_grad=True
            )
        h_forget_ref = h_forget_ref.to(h_forget.dtype)

        # Encoder direction (gradient flows through r → encoder).
        r_normed, r_scaled = self._compute_steering(
            h_forget, self.steering_coeff, forget_inputs["attention_mask"]
        )
        target = self._build_steering_target(h_forget_ref, r_scaled)

        # Phase-2 simulated forget MSE — replaces grad_conflict orth.
        # Gradient flows back through r_scaled (encoder).
        forget_loss = self.compute_activation_loss(
            h_forget.float(), target.float(), mask
        )

        # IDK anchor: r_normed toward dontknow direction.
        dontknow_dirs = self._get_dontknow_activations(forget_inputs)
        anchor_loss = 1 - (r_normed * dontknow_dirs).sum(dim=-1).mean()

        loss = (
            self.anchor_weight * anchor_loss
            + self.simulated_forget_weight * forget_loss
        )

        with torch.no_grad():
            displacement = (target - h_forget).norm(dim=-1).mean().item()
            r_norm_unnormed = r_normed.norm(dim=-1).mean().item()

        self.log({
            "phase1/anchor_loss": anchor_loss.item(),
            "phase1/cos_r_dontknow": 1.0 - anchor_loss.item(),
            "phase1/forget_loss_sim": forget_loss.item(),
            "phase1/displacement_h_to_target": displacement,
            "phase1/r_norm": r_norm_unnormed,
        })
        self._phase1_log_count = getattr(self, "_phase1_log_count", 0) + 1
        if self._phase1_log_count % 5 == 0:
            try:
                _refresh_phase1_plot(self.args.output_dir, self.state.log_history)
            except Exception:
                pass
        return loss

    # ---- Phase 2 ----------------------------------------------------- #

    def _phase2_loss(self, model, forget_inputs, retain_inputs, mask):
        h_forget, forget_outputs = self.forward_with_cache(
            model, forget_inputs, self.model_module, no_grad=False
        )
        with torch.no_grad():
            h_forget_ref, _ = self.forward_with_cache(
                self.ref_model, forget_inputs, self.ref_module, no_grad=True
            )
            # Same target construction as Phase 1
            _, r_scaled = self._compute_steering(
                h_forget_ref, self.steering_coeff, forget_inputs["attention_mask"]
            )
            target = self._build_steering_target(h_forget_ref, r_scaled)

        target = target.expand_as(h_forget)
        forget_loss = self.compute_activation_loss(h_forget, target, mask)
        retain_loss = self.compute_retain_loss(model, retain_inputs)

        if self.forget_warmup_steps > 0:
            warmup_coeff = min(1.0, self._phase2_step / self.forget_warmup_steps)
        else:
            warmup_coeff = 1.0
        self._phase2_step += 1

        loss = self.gamma * warmup_coeff * forget_loss + self.alpha * retain_loss

        self.log({
            "train/forget_loss": forget_loss.item(),
            "train/retain_loss": retain_loss.item(),
            "train/forget_warmup_coeff": warmup_coeff,
            "train/steering_coeff": self._ramped_steering_coeff(),
            "train/displacement": (target[:, :1, :] - h_forget_ref).norm(dim=-1).mean().item(),
        })
        return loss, forget_outputs

    # ------------------------------------------------------------------ #

    def compute_retain_loss(self, model, retain_inputs):
        if self.retain_loss_type == "EMBED_DIFF":
            model_retain_act, _ = self.forward_with_cache(
                model, retain_inputs, self.model_module, no_grad=False
            )
            with torch.no_grad():
                ref_retain_act, _ = self.forward_with_cache(
                    self.ref_model, retain_inputs, self.ref_module, no_grad=True
                )
            mask = retain_inputs["labels"] != -100
            return self.compute_activation_loss(
                model_retain_act, ref_retain_act.to(model_retain_act.device), mask,
            )
        return super().compute_retain_loss(model, retain_inputs)
