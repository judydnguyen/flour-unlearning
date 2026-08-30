import logging
import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import deepspeed
from trainer.unlearn.grad_diff import GradDiff

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Distributed helpers                                               #
# ------------------------------------------------------------------ #

def _is_dist():
    return dist.is_available() and dist.is_initialized()


def _world_size():
    return dist.get_world_size() if _is_dist() else 1


def _rank():
    return dist.get_rank() if _is_dist() else 0


def _all_reduce_mean_(tensor: torch.Tensor) -> torch.Tensor:
    """In-place all-reduce + average. For non-grad tensors only."""
    if not _is_dist():
        return tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor.div_(_world_size())
    return tensor


# ------------------------------------------------------------------ #
#  Encoder                                                            #
# ------------------------------------------------------------------ #

class PerSampleEncoder(nn.Module):
    """Residual MLP encoder that produces a steering direction.

    forward: r = alpha * h_pooled + (1 - alpha) * delta(h_pooled)
    """
    def __init__(self, hidden_size: int, latent_dim: int = 256,
                 num_layers: int = 2, alpha: float = 0.2):
        super().__init__()
        assert num_layers >= 2, "num_layers must be at least 2"
        self.alpha = alpha
        layers = [nn.Linear(hidden_size, latent_dim), nn.GELU()]
        for _ in range(num_layers - 2):
            layers += [nn.Linear(latent_dim, latent_dim), nn.GELU()]
        layers.append(nn.Linear(latent_dim, hidden_size))
        self.net = nn.Sequential(*layers)

        nn.init.normal_(self.net[-1].weight, std=0.02)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h_pooled: torch.Tensor) -> torch.Tensor:
        delta = self.net(h_pooled)
        return self.alpha * h_pooled + (1 - self.alpha) * delta


# ------------------------------------------------------------------ #
#  Trainer                                                            #
# ------------------------------------------------------------------ #

class FLOURParallel(GradDiff):
    """FLOUR adapted for distributed (DDP) training.

    Implements the same algorithm as the single-GPU ``FLOUR``
    (trainer.unlearn.flour): Phase 1 trains only the encoder with
    ``anchor_weight * anchor_loss + simulated_forget_weight * forget_MSE`` (the
    older grad-conflict / cosine objective has been removed in both files);
    Phase 2 freezes the encoder and unlearns the LLM toward ``h_ref + coeff*r``.

    DDP-specific handling:
      - Encoder is wrapped in DDP so its gradients are all-reduced across
        ranks during Phase 1's loss.backward().
      - Module-pattern matching unwraps DDP-wrapped models correctly.
      - Phase 1 / Phase 2 logs are reduced across ranks for clean monitoring.
      - Optional token-level ``forget_nll`` term in Phase 2 (DDP-only feature,
        off by default).
    """

    def __init__(
        self,
        module_regex=r"model\.layers\.7",
        trainable_params_regex=(r"model\.layers\.(5|6|7)\.mlp\.down_proj\.weight",),
        steering_coeff=20,
        latent_dim=256,
        encoder_epochs=2,
        encoder_layers=2,
        encoder_alpha=0.5,
        orth_weight=1.0,              # deprecated: grad_conflict removed; kept for config compat
        anchor_weight=1.0,
        simulated_forget_weight=1.0,  # weight on the Phase-2-simulated forget MSE in Phase 1
        encoder_warmup_steps=0,       # pre-Phase-1 steps aligning r with dontknow
        encoder_lr=1e-3,
        forget_warmup_steps=0,
        coeff_warmup_steps=0,
        forget_nll_weight=0.0,
        forget_nll_warmup_steps=0,
        *args, **kwargs,
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
        self.simulated_forget_weight = simulated_forget_weight
        self.encoder_warmup_steps = encoder_warmup_steps
        self.encoder_lr = encoder_lr
        self.forget_warmup_steps = forget_warmup_steps
        self.coeff_warmup_steps = coeff_warmup_steps
        self.forget_nll_weight = forget_nll_weight
        self.forget_nll_warmup_steps = forget_nll_warmup_steps

        self._phase = 1
        self._phase2_step = 0

        hidden_size = self.model.config.hidden_size
        device = next(self.model.parameters()).device

        # Raw encoder lives here. DDP wrapper is created lazily once dist
        # is initialized (HF Trainer initializes dist inside train()).
        self._encoder_raw = PerSampleEncoder(
            hidden_size, latent_dim, encoder_layers, encoder_alpha
        )
        self._encoder_raw.to(device)
        self.encoder = self._encoder_raw  # replaced by DDP wrapper after init
        self._encoder_ddp_wrapped = False

    # ------------------------------------------------------------------ #
    #  DDP setup                                                          #
    # ------------------------------------------------------------------ #

    def _maybe_wrap_encoder(self):
        """Idempotent DDP wrapping. Call before encoder is used in Phase 1.

        Skipped when no encoder param requires grad (Phase 2, or encoder_epochs=0):
        DDP rejects modules with zero trainable params, and the encoder is only
        used under torch.no_grad() in Phase 2 anyway.
        """
        if self._encoder_ddp_wrapped or not _is_dist():
            return
        if not any(p.requires_grad for p in self._encoder_raw.parameters()):
            return

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}")
        self._encoder_raw.to(device)

        self.encoder = DDP(
            self._encoder_raw,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
        self._encoder_ddp_wrapped = True

    def _encoder_module(self) -> nn.Module:
        """Return the underlying (un-DDP) encoder for state access / freezing."""
        return self._encoder_raw

    # ------------------------------------------------------------------ #
    #  Helpers                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _unwrap(model):
        """Strip DDP / DeepSpeed wrappers to expose the underlying module."""
        if isinstance(model, deepspeed.DeepSpeedEngine):
            model = model.module
        if isinstance(model, DDP):
            model = model.module
        return model

    def _get_matching_module(self, model, module_regex):
        m = self._unwrap(model)
        matched = {n: mod for n, mod in m.named_modules() if re.fullmatch(module_regex, n)}
        if len(matched) != 1:
            raise ValueError(f"Expected 1 module match for {module_regex}, got {len(matched)}")
        return next(iter(matched.values()))

    def _freeze_all_params(self, model, requires_grad=False):
        for p in model.parameters():
            p.requires_grad = requires_grad

    def _set_trainable_params(self, model, trainable_params_regex, requires_grad=True):
        m = self._unwrap(model)
        for name, param in m.named_parameters():
            if any(re.fullmatch(pat, name) for pat in trainable_params_regex):
                param.requires_grad = requires_grad

    def _get_phase2_params(self):
        m = self._unwrap(self.model)
        return [
            p for n, p in m.named_parameters()
            if any(re.fullmatch(pat, n) for pat in self.trainable_params_regex)
            and p.requires_grad
        ]

    def _log_phase2_param_summary(self):
        """Log which params matched trainable_params_regex. Catches the silent
        no-op case where the regex matches zero params and the optimizer steps
        on nothing."""
        if _rank() != 0:
            return
        m = self._unwrap(self.model)
        matched = [(n, p) for n, p in m.named_parameters() if p.requires_grad]
        total_numel = sum(p.numel() for _, p in matched)
        logger.info(
            "[phase2] %d trainable params matched %s (total numel=%s)",
            len(matched), self.trainable_params_regex, f"{total_numel:,}",
        )
        for n, p in matched[:16]:
            logger.info("[phase2]   %s shape=%s", n, tuple(p.shape))
        if len(matched) > 16:
            logger.info("[phase2]   ... and %d more", len(matched) - 16)
        if not matched:
            logger.error(
                "[phase2] NO PARAMS MATCH %s. Optimizer step will be a no-op. "
                "Check the regex against model.named_parameters().",
                self.trainable_params_regex,
            )

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

    def _get_dontknow_activations(self, forget_inputs):
        input_ids = forget_inputs["input_ids"].clone()
        labels = forget_inputs["labels"]
        answer_mask = labels != -100
        input_ids[answer_mask] = self.tokenizer.unk_token_id or self.tokenizer.pad_token_id
        proxy_inputs = {
            "input_ids": input_ids,
            "attention_mask": forget_inputs["attention_mask"],
        }
        with torch.no_grad():
            h, _ = self.forward_with_cache(
                self.ref_model, proxy_inputs, self.ref_module, no_grad=True
            )
        return F.normalize(h.float().mean(dim=1), dim=-1)

    def _warmup_encoder(self, num_steps):
        """Pre-align the encoder direction r with the dontknow direction.

        Runs `num_steps` of gradient descent on (1 - cos(r, dontknow)) using
        forget batches, before Phase 1. Only the (raw) encoder is updated; the
        model is forwarded under no_grad. No collectives are used, so ranks run
        independently; the DDP wrap created later in create_optimizer broadcasts
        rank-0's weights, so all ranks enter Phase 1 from a consistent encoder.
        """
        from torch.utils.data import DataLoader
        if self.train_dataset is None or num_steps <= 0:
            return
        if _rank() == 0:
            logger.info("[encoder-warmup] aligning r with dontknow for %d steps", num_steps)
        self._encoder_raw.train()
        opt = torch.optim.AdamW(self._encoder_raw.parameters(), lr=self.encoder_lr)
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            collate_fn=self.data_collator,
            shuffle=True,
        )
        device = next(self._encoder_raw.parameters()).device
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
            pooled = h_forget.float().mean(dim=1)
            r = self._encoder_raw(pooled)
            r_normed = r / (r.norm(dim=-1, keepdim=True) + 1e-8)
            dontknow = self._get_dontknow_activations(forget_inputs)
            anchor_loss = 1 - (r_normed * dontknow).sum(dim=-1).mean()
            opt.zero_grad()
            anchor_loss.backward()
            opt.step()
            last_loss = anchor_loss.item()
            if _rank() == 0 and step % 10 == 0:
                logger.info(
                    "[encoder-warmup] step %d/%d: 1 - cos(r, dontknow) = %.4f",
                    step, num_steps, last_loss,
                )
            step += 1
        if _rank() == 0 and last_loss is not None:
            logger.info("[encoder-warmup] done. final 1 - cos(r, dontknow) = %.4f", last_loss)

    # ------------------------------------------------------------------ #
    #  Two-phase orchestration                                            #
    # ------------------------------------------------------------------ #

    def evaluate(self, *args, **kwargs):
        if self._phase == 1:
            return {}
        return super().evaluate(*args, **kwargs)

    def train(self, resume_from_checkpoint=None, **kwargs):
        original_epochs = self.args.num_train_epochs

        # Phase 2 trains a small subset of params under gradient_checkpointing.
        # With most params frozen the embedding output has requires_grad=False,
        # so checkpointed layers see no grad-bearing inputs and the loss loses
        # its grad_fn. HF only auto-installs this hook when PEFT is loaded
        # (modeling_utils.py:2421); install it ourselves, and force
        # use_reentrant=False so partial-grad checkpointing returns real grads.
        if getattr(self.args, "gradient_checkpointing", False):
            self._unwrap(self.model).enable_input_require_grads()
            gc_kwargs = self.args.gradient_checkpointing_kwargs or {}
            if gc_kwargs.get("use_reentrant", True):
                gc_kwargs = {**gc_kwargs, "use_reentrant": False}
                self.args.gradient_checkpointing_kwargs = gc_kwargs

        # Phase 1 — encoder only. The simulated-forget loss reaches the encoder
        # through `target`, so the model is forwarded under no_grad and receives
        # no gradient. We still mark the Phase-2 param subset requires_grad=True
        # (but leave it OUT of the optimizer): plain-DDP / DeepSpeed model
        # wrapping refuses a module with zero trainable params, so this keeps the
        # wrapper legal while the optimizer steps the encoder alone — the model
        # is never updated. Skipped when encoder_epochs <= 0 so Phase 2 can be
        # tested in isolation against the encoder's init.
        if self.encoder_epochs > 0:
            self._phase = 1
            self._freeze_all_params(self.model, requires_grad=False)
            self._set_trainable_params(self.model, self.trainable_params_regex, True)
            if self.encoder_warmup_steps > 0:
                self._warmup_encoder(self.encoder_warmup_steps)
            self.args.num_train_epochs = self.encoder_epochs
            super().train(resume_from_checkpoint=resume_from_checkpoint, **kwargs)
            resume_from_checkpoint = None

            # Persist encoder weights for offline trajectory analysis.
            if _rank() == 0 and self.args.output_dir:
                enc_path = os.path.join(self.args.output_dir, "encoder.pt")
                torch.save(self._encoder_module().state_dict(), enc_path)
                logger.info("[phase1] saved encoder to %s", enc_path)

        # Phase 2: freeze encoder, unlearn the LLM. Phase-2 length =
        # num_train_epochs - encoder_epochs, matching single-GPU FLOUR so a
        # given num_train_epochs is the SAME total budget in both trainers.
        self._phase = 2
        self._freeze_all_params(self._encoder_module(), requires_grad=False)
        self._encoder_module().eval()
        self.optimizer = None
        self.lr_scheduler = None
        self.args.num_train_epochs = original_epochs - self.encoder_epochs
        super().train(resume_from_checkpoint=resume_from_checkpoint, **kwargs)

        self.args.num_train_epochs = original_epochs

    def create_optimizer(self):
        # Wrap encoder in DDP on first call (dist is initialized by now)
        self._maybe_wrap_encoder()

        if self._phase == 1:
            # Phase 1 optimizes ONLY the encoder. The Phase-2 param subset is
            # marked trainable (so DDP/DeepSpeed will wrap the model — it rejects
            # a module with zero trainable params) but is deliberately left out
            # of the optimizer, so the model never updates during Phase 1.
            self._freeze_all_params(self.model, requires_grad=False)
            self._set_trainable_params(self.model, self.trainable_params_regex, True)
            optimizer_cls, _ = self.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(
                self._encoder_raw.parameters(),
                lr=self.encoder_lr,
            )
        else:
            self._freeze_all_params(self._encoder_module(), requires_grad=False)
            self._freeze_all_params(self.model, requires_grad=False)
            self._set_trainable_params(self.model, self.trainable_params_regex, True)
            self._log_phase2_param_summary()
            super().create_optimizer()

    # ------------------------------------------------------------------ #
    #  Steering vector                                                    #
    # ------------------------------------------------------------------ #

    def _ramped_steering_coeff(self) -> float:
        if self.coeff_warmup_steps <= 0:
            return self.steering_coeff
        return self.steering_coeff * max(0.0, 1.0 - self._phase2_step / self.coeff_warmup_steps)

    def _compute_steering(self, h_forget: torch.Tensor, steering_coeff: float):
        """(r_normed, r_scaled_unsqueezed)."""
        pooled = h_forget.float().mean(dim=1)
        r = self.encoder(pooled)
        r_normed = r / (r.norm(dim=-1, keepdim=True) + 1e-8)
        r_scaled = r_normed.unsqueeze(1).to(h_forget.dtype) * steering_coeff
        return r_normed, r_scaled

    def _build_steering_target(self, h_forget_ref: torch.Tensor, r_scaled: torch.Tensor) -> torch.Tensor:
        return h_forget_ref + r_scaled

    # ------------------------------------------------------------------ #
    #  Loss computation                                                   #
    # ------------------------------------------------------------------ #

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # transformers 4.46+ passes num_items_in_batch into compute_loss; we
        # don't use it (loss is already mean-reduced), so swallow it via
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
        # Phase 1 trains ONLY the encoder; the model is fully frozen. Gradient
        # reaches the encoder through `target` (which depends on r), so both
        # model forwards run under no_grad and we keep no model activation graph.
        with torch.no_grad():
            h_forget, _ = self.forward_with_cache(
                self.model, forget_inputs, self.model_module, no_grad=True
            )
            h_forget_ref, _ = self.forward_with_cache(
                self.ref_model, forget_inputs, self.ref_module, no_grad=True
            )
        h_forget_ref = h_forget_ref.to(h_forget.dtype)

        # Encoder direction (gradient flows through r → encoder; the DDP-wrapped
        # encoder all-reduces these grads on loss.backward()).
        r_normed, r_scaled = self._compute_steering(h_forget, self.steering_coeff)
        target = self._build_steering_target(h_forget_ref, r_scaled)

        # Phase-2-simulated forget MSE: pull the (would-be) model activation
        # toward the steering target. Replaces the old grad-conflict term.
        forget_loss = self.compute_activation_loss(
            h_forget.float(), target.float(), mask
        )

        # IDK anchor: align r_normed with the dontknow direction.
        dontknow_dirs = self._get_dontknow_activations(forget_inputs)
        anchor_loss = 1 - (r_normed * dontknow_dirs).sum(dim=-1).mean()

        loss = (
            self.anchor_weight * anchor_loss
            + self.simulated_forget_weight * forget_loss
        )

        # Logged values: averaged across ranks for cleaner monitoring.
        with torch.no_grad():
            displacement = (target - h_forget).norm(dim=-1).mean()
            r_norm = r_normed.norm(dim=-1).mean()
            log_tensors = {
                "phase1/anchor_loss": anchor_loss.detach().clone(),
                "phase1/forget_loss_sim": forget_loss.detach().clone(),
                "phase1/displacement_h_to_target": displacement,
                "phase1/r_norm": r_norm,
            }
            for v in log_tensors.values():
                _all_reduce_mean_(v)
            log_dict = {k: v.item() for k, v in log_tensors.items()}
            log_dict["phase1/cos_r_dontknow"] = 1.0 - log_dict["phase1/anchor_loss"]

        if _rank() == 0:
            self.log(log_dict)
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
            # Build the target from the REFERENCE activations (stable across
            # Phase 2), matching single-GPU FLOUR. _ramped_steering_coeff()
            # equals steering_coeff unless coeff_warmup_steps > 0.
            _, r_scaled = self._compute_steering(h_forget_ref, self.steering_coeff)
            target = self._build_steering_target(h_forget_ref, r_scaled)

        target = target.expand_as(h_forget)
        forget_loss = self.compute_activation_loss(h_forget, target, mask)
        retain_loss = self.compute_retain_loss(model, retain_inputs)

        if self.forget_warmup_steps > 0:
            warmup_coeff = min(1.0, self._phase2_step / self.forget_warmup_steps)
        else:
            warmup_coeff = 1.0
        if self.forget_nll_weight > 0.0 and self.forget_nll_warmup_steps > 0:
            nll_warmup = min(1.0, self._phase2_step / self.forget_nll_warmup_steps)
        elif self.forget_nll_weight > 0.0:
            nll_warmup = 1.0
        else:
            nll_warmup = 0.0
        self._phase2_step += 1

        loss = self.gamma * warmup_coeff * forget_loss + self.alpha * retain_loss

        # Token-level forget pressure: subtract β * forget_nll so the LM head
        # is pushed away from the memorized continuation. Activation steering
        # alone leaves token likelihoods (and MIA-based privleak) close to
        # the target model — this term targets that residual.
        forget_nll_val = 0.0
        if self.forget_nll_weight > 0.0 and forget_outputs.loss is not None:
            forget_nll = forget_outputs.loss
            loss = loss - self.forget_nll_weight * nll_warmup * forget_nll
            forget_nll_val = forget_nll.detach()

        with torch.no_grad():
            log_tensors = {
                "train/forget_loss": forget_loss.detach().clone(),
                "train/retain_loss": retain_loss.detach().clone(),
                "train/displacement": r_scaled.norm(dim=-1).mean(),
            }
            if self.forget_nll_weight > 0.0:
                log_tensors["train/forget_nll"] = (
                    forget_nll_val.clone() if torch.is_tensor(forget_nll_val)
                    else torch.tensor(0.0, device=h_forget.device)
                )
            for v in log_tensors.values():
                _all_reduce_mean_(v)
            log_dict = {k: v.item() for k, v in log_tensors.items()}
            log_dict["train/forget_warmup_coeff"] = warmup_coeff
            log_dict["train/forget_nll_warmup_coeff"] = nll_warmup
            log_dict["train/steering_coeff"] = self._ramped_steering_coeff()

        if _rank() == 0:
            self.log(log_dict)
        return loss, forget_outputs

    # ------------------------------------------------------------------ #
    #  Phase 2 grad-norm diagnostic                                       #
    # ------------------------------------------------------------------ #

    def training_step(self, *args, **kwargs):
        """Wrap super().training_step to log phase-2 grad norm.

        Catches the silent no-op case: if forget/retain losses look fine but
        grad norm on trainable params is ~0, the optimizer is stepping on
        gradients that don't carry signal.
        """
        loss = super().training_step(*args, **kwargs)

        if self._phase != 2:
            return loss

        # Log every `logging_steps` (default 10). _phase2_step was incremented
        # in _phase2_loss, so it reflects the current micro-batch index.
        log_every = max(1, int(getattr(self.args, "logging_steps", 10)))
        if self._phase2_step % log_every != 0:
            return loss

        with torch.no_grad():
            params = self._get_phase2_params()
            sq_sum = torch.zeros((), device=loss.device, dtype=torch.float32)
            n_with_grad = 0
            for p in params:
                if p.grad is not None:
                    sq_sum += p.grad.detach().float().pow(2).sum()
                    n_with_grad += 1
            total_norm = sq_sum.sqrt()
            _all_reduce_mean_(total_norm)
            if _rank() == 0:
                self.log({
                    "train/phase2_param_grad_norm": total_norm.item(),
                    "train/phase2_n_params_with_grad": n_with_grad,
                    "train/phase2_n_params_total": len(params),
                })
        return loss

    # ------------------------------------------------------------------ #
    #  Retain loss                                                        #
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