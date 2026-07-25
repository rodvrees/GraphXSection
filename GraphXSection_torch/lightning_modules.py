"""
PyTorch Lightning modules for GraphXSection_torch.

Supported Loss Functions:
- "mae" or "l1": Mean Absolute Error (L1 Loss) - DEFAULT
- "mse" or "l2": Mean Squared Error (L2 Loss)
- "huber" or "smooth_l1": Huber Loss (Smooth L1 Loss)
- "rmse": Root Mean Squared Error

Supported Optimizers:
- "adamw": AdamW optimizer - DEFAULT (with weight_decay=0.01)
- "adam": Adam optimizer
- "sgd": SGD optimizer (with momentum=0.9)
- "rmsprop": RMSprop optimizer

Configuration:
- Set "loss_function" in config to specify loss type
- Set "huber_delta" in config for Huber loss delta parameter (default: 1.0)
- Set "optimizer" in config to specify optimizer type
- Set "weight_decay" in config for weight decay (default: 0.01 for AdamW, 0.0 for others)
- Set "momentum" in config for SGD momentum (default: 0.9)

Example config:
{
    "loss_function": "mae",
    "huber_delta": 1.0,
    "optimizer": "adamw",
    "weight_decay": 0.01,
    "momentum": 0.9,
    ...
}
"""

import logging
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, Any, Optional
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scipy.optimize import linear_sum_assignment
from GraphXSection_torch.models.Pyggy import Pyggy
from GraphXSection_torch.models.AttentiveFP import AttentiveFPWrapper
from GraphXSection_torch.models.Sigma import SigmaModel
from GraphXSection_torch.models.Ginny import GinnyModel

log = logging.getLogger(__name__)


class GraphXSectionLightningModule(pl.LightningModule):
    """PyTorch Lightning module for GraphXSection models."""

    def __init__(
        self,
        config: Dict[str, Any],
        model_type: str = "Pyggy",
        node_feature_sizes: Optional[list[int]] = None,
        edge_feature_sizes: Optional[list[int]] = None,
        global_feature_sizes: Optional[list[int]] = None,
    ):
        """
        Initialize the Lightning module.

        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary
        model_type : str
            Type of model to use ('Ginny', 'Gato', 'QSAR_linear', 'Eddy', 'Pyggy')
        input_node_dim : Optional[int]
            Node feature dimension (required for Pyggy)
        edge_dim : Optional[int]
            Edge feature dimension (required for Pyggy)
        expected_global_dim : Optional[int]
            Global feature dimension to concatenate/process (Pyggy)
        """
        super().__init__()
        # ensure config is a plain dict
        self.config = config if isinstance(config, dict) else dict(config)
        model_cfg = dict(self.config.get("model", {}))

        # Merge top-level model-like keys if present (backwards compatibility)
        for k in [
            "learning_rate",
            "loss_function",
            "gnn_units",
            "dropout",
            "num_GNN_layers",
        ]:
            if k in self.config and k not in model_cfg:
                model_cfg[k] = self.config[k]

        # If loss_function still missing, attempt to read it from wandb config (handles flattened keys)
        try:
            import wandb

            wb_cfg = dict(getattr(wandb, "config", {}) or {})
            applied_from = None
            if model_cfg.get("loss_function") is None:
                # 1) flattened key like 'model.loss_function'
                if (
                    "model.loss_function" in wb_cfg
                    and wb_cfg["model.loss_function"] is not None
                ):
                    model_cfg["loss_function"] = wb_cfg["model.loss_function"]
                    applied_from = "wandb:model.loss_function"
                # 2) top-level flattened 'loss_function'
                elif "loss_function" in wb_cfg and wb_cfg["loss_function"] is not None:
                    model_cfg["loss_function"] = wb_cfg["loss_function"]
                    applied_from = "wandb:loss_function"
                # 3) nested dict under wandb.config['model']
                elif (
                    isinstance(wb_cfg.get("model"), dict)
                    and wb_cfg["model"].get("loss_function") is not None
                ):
                    model_cfg["loss_function"] = wb_cfg["model"]["loss_function"]
                    applied_from = "wandb:config.model.loss_function"

            if applied_from:
                log.info(
                    "Applied loss_function from %s -> %s",
                    applied_from,
                    model_cfg.get("loss_function"),
                )
        except Exception:
            # No wandb available or nothing to merge; silently continue
            pass

        # Save hyperparameters for visibility in logs/checkpoints
        # select a stable set to save
        hparams_to_save = {
            "learning_rate": model_cfg.get("learning_rate"),
            "gnn_units": model_cfg.get("gnn_units"),
            "num_GNN_layers": model_cfg.get("num_GNN_layers"),
            "dropout": model_cfg.get("dropout"),
            "readout": model_cfg.get("readout"),
            "loss_function": model_cfg.get("loss_function"),
        }
        # persist hyperparameters using Lightning API (do NOT assign self.hparams directly)
        try:
            # save_hyperparameters will populate self.hparams in a Lightning-safe way
            # Note: do not call save_hyperparameters twice; call once with the selected hparams
            self.save_hyperparameters(hparams_to_save)
            saved_hparams_preview = self.hparams
        except Exception:
            # fallback: attach to a private attribute for debugging if save_hyperparameters not available
            self._saved_hparams = hparams_to_save
            saved_hparams_preview = self._saved_hparams

        log.info(
            "Initializing LightningModule with hyperparameters: %s",
            saved_hparams_preview,
        )

        # === Loss function selection (robust mapping) ===
        # Try model.loss_function first, then top-level loss_function
        loss_key = (
            model_cfg.get("loss_function")
            if model_cfg.get("loss_function") is not None
            else self.config.get("loss_function")
        )
        if isinstance(loss_key, str):
            loss_key = loss_key.lower()
        if loss_key in ("mse", "mean_squared_error", "squared", "mse_loss"):
            self.loss_fn = torch.nn.MSELoss()
            self.loss_name = "mse"
        elif loss_key in ("huber", "smooth_l1", "smooth_l1_loss"):
            huber_delta = float(
                model_cfg.get("huber_delta", self.config.get("huber_delta", 1.0))
            )
            try:
                self.loss_fn = torch.nn.SmoothL1Loss(beta=huber_delta)
            except TypeError:
                self.loss_fn = torch.nn.SmoothL1Loss()
            self.loss_name = "huber"
        else:
            # default to MAE
            self.loss_fn = torch.nn.L1Loss()
            self.loss_name = "mae"

        log.info("Configured loss function: %s", self.loss_name)

        # if wandb is present update config so UI shows actual used hyperparams
        try:
            import wandb

            if getattr(wandb, "run", None):
                # update wandb config using the saved dict (do not rely on self.hparams setter)
                wandb.run.config.update(
                    {
                        "model." + k: v
                        for k, v in hparams_to_save.items()
                        if v is not None
                    },
                    allow_val_change=True,
                )
        except Exception:
            pass

        self.model_type = model_type
        self.save_hyperparameters()

        if model_type == "Pyggy":
            self.model = Pyggy(
                config,
                node_feature_sizes=node_feature_sizes,
                edge_feature_sizes=edge_feature_sizes,
                global_feature_sizes=global_feature_sizes,
            )
        elif model_type == "AttentiveFP":
            self.model = AttentiveFPWrapper(
                config,
                node_feature_sizes=node_feature_sizes,
                edge_feature_sizes=edge_feature_sizes,
                global_feature_sizes=global_feature_sizes,
            )
        elif model_type == "Sigma":
            self.model = SigmaModel(
                config,
                node_feature_sizes=node_feature_sizes,
                edge_feature_sizes=edge_feature_sizes,
                global_feature_sizes=global_feature_sizes,
            )
        elif model_type == "Ginny":
            self.model = GinnyModel(
                config,
                node_feature_sizes=node_feature_sizes,
                edge_feature_sizes=edge_feature_sizes,
                global_feature_sizes=global_feature_sizes,
            )

        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Metrics tracking
        self.train_mae_list = []
        self.val_mae_list = []
        self.val_losses = []

        # Best validation MAE tracking (like original GraphXSection)
        self.best_val_mae = float("inf")

        # Debug configuration
        dbg = (
            config.get("debug", {}) if isinstance(config.get("debug", {}), dict) else {}
        )
        self.debug_enabled = bool(dbg.get("enable", False))
        self.debug_max_show = int(dbg.get("max_show", 3))
        self.debug_every_n_steps = int(dbg.get("every_n_steps", 1))
        self._debug_seen = {"train": 0, "val": 0, "test": 0}

    def forward(self, batch):
        """Forward pass."""

        return self.model(batch)

    def training_step(self, batch, batch_idx):
        """Training step."""
        # Rich debug of feature flow at throttled steps: register hooks before forward

        outputs = self(batch)

        # After forward, remove hooks and log summaries
        targets = batch["target"]

        # compute configured loss
        loss = self.loss_fn(outputs, targets)

        # log main loss under a deterministic key
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(targets),
        )

        # also log explicit metrics for verification
        try:
            mae = F.l1_loss(outputs, targets)
            mse = F.mse_loss(outputs, targets)
            self.log(
                "train_mae",
                mae,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=len(targets),
            )
            self.log(
                "train_mse",
                mse,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=len(targets),
            )
        except Exception:
            pass
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        outputs = self(batch)
        targets = batch["target"]

        val_loss = self.loss_fn(outputs, targets)
        self.log(
            "val_loss",
            val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(targets),
        )

        # also log explicit metrics for verification
        try:
            val_mae = F.l1_loss(outputs, targets)
            val_mse = F.mse_loss(outputs, targets)
            self.log(
                "val_mae",
                val_mae,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=len(targets),
            )
            self.log(
                "val_mse",
                val_mse,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=len(targets),
            )
        except Exception:
            pass

        return {"val_loss": val_loss}

    def test_step(self, batch, batch_idx):
        """Test step."""
        outputs = self(batch)
        targets = batch["target"]

        # Calculate loss
        if self.loss_name == "rmse":
            loss = torch.sqrt(self.loss_fn(outputs, targets))
        else:
            loss = self.loss_fn(outputs, targets)

        # Calculate MAE and relative errors (always for comparison)
        mae = torch.mean(torch.abs(outputs - targets))
        relative_errors = torch.abs(outputs - targets) / targets
        rmae = torch.mean(relative_errors) * 100
        rmedae = torch.median(relative_errors) * 100

        # Log metrics
        self.log(
            "test_loss", loss, on_step=False, on_epoch=True, batch_size=len(targets)
        )
        self.log("test_mae", mae, on_step=False, on_epoch=True, batch_size=len(targets))
        self.log(
            "test_rmae", rmae, on_step=False, on_epoch=True, batch_size=len(targets)
        )
        self.log(
            "test_rmedae", rmedae, on_step=False, on_epoch=True, batch_size=len(targets)
        )

        return {
            "test_loss": loss,
            "test_mae": mae,
            "test_rmae": rmae,
            "test_rmedae": rmedae,
        }

    def on_validation_epoch_end(self):
        """Called at the end of validation epoch."""
        # Get current validation metrics
        val_loss = self.trainer.callback_metrics.get("val_loss")
        val_mae = self.trainer.callback_metrics.get("val_mae")
        val_mse = self.trainer.callback_metrics.get("val_mse")

        if val_loss is not None:
            self.val_losses.append(val_loss.item())
        if val_mae is not None:
            current_mae = val_mae.item()
            self.val_mae_list.append(current_mae)

            # Track best validation MAE (like original GraphXSection)
            if current_mae < self.best_val_mae:
                self.best_val_mae = current_mae

            # Log best validation MAE to wandb (like WandbBestValMAELogger in original)
            if self.logger and hasattr(self.logger, "experiment"):
                if hasattr(self.logger.experiment, "log"):
                    self.logger.experiment.log({"best_val_mae": self.best_val_mae})

        # Runtime verification: if configured to use MSE, ensure val_loss ~= val_mse
        try:
            if getattr(self, "loss_name", None) == "mse":
                if val_loss is not None and val_mse is not None:
                    vloss = float(val_loss.item())
                    vmse = float(val_mse.item())
                    # Tolerance: 1e-6 relative or absolute small value
                    tol = max(1e-6, 1e-6 * abs(vmse))
                    if abs(vloss - vmse) > tol:
                        log.warning(
                            "Configured loss is MSE but val_loss (%.6g) differs from logged val_mse (%.6g). Check loss computation path.",
                            vloss,
                            vmse,
                        )
                    else:
                        log.info(
                            "Validated: configured MSE matches computed val_mse (%.6g).",
                            vmse,
                        )
        except Exception:
            # keep robust if metrics missing or shapes unexpected
            pass

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        # Get optimizer configuration
        optimizer_name = self.config["model"].get("optimizer", "adamw").lower()
        learning_rate = self.config["model"].get("learning_rate", 0.001)
        weight_decay = self.config["model"].get(
            "weight_decay", 0.01
        )  # Default weight decay for AdamW

        # Create optimizer based on configuration
        if optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
            )
            log.info(
                "Created optimizer AdamW: lr=%s, weight_decay=%s",
                learning_rate,
                weight_decay,
            )
        elif optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
            )
            log.info(
                "Created optimizer Adam: lr=%s, weight_decay=%s",
                learning_rate,
                weight_decay,
            )
        elif optimizer_name == "sgd":
            momentum = self.config["model"].get("momentum", 0.9)
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                momentum=momentum,
            )
            log.info(
                "Created optimizer SGD: lr=%s, weight_decay=%s, momentum=%s",
                learning_rate,
                weight_decay,
                momentum,
            )
        elif optimizer_name == "rmsprop":
            optimizer = torch.optim.RMSprop(
                self.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
            )
            log.info(
                "Created optimizer RMSprop: lr=%s, weight_decay=%s",
                learning_rate,
                weight_decay,
            )
        else:
            raise ValueError(
                f"Unknown optimizer: {optimizer_name}. Supported: 'adamw', 'adam', 'sgd', 'rmsprop'"
            )

        # Learning rate scheduler
        # Check if scheduler is configured (either nested dict or presence of lr_scheduler.* keys)
        lr_scheduler_config = self.config.get("lr_scheduler", {})
        has_scheduler = (
            bool(lr_scheduler_config)
            and isinstance(lr_scheduler_config, dict)
            and "factor" in lr_scheduler_config
        )

        if has_scheduler:
            scheduler_factor = lr_scheduler_config.get("factor", 0.1)
            scheduler_patience = lr_scheduler_config.get("patience", 10)
            scheduler_min_lr = lr_scheduler_config.get("min_lr", 1e-6)

            log.info(
                "Configuring ReduceLROnPlateau scheduler: factor=%s, patience=%s, min_lr=%s",
                scheduler_factor,
                scheduler_patience,
                scheduler_min_lr,
            )

            scheduler = ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=scheduler_factor,
                patience=scheduler_patience,
                min_lr=scheduler_min_lr,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "frequency": 1,
                },
            }
        else:
            # No scheduler configured - log optimizer summary
            log.info(
                "No learning rate scheduler configured. Using optimizer '%s' with fixed lr=%s, weight_decay=%s",
                optimizer_name,
                learning_rate,
                weight_decay,
            )
            if optimizer_name == "sgd":
                log.info("SGD momentum=%s", self.config["model"].get("momentum", 0.9))
            return optimizer


# ---------------------------------------------------------------------------
# Anchored set loss (protomer-aware training objective)
# ---------------------------------------------------------------------------


def anchored_set_loss(
    preds: torch.Tensor,
    mol_index: torch.Tensor,
    dominant_index: torch.Tensor,
    peaks: list[torch.Tensor],
    loss_fn,
    gap_lambda: float = 1.0,
):
    """Anchored set loss over a `collate_protomer` batch.

    Two terms, computed per molecule and pooled over the batch:

    - Anchor term: the dominant candidate's predicted CCS is regressed to the
      base (most intense) observed peak. Applies to every molecule, so
      single-peak molecules reduce to plain base-CCS regression.
    - Gap term (molecules with >=2 observed peaks only): the remaining observed
      peaks are matched permutation-invariantly to the remaining candidates by a
      Hungarian assignment on the absolute gap error |pred_gap - obs_gap|, and
      the loss regresses each matched predicted gap (pred_cand - pred_dominant)
      to the observed gap (obs_peak - obs_base). This teaches the gap magnitude
      without asserting a specific (unreliable) site-to-peak identity.

    Parameters
    ----------
    preds : Tensor [total_candidates] or [total_candidates, 1]
        One predicted CCS per forwarded candidate graph (flat over the batch).
    mol_index : LongTensor [total_candidates]
        Segment id mapping each candidate graph to its molecule.
    dominant_index : LongTensor [num_molecules]
        Global index (into `preds`) of each molecule's dominant candidate.
    peaks : list[Tensor]
        Ragged observed peak set per molecule, base peak first.
    loss_fn : callable
        Elementwise-reducible regression loss (e.g. SmoothL1Loss / L1Loss).
    gap_lambda : float
        Weight on the gap term relative to the anchor term.

    Returns
    -------
    dict with keys: loss, anchor_loss, anchor_mae, gap_loss, gap_mae, n_gap_pairs.
    """
    preds = preds.view(-1)
    device = preds.device
    num_molecules = len(peaks)

    # --- anchor term (vectorised over all molecules) ---
    anchor_pred = preds[dominant_index]
    anchor_obs = torch.stack([p[0] for p in peaks]).to(device)
    anchor_loss = loss_fn(anchor_pred, anchor_obs)
    # base-CCS MAE in absolute units (interpretable "how good is CCS")
    anchor_mae = (anchor_pred - anchor_obs).abs().mean().detach()

    # --- gap term (loop over multi-peak molecules only) ---
    pred_gap_terms: list[torch.Tensor] = []
    obs_gap_terms: list[torch.Tensor] = []
    for m in range(num_molecules):
        obs = peaks[m].to(device)
        if obs.numel() < 2:
            continue
        cand_idx = (mol_index == m).nonzero(as_tuple=False).view(-1)
        dom = int(dominant_index[m].item())
        rem_cand = cand_idx[cand_idx != dom]
        if rem_cand.numel() == 0:
            continue

        pred_dom = preds[dom]
        pred_gaps = preds[rem_cand] - pred_dom            # [n_rem_cand]
        obs_gaps = obs[1:] - obs[0]                        # [n_rem_obs]

        # Hungarian match on the detached absolute gap error.
        cost = (obs_gaps[:, None] - pred_gaps[None, :]).abs().detach().cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost)
        pred_gap_terms.append(pred_gaps[col_ind])
        obs_gap_terms.append(obs_gaps[row_ind])

    if pred_gap_terms:
        pred_gap_cat = torch.cat(pred_gap_terms)
        obs_gap_cat = torch.cat(obs_gap_terms)
        gap_loss = loss_fn(pred_gap_cat, obs_gap_cat)
        gap_mae = (pred_gap_cat - obs_gap_cat).abs().mean().detach()
        n_gap_pairs = pred_gap_cat.numel()
    else:
        gap_loss = torch.zeros((), device=device)
        gap_mae = torch.zeros((), device=device)
        n_gap_pairs = 0

    total = anchor_loss + gap_lambda * gap_loss
    return {
        "loss": total,
        "anchor_loss": anchor_loss.detach(),
        "anchor_mae": anchor_mae,
        "gap_loss": gap_loss.detach(),
        "gap_mae": gap_mae,
        "n_gap_pairs": n_gap_pairs,
    }


class ProtomerLightningModule(GraphXSectionLightningModule):
    """Lightning module for protomer-aware training with the anchored set loss.

    Batches follow `collate_protomer` (keys: graph, global_feats, mol_index,
    dominant_index, peaks, num_molecules). The backbone model is unchanged: it
    produces one scalar CCS per candidate graph. Only the loss and the
    batch grouping differ from the base module, so the matched baseline keeps
    using `GraphXSectionLightningModule` with plain regression.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        model_cfg = dict(self.config.get("model", {}))
        self.gap_lambda = float(
            model_cfg.get("gap_lambda", self.config.get("gap_lambda", 1.0))
        )
        log.info("ProtomerLightningModule gap_lambda=%s", self.gap_lambda)
        self.best_val_base_mae = float("inf")
        self.best_val_gap_mae = float("inf")

    def _shared_step(self, batch, stage: str):
        preds = self.model(batch)
        out = anchored_set_loss(
            preds,
            batch["mol_index"],
            batch["dominant_index"],
            batch["peaks"],
            self.loss_fn,
            gap_lambda=self.gap_lambda,
        )
        bs = batch["num_molecules"]
        # base_mae (absolute CCS error on the base peak) and gap_mae are the two
        # interpretable metrics; show both live in the progress bar.
        self.log(f"{stage}_loss", out["loss"], on_step=False, on_epoch=True,
                 prog_bar=True, batch_size=bs)
        self.log(f"{stage}_base_mae", out["anchor_mae"], on_step=False,
                 on_epoch=True, prog_bar=True, batch_size=bs)
        self.log(f"{stage}_gap_mae", out["gap_mae"], on_step=False,
                 on_epoch=True, prog_bar=True, batch_size=bs)
        self.log(f"{stage}_anchor_loss", out["anchor_loss"], on_step=False,
                 on_epoch=True, prog_bar=False, batch_size=bs)
        self.log(f"{stage}_gap_loss", out["gap_loss"], on_step=False,
                 on_epoch=True, prog_bar=False, batch_size=bs)
        return out

    def training_step(self, batch, batch_idx):
        return {"loss": self._shared_step(batch, "train")["loss"]}

    def validation_step(self, batch, batch_idx):
        return {"val_loss": self._shared_step(batch, "val")["loss"]}

    def test_step(self, batch, batch_idx):
        out = self._shared_step(batch, "test")
        return {"test_loss": out["loss"]}

    def on_validation_epoch_end(self):
        """Print a clean per-epoch summary to the terminal and track best MAEs."""
        cm = self.trainer.callback_metrics

        def _get(k):
            v = cm.get(k)
            return float(v.item()) if v is not None else None

        tr_base, tr_gap = _get("train_base_mae"), _get("train_gap_mae")
        va_loss = _get("val_loss")
        va_base, va_gap = _get("val_base_mae"), _get("val_gap_mae")

        if va_base is not None:
            self.best_val_base_mae = min(self.best_val_base_mae, va_base)
        if va_gap is not None:
            self.best_val_gap_mae = min(self.best_val_gap_mae, va_gap)

        def _f(x):
            return f"{x:.3f}" if x is not None else "   -  "

        log.info(
            "epoch %3d | val_loss %s | base_mae train %s val %s (best %s) "
            "| gap_mae train %s val %s (best %s)",
            self.current_epoch, _f(va_loss),
            _f(tr_base), _f(va_base), _f(self.best_val_base_mae),
            _f(tr_gap), _f(va_gap), _f(self.best_val_gap_mae),
        )

        # mirror the best metrics into the logger (e.g. W&B) for tracking. Log
        # via self.log (no explicit step) so it rides Lightning's own step and
        # does not trip W&B's monotonic-step warning.
        if va_base is not None:
            self.log("best_val_base_mae", self.best_val_base_mae, prog_bar=False)
        if va_gap is not None:
            self.log("best_val_gap_mae", self.best_val_gap_mae, prog_bar=False)
