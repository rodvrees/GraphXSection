"""Train the protomer-aware CCS model with the anchored set loss.

Wires the protomer corpus -> ProtomerDataModule -> ProtomerLightningModule
(SigmaModel backbone + anchored set loss + optional charged-atom readout), and
reuses the existing Lightning callback/logger builders so checkpointing, LR
scheduling and W&B behave exactly like the base pipeline.

IMPORTANT (graphamole import order): SigmaModel is imported first, before
feature_config, because feature_config rebinds the top-level `graphamole`
package to the inner package, which otherwise breaks Sigma's
`from graphamole.graphamole.layers ...` import.

Usage:
    python scripts/train_protomer.py --config configs/sigma_config_protomer.json
    python scripts/train_protomer.py --config configs/sigma_config_protomer.json --fast-dev-run
"""
from __future__ import annotations

# --- import order fix: pull in SigmaModel (hence graphamole.graphamole.*) first ---
from GraphXSection_torch.models.Sigma import SigmaModel  # noqa: F401,E402

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402

import pytorch_lightning as pl  # noqa: E402
import torch  # noqa: E402

from GraphXSection_torch.feature_config import load_feature_config  # noqa: E402
from GraphXSection_torch.protomer_data import ProtomerDataModule  # noqa: E402
from GraphXSection_torch.lightning_modules import ProtomerLightningModule  # noqa: E402
from GraphXSection_torch.train_lightning import (  # noqa: E402
    create_lightning_callbacks,
    create_lightning_logger,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("train_protomer")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="Path to sigma_config_protomer.json")
    p.add_argument("--fast-dev-run", action="store_true",
                   help="Run a single train/val batch to smoke-test the pipeline")
    p.add_argument("--max-epochs", type=int, default=None,
                   help="Override model.epochs")
    return p.parse_args()


def main():
    args = parse_args()
    config = json.load(open(args.config))

    # Tensor Cores: trade a little float32 precision for throughput on Ampere+.
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    feature_config_path = config["feature_config_path"]
    log.info("Loading featurizers from %s", feature_config_path)
    collection = load_feature_config(feature_config_path)

    dm = ProtomerDataModule(config, collection)
    dm.setup()
    log.info(
        "node=%s edge=%s global=%s",
        collection.node_feature_sizes, collection.edge_feature_sizes,
        dm.global_feature_sizes,
    )

    lm = ProtomerLightningModule(
        config,
        model_type="Sigma",
        node_feature_sizes=dm.node_feature_sizes,
        edge_feature_sizes=dm.edge_feature_sizes,
        global_feature_sizes=dm.global_feature_sizes,
    )

    callbacks = create_lightning_callbacks(config)
    lightning_logger = create_lightning_logger(config)

    lt = config.get("lightning", {})
    trainer = pl.Trainer(
        max_epochs=args.max_epochs or config["model"].get("epochs", 100),
        accelerator="auto",
        devices="auto",
        callbacks=callbacks,
        logger=lightning_logger,
        enable_progress_bar=True,
        enable_model_summary=True,
        gradient_clip_val=lt.get("gradient_clip_val", None),
        accumulate_grad_batches=lt.get("accumulate_grad_batches", 1),
        precision=lt.get("precision", 32),
        deterministic=lt.get("deterministic", False),
        enable_checkpointing=bool(config.get("mcp", False)),
        fast_dev_run=args.fast_dev_run or lt.get("fast_dev_run", False),
    )

    trainer.fit(lm, dm)

    # restore best checkpoint (by the configured monitor) before saving
    from pytorch_lightning.callbacks import ModelCheckpoint

    if config.get("use_best_model", False) and config.get("mcp", False):
        best_path = None
        for cb in trainer.callbacks:
            if isinstance(cb, ModelCheckpoint) and getattr(cb, "best_model_path", None):
                best_path = cb.best_model_path
                break
        if best_path and os.path.exists(best_path):
            ckpt = torch.load(best_path, map_location="cpu")
            lm.load_state_dict(ckpt["state_dict"], strict=True)
            log.info("Loaded best checkpoint: %s", best_path)

    if config.get("save_model", False) and not (args.fast_dev_run or lt.get("fast_dev_run", False)):
        out_dir = Path(config.get("output_path", "./outputs/"))
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{config['run_name']}_protomer_model.pth"
        torch.save(lm.state_dict(), out_path)
        log.info("Saved model to %s", out_path)

    log.info("Done")


if __name__ == "__main__":
    main()
