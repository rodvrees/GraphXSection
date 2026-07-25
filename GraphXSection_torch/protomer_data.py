"""Protomer-aware dataset + collate for the anchored set-loss training
(protomer_ccs_predictor plan, Sections 1-3).

One dataset item = one (molecule, adduct) corpus row. It yields the SET of
forwarded candidate graphs plus the observed CCS peak set, so the anchored set
loss (lightning_modules) can regress the dominant candidate to the base peak and
match remaining peaks to remaining candidates.

Forwarding policy (keeps compute down; the loss only supervises these):
  - base-only adduct (no protomer candidates): the NEUTRAL parent graph, base peak only.
  - single-peak H+/- molecule: ONLY the dominant candidate (charged SMILES).
  - multi-peak H+/- molecule: ALL candidates (charged SMILES).

Decoupling: the graph is built from the (charged) candidate SMILES, but the
adduct/global descriptor is computed on the NEUTRAL parent SMILES so its
mass/mz/Gasteiger-quantile terms stay physical (the adduct string alone carries
the charge/proton bookkeeping). The base SigmaModel is unchanged: each candidate
is a separate graph in the batch and gets one predicted CCS; the collate returns
segment ids (`mol_index`) so the loss can group predictions by molecule.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.data import Dataset
from torch_geometric.data import Batch

from GraphXSection_torch.data_extract import MoleculeIon
from GraphXSection_torch.adduct_features import AdductFeatureExtractor

logger = logging.getLogger(__name__)

# Set by ProtomerCorpusDataset.precompute() before forking the worker pool, so
# fork children inherit the dataset (and its DataFrame) by copy-on-write instead
# of pickling it per task. Workers write their featurized shard to disk and
# return only the path, so the ~GB of results never crosses a pipe.
_PRECOMPUTE_DS = None


def _shard_worker(task: tuple) -> str:
    start, stop, shard_path = task
    items = [_PRECOMPUTE_DS._build_item(i) for i in range(start, stop)]
    torch.save(items, shard_path)
    del items
    return shard_path


class ProtomerCorpusDataset(Dataset):
    def __init__(self, corpus_df, featurizer_collection, extractor: AdductFeatureExtractor | None = None,
                 eval_all_candidates: bool = False, graph_smiles_mode: str = "charged",
                 primary_only: bool = False):
        """
        Parameters
        ----------
        eval_all_candidates : bool
            Training forwarding policy (False) forwards only the dominant
            candidate for single-peak molecules to save compute. Evaluation
            (True) forwards ALL candidates for every H+/- molecule so the full
            predicted peak set is available for the gap metric and the
            permutation null. Base-only adducts always forward the neutral parent.
        graph_smiles_mode : {"charged", "neutral"}
            "charged" (default) featurizes the charge-positioned candidate SMILES.
            "neutral" replaces every forwarded candidate with the neutral parent,
            so all candidate graphs of a molecule are identical (no protomer
            distinction). This is the charged-vs-neutral-input ablation: it must
            collapse the gap signal if the charged SMILES is what carries it.
        primary_only : bool
            Training-only ablation: forward ONLY the dominant candidate and keep
            ONLY the base (primary) peak for every molecule, so no secondary CCS
            is ever supervised and the gap term never fires (each molecule reduces
            to base-CCS regression). Ignored when eval_all_candidates is True, so
            evaluation still forwards all candidates to measure the gap a
            primary-only-trained model produces on the same set.
        """
        self.df = corpus_df.reset_index(drop=True)
        self.featurizer_collection = featurizer_collection
        self.extractor = extractor if extractor is not None else AdductFeatureExtractor()
        self.eval_all_candidates = eval_all_candidates
        self.graph_smiles_mode = graph_smiles_mode
        self.primary_only = primary_only
        self._cache: list | None = None

    def __len__(self) -> int:
        return len(self.df)

    def _forward_smiles(self, row) -> tuple[list[str], int]:
        """Return (candidate SMILES to forward, dominant local index)."""
        cands = list(row["candidate_smiles"]) if row["candidate_smiles"] is not None else []
        n_cands = len(cands)
        dom = int(row["dominant_site_idx"]) if row["dominant_site_idx"] is not None else -1
        peaks = list(row["observed_ccs_peaks"])

        if n_cands == 0 or dom < 0:
            # base-only adduct (metal/other) -> neutral parent, base peak only
            smis, dom_local = [row["parent_smiles"]], 0
        elif self.eval_all_candidates:
            smis, dom_local = cands, dom            # eval: all candidates
        elif self.primary_only:
            smis, dom_local = [cands[dom]], 0       # train: primary-only, no secondary
        elif len(peaks) >= 2:
            smis, dom_local = cands, dom            # multi-peak (train): all candidates
        else:
            smis, dom_local = [cands[dom]], 0       # single-peak (train): dominant only

        if self.graph_smiles_mode == "neutral":
            # strip charge: forward the neutral parent in every candidate slot
            smis = [row["parent_smiles"]] * len(smis)
        return smis, dom_local

    def _build_item(self, idx: int) -> dict[str, Any]:
        """Featurize one molecule (the expensive RDKit path). Called once per
        item when precomputing, or per access when uncached."""
        row = self.df.iloc[idx]
        smis, dom_local = self._forward_smiles(row)
        graphs = [MoleculeIon(s).get_data(self.featurizer_collection) for s in smis]
        # adduct descriptor on the NEUTRAL parent (charge comes from the adduct string)
        gfeat = torch.as_tensor(
            self.extractor.extract_features(row["adduct"], smiles=row["parent_smiles"]),
            dtype=torch.float32,
        )
        peaks = list(row["observed_ccs_peaks"])
        if len(graphs) == 1:
            peaks = peaks[:1]  # only the base peak is supervised when a single graph is forwarded
        return {
            "graphs": graphs,
            "global_feat": gfeat,
            "peaks": torch.as_tensor(peaks, dtype=torch.float32),
            "dominant_local_idx": dom_local,
        }

    def precompute(self, num_workers: int = 0, label: str = "",
                   cache_dir: str | None = None, cache_key: str | None = None) -> None:
        """Featurize the whole dataset once into an in-memory cache so epochs are
        no longer bottlenecked on per-item RDKit featurization (~5 ms/mol).

        Persistence: when cache_dir+cache_key are given, featurized shards are
        stored under cache_dir/cache_key. A complete shard set is loaded directly
        (subsequent runs / ablation variants sharing a key start instantly);
        otherwise it is computed and written there.

        Parallelism: `num_workers` fork processes each featurize a contiguous
        shard and torch.save it to disk, returning only the path, so the ~GB of
        results never crosses a pipe (returning objects through a Pool is slower
        than serial). Serve with DataLoader num_workers=0 afterwards.
        """
        import time as _time

        n = len(self)
        t0 = _time.time()
        shard_dir = Path(cache_dir) / cache_key if (cache_dir and cache_key) else None

        # 1. load a complete persisted cache
        if shard_dir is not None and (shard_dir / "manifest.json").exists():
            try:
                self._cache = self._load_shards(shard_dir, n)
                logger.info("Loaded %d cached items [%s] from %s in %.1fs",
                            n, label, shard_dir, _time.time() - t0)
                return
            except Exception as e:  # corrupt/partial cache -> recompute
                logger.warning("Cache load failed (%s); recomputing", e)

        # 2. compute (serial, or parallel via disk shards)
        persist = shard_dir if shard_dir is not None else None
        if persist is not None:
            persist.mkdir(parents=True, exist_ok=True)
        if num_workers and num_workers > 1 and n > 2000:
            self._cache = self._shard_precompute(n, num_workers, persist)
        else:
            self._cache = [self._build_item(i) for i in range(n)]
            if persist is not None:
                torch.save(self._cache, persist / "shard_00000.pt")
                self._write_manifest(persist, [("shard_00000.pt", 0, n)])
        logger.info("Precomputed %d items [%s] in %.1fs (%d workers)%s",
                    n, label, _time.time() - t0, num_workers,
                    f", cached to {shard_dir}" if shard_dir else "")

    def _shard_precompute(self, n: int, num_workers: int, persist: Path | None) -> list:
        import multiprocessing as mp
        import shutil
        import tempfile

        global _PRECOMPUTE_DS
        out_dir = persist if persist is not None else Path(tempfile.mkdtemp(prefix="protomer_pc_"))
        n_shards = num_workers * 4  # over-shard for load balance
        step = (n + n_shards - 1) // n_shards
        tasks, meta = [], []
        for k, start in enumerate(range(0, n, step)):
            stop = min(start + step, n)
            fname = f"shard_{k:05d}.pt"
            tasks.append((start, stop, str(out_dir / fname)))
            meta.append((fname, start, stop))

        _PRECOMPUTE_DS = self
        try:
            ctx = mp.get_context("fork")
            with ctx.Pool(num_workers) as pool:
                pool.map(_shard_worker, tasks)
        finally:
            _PRECOMPUTE_DS = None

        cache: list = [None] * n
        for fname, start, stop in meta:
            for off, item in enumerate(torch.load(out_dir / fname, weights_only=False)):
                cache[start + off] = item
        if persist is not None:
            self._write_manifest(persist, meta)
        else:
            shutil.rmtree(out_dir, ignore_errors=True)
        return cache

    @staticmethod
    def _write_manifest(shard_dir: Path, meta: list) -> None:
        import json as _json
        (shard_dir / "manifest.json").write_text(_json.dumps(meta))

    @staticmethod
    def _load_shards(shard_dir: Path, n: int) -> list:
        import json as _json
        meta = _json.loads((shard_dir / "manifest.json").read_text())
        total = sum(stop - start for _, start, stop in meta)
        if total != n:
            raise ValueError(f"cache size {total} != dataset {n}")
        cache: list = [None] * n
        for fname, start, stop in meta:
            for off, item in enumerate(torch.load(shard_dir / fname, weights_only=False)):
                cache[start + off] = item
        return cache

    def get(self, idx: int) -> dict[str, Any]:
        if self._cache is not None:
            return self._cache[idx]
        return self._build_item(idx)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.get(idx)


def collate_protomer(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Flatten all candidate graphs across molecules into one PyG Batch, with:
      - global_feats  [total_graphs, F]  (each molecule's descriptor replicated per candidate)
      - mol_index     [total_graphs]      segment id mapping each graph to its molecule
      - dominant_index[num_molecules]     GLOBAL index (into the flat graph list) of each
                                          molecule's dominant candidate
      - peaks         list[Tensor]        ragged observed peak set per molecule (base first)
      - charged_atom_index [total_graphs] global node index of each graph's charged
                                          atom (or -1 for neutral graphs), for the
                                          optional charged-atom readout
    """
    all_graphs: list = []
    global_feats: list = []
    mol_index: list = []
    dominant_index: list = []
    peaks: list = []
    charged_atom_index: list = []

    node_offset = 0
    for m, item in enumerate(batch):
        start = len(all_graphs)
        g = item["graphs"]
        all_graphs.extend(g)
        global_feats.extend([item["global_feat"]] * len(g))
        mol_index.extend([m] * len(g))
        dominant_index.append(start + int(item["dominant_local_idx"]))
        peaks.append(item["peaks"])
        for graph in g:
            local = int(getattr(graph, "charged_atom_local", -1))
            charged_atom_index.append(node_offset + local if local >= 0 else -1)
            node_offset += graph.num_nodes

    return {
        "graph": Batch.from_data_list(all_graphs),
        "global_feats": torch.stack(global_feats),
        "mol_index": torch.tensor(mol_index, dtype=torch.long),
        "dominant_index": torch.tensor(dominant_index, dtype=torch.long),
        "peaks": peaks,
        "charged_atom_index": torch.tensor(charged_atom_index, dtype=torch.long),
        "num_molecules": len(batch),
    }


class ProtomerDataModule(pl.LightningDataModule):
    """Lightning data module over the protomer training corpus.

    Loads the scaffold-split corpus parquets, builds `ProtomerCorpusDataset`
    with a shared featurizer collection and adduct extractor, and serves batches
    via `collate_protomer`. Gap (multi-CCS) molecules are a ~14% minority, so the
    training loader can oversample them with a `WeightedRandomSampler`
    (`gap_oversample_factor` in the model config, 1.0 = off).

    Feature-size properties mirror `GraphXSectionDataModule` so the same trainer
    wiring can build the model.
    """

    def __init__(
        self,
        config: dict[str, Any],
        featurizer_collection,
        train_df: Optional[pd.DataFrame] = None,
        valid_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None,
    ):
        super().__init__()
        self.config = config
        self.featurizer_collection = featurizer_collection
        self.extractor = AdductFeatureExtractor()

        model_cfg = config.get("model", {})
        self.batch_size = int(model_cfg.get("batch_size", 32))
        self.gap_oversample_factor = float(model_cfg.get("gap_oversample_factor", 1.0))
        self.graph_smiles_mode = str(model_cfg.get("graph_smiles_mode", "charged"))
        self.primary_only = bool(model_cfg.get("primary_only", False))

        import os

        default_workers = min(os.cpu_count() or 1, 8)
        self.num_workers = int(config.get("num_workers", default_workers))

        # Featurization is the bottleneck (per-item RDKit, ~5 ms/mol), so cache
        # the whole corpus once up front and serve tensors thereafter. Precompute
        # is parallelised; serving is then num_workers=0 (the cache lives in the
        # main process and a large in-memory dataset + fork workers blows up
        # memory via copy-on-read).
        self.cache_features = bool(config.get("cache_features", True))
        self.precompute_workers = int(
            config.get("precompute_workers", min(os.cpu_count() or 1, 16))
        )
        # Persistent featurization cache (survives across epochs, reruns, and
        # ablation variants that share a feature config). Local disk, not /mnt/c.
        self.cache_dir = config.get(
            "cache_dir", os.path.expanduser("~/.cache/protomer_precompute")
        )

        self._train_df = train_df
        self._valid_df = valid_df
        self._test_df = test_df
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None

    # -- loading -------------------------------------------------------------
    def _load(self, key: str, explicit: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        if explicit is not None:
            return explicit
        path = self.config.get(f"{key}_data_path")
        if path is None:
            return None
        logger.info("Loading %s corpus from %s", key, path)
        return pd.read_parquet(path)

    def _cache_key(self, ds, split: str) -> Optional[str]:
        """Identity of a split's featurization for the disk cache. None disables
        it (explicit in-memory DataFrames, whose provenance is not path-stable)."""
        import hashlib

        if getattr(self, f"_{split}_df") is not None:
            return None  # caller passed a DataFrame directly; skip disk cache
        path = self.config.get(f"{split}_data_path")
        if path is None:
            return None
        import os

        try:
            feat_bytes = open(self.config.get("feature_config_path", ""), "rb").read()
        except Exception:
            feat_bytes = b""
        try:
            mtime = str(os.path.getmtime(path))  # invalidate cache if the corpus changes
        except OSError:
            mtime = "0"
        sig = "|".join([
            split, str(path), str(len(ds)), mtime,
            hashlib.md5(feat_bytes).hexdigest(),
            ds.graph_smiles_mode, f"po{int(ds.primary_only)}",
            f"eval{int(ds.eval_all_candidates)}",
        ])
        return f"{split}_{hashlib.md5(sig.encode()).hexdigest()[:16]}"

    def setup(self, stage: Optional[str] = None):
        train_df = self._load("train", self._train_df)
        valid_df = self._load("valid", self._valid_df)
        test_df = self._load("test", self._test_df)
        if train_df is not None:
            self.train_dataset = ProtomerCorpusDataset(
                train_df, self.featurizer_collection, self.extractor,
                graph_smiles_mode=self.graph_smiles_mode, primary_only=self.primary_only,
            )
        if valid_df is not None:
            self.valid_dataset = ProtomerCorpusDataset(
                valid_df, self.featurizer_collection, self.extractor,
                graph_smiles_mode=self.graph_smiles_mode, primary_only=self.primary_only,
            )
        if test_df is not None:
            self.test_dataset = ProtomerCorpusDataset(
                test_df, self.featurizer_collection, self.extractor,
                graph_smiles_mode=self.graph_smiles_mode, primary_only=self.primary_only,
            )

        # Precompute the train/val features once (test is not used by fit; it is
        # cached lazily in test_dataloader). This is the fix for the CPU-bound
        # dataloader: after this, epochs are GPU/collate-bound.
        if self.cache_features:
            if self.train_dataset is not None:
                self.train_dataset.precompute(
                    self.precompute_workers, label="train",
                    cache_dir=self.cache_dir, cache_key=self._cache_key(self.train_dataset, "train"))
            if self.valid_dataset is not None:
                self.valid_dataset.precompute(
                    self.precompute_workers, label="valid",
                    cache_dir=self.cache_dir, cache_key=self._cache_key(self.valid_dataset, "valid"))

    def _loader_workers(self) -> int:
        # A cached dataset returns tensors instantly, so serve single-process to
        # avoid the fork copy-on-read memory growth of a large in-memory dataset.
        return 0 if self.cache_features else self.num_workers

    # -- loaders -------------------------------------------------------------
    def train_dataloader(self):
        sampler = None
        shuffle = True
        if self.gap_oversample_factor > 1.0:
            has_gap = self.train_dataset.df["has_gap"].to_numpy()
            weights = torch.where(
                torch.as_tensor(has_gap, dtype=torch.bool),
                torch.tensor(self.gap_oversample_factor),
                torch.tensor(1.0),
            ).double()
            sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
            shuffle = False
            logger.info(
                "Gap oversampling on: factor=%s (%d gap rows of %d)",
                self.gap_oversample_factor, int(has_gap.sum()), len(has_gap),
            )
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            collate_fn=collate_protomer,
            num_workers=self._loader_workers(),
            pin_memory=torch.cuda.is_available(),
            persistent_workers=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_protomer,
            num_workers=self._loader_workers(),
            pin_memory=torch.cuda.is_available(),
            persistent_workers=False,
        )

    def test_dataloader(self):
        if self.cache_features and self.test_dataset is not None and self.test_dataset._cache is None:
            self.test_dataset.precompute(
                self.precompute_workers, label="test",
                cache_dir=self.cache_dir, cache_key=self._cache_key(self.test_dataset, "test"))
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_protomer,
            num_workers=self._loader_workers(),
            pin_memory=torch.cuda.is_available(),
            persistent_workers=False,
        )

    # -- feature sizes (mirror GraphXSectionDataModule) ----------------------
    @property
    def node_feature_sizes(self):
        return self.featurizer_collection.node_feature_sizes

    @property
    def edge_feature_sizes(self):
        return self.featurizer_collection.edge_feature_sizes

    @property
    def global_feature_sizes(self) -> int:
        """Adduct-descriptor dimension (int), computed on a probe row."""
        return int(len(self.extractor.extract_features("[M+H]+", smiles="CCO")))
