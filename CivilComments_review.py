#!/usr/bin/env python3
# ============================================================
# Ready-to-run: TEXT monitoring under shift with ACSIncome-style
# ablations, pure CivilComments ID/OOD by default, and
# reviewer-oriented fairness/statistics reporting.
#
# Key features:
# - Pure CivilComments ID/OOD by default (OOD dataset defaults to None => same as ID)
# - ACSIncome-style ablations:
#     * full
#     * abl_no_swap
#     * abl_no_perm
#   with per-experiment `exp` column in all tables
# - Male vs female fairness only (ambiguous / unknown dropped)
# - Reviewer-facing dataset / class / group / target-event statistics
# - Outputs aligned with richer CivilComments->Jigsaw pipeline naming
#
# FAIRNESS POLICY ADDITIONS:
#   - Global selection (original)
#   - Parity (equal alert rates across groups)
#   - Bounded (budget within ±delta of global per group)
#   - All policies are applied to all detectors; results are saved with
#     policy and delta columns in the fairness and alert tables.
#
# Outputs (OUTDIR):
#   tables/
#     raw_monitor_overall.csv
#     raw_monitor_alerts.csv
#     raw_monitor_fairness_alerts.csv
#     raw_monitor_group_alerts.csv
#     raw_dataset_statistics.csv
#     raw_target_statistics.csv
#     run_metadata.csv
#     summary_monitor_overall_mean_std_ci.csv
#     summary_monitor_alerts_mean_std_ci.csv
#     summary_monitor_fairness_mean_std_ci.csv
#     summary_monitor_group_mean_std_ci.csv
#     summary_dataset_statistics_mean_std_ci.csv
#     summary_target_statistics_mean_std_ci.csv
#     summary_monitor_tradeoff_mean_std_ci.csv
#     ablation_summary_budget0.05.csv
#   report.md
#   OUTDIR.zip
# ============================================================

import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HUGGINGFACE_HUB_BASE_URL", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from huggingface_hub import login
# --- LOGIN (this was missing) ---
# Hardcoded token removed; use environment variable HF_TOKEN
# login(token=HF_TOKEN, add_to_git_credential=True)   # or simply login(token=HF_TOKEN)
HF_TOKEN = os.environ.get("HF_TOKEN", None)
if HF_TOKEN is not None:
    login(token=HF_TOKEN, add_to_git_credential=True)

import sys
import json
import math
import random
import zipfile
import argparse
import subprocess
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

def pip_install(pkgs: List[str]):
    subprocess.check_call([sys.executable, "-m", "pip", "-q", "install"] + pkgs)


pip_install([
    "numpy",
    "pandas",
    "tqdm",
    "scikit-learn>=1.2",
    "tabulate",
    "transformers>=4.38.0",
    "datasets>=2.18.0",
    "accelerate>=0.26.0",
    "huggingface_hub>=0.20.0",
])

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.isotonic import IsotonicRegression
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModel


# -----------------------------
# Args & configs
# -----------------------------
@dataclass
class Args:
    dataset: str = "civilcomments"
    ood_dataset: Optional[str] = None

    OUTDIR: str = "./out_text_cc_pure_ablations"
    CACHE_DIR: str = "./hf_cache"

    DEVICE: str = "auto"
    DETERMINISTIC: bool = True
    NUM_WORKERS: int = 0

    SEEDS: List[int] = None
    MAX_TRAIN: int = 80000
    MAX_VAL: int = 30000
    MAX_ID_TEST: int = 30000
    MAX_OOD_VAL: int = 20000
    MAX_OOD_TEST: int = 30000
    N_TARGET: int = 50000

    MODEL_NAME: str = "distilbert-base-uncased"
    MAX_LEN: int = 160
    BATCH_SIZE: int = 32
    EPOCHS_CLS: int = 10
    LR_CLS: float = 2e-5
    WD: float = 0.01
    EARLY_STOP_METRIC: str = "nll"
    EARLY_STOP_PATIENCE: int = 2
    EARLY_STOP_MIN_DELTA: float = 1e-4

    USE_TEMP_SCALING: bool = True
    TS_MAX_ITERS: int = 200

    HIGH_CONF_REGION: float = 0.80
    TAU_TARGET: float = 0.90
    ALERT_BUDGETS: List[float] = None
    RHO_VALUES: List[float] = None

    DRIFT_BASE: float = 0.05
    DRIFT_GAIN: float = 0.55
    DRIFT_ON_ID_AT_HIGH_RHO: bool = True
    ID_DRIFT_SCALE: float = 0.25

    IDENTITY_THRESH: float = 0.5

    TAU_LIST: List[float] = None
    KOF_HIDDEN: int = 256
    EPOCHS_DETECT: int = 3
    LR_DETECT: float = 2e-3

    EPOCHS_TCPM: int = 8
    TCPM_EARLY_STOP_PATIENCE: int = 2
    TCPM_EARLY_STOP_MIN_DELTA: float = 1e-4
    TCPM_TRAIN_ON_TAU_SLICE: bool = True
    TCPM_USE_POS_WEIGHT: bool = True
    TCPM_APPLY_TAU_GATE_AT_INFERENCE: bool = True

    USE_STABILITY: bool = True
    STAB_K: int = 4
    STAB_TOKEN_DROP_PROB: float = 0.20
    STAB_CHAR_NOISE_PROB: float = 0.08
    STAB_BATCH_MULT: int = 4

    BOOTSTRAP_N: int = 1200
    BOOTSTRAP_ALPHA: float = 0.05

    # --- FAIRNESS POLICY ADDITIONS ---
    ENABLE_FAIRNESS_MITIGATION: bool = True
    FAIRNESS_POLICIES: List[str] = None
    FAIR_TOPK_DELTAS: List[float] = None
    # Models to apply mitigation to; if empty, apply to all.
    MITIGATE_MODELS: List[str] = None


@dataclass
class ExperimentConfig:
    name: str
    use_swap_drift: bool = True
    use_perm_drift: bool = True
    enabled: bool = True


def make_args() -> Args:
    a = Args()
    a.SEEDS = list(range(42, 62))
    a.RHO_VALUES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]   # for quick test; change to [0.0,0.2,0.4,0.6,0.8,1.0] for full
    a.TAU_LIST = [0.70, 0.80, 0.90, 0.95]
    a.ALERT_BUDGETS = [0.05, 0.10]
    a.dataset = "civilcomments"
    a.ood_dataset = None

    # --- FAIRNESS POLICY ADDITIONS ---
    a.FAIRNESS_POLICIES = ["global", "bounded"]   # can add "parity" if needed, but bounded preserves performance better
    a.FAIR_TOPK_DELTAS = [0.01, 0.02]   # small deltas to gently enforce near-equality
    # Mitigate all models (or specify a list, e.g., ["tcp_m_gate_tau"])
    a.MITIGATE_MODELS = []   # empty => apply to all models
    return a


args = make_args()

EXPERIMENTS = [
    ExperimentConfig(name="full", use_swap_drift=True, use_perm_drift=True, enabled=True),
    # ExperimentConfig(name="abl_no_swap", use_swap_drift=False, use_perm_drift=True, enabled=True),
    # ExperimentConfig(name="abl_no_perm", use_swap_drift=True, use_perm_drift=False, enabled=True),
]


def dataset_label() -> str:
    ood = args.ood_dataset or args.dataset
    if (ood or "").lower().strip() == (args.dataset or "").lower().strip():
        return args.dataset
    return f"{args.dataset}->{ood}"

# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_determinism(enable: bool):
    if not enable:
        return
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def get_device(name: str):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def ensure_outdirs(outdir: str):
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, "tables"), exist_ok=True)


def safe_auc(y_true, y_score) -> float:
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_score = np.asarray(y_score).astype(float).reshape(-1)
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def batch_iterable(n: int, bs: int):
    for i in range(0, n, bs):
        yield i, min(n, i + bs)


def p_drift_from_rho(rho: float) -> float:
    return float(np.clip(args.DRIFT_BASE + args.DRIFT_GAIN * float(rho), 0.0, 1.0))


def mean_std_ci_over_seeds(
    df: pd.DataFrame,
    group_cols: List[str],
    metric_cols: List[str],
    seed_col: str = "seed",
    n_boot: int = 1000,
    alpha: float = 0.05,
) -> pd.DataFrame:
    out_rows = []
    base = df.copy()
    base_seed = base.groupby(group_cols + [seed_col], as_index=False)[metric_cols].mean()

    nb = int(n_boot) if (n_boot is not None and int(n_boot) > 0) else 0
    rng = np.random.default_rng(12345)

    for keys, sub in base_seed.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {c: v for c, v in zip(group_cols, keys)}

        for m in metric_cols:
            vals = sub[m].to_numpy(dtype=float)
            row[f"{m}_mean"] = float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")
            row[f"{m}_std"] = float(np.nanstd(vals, ddof=1)) if np.isfinite(vals).sum() > 1 else float("nan")

            seed_vals = sub.groupby(seed_col)[m].mean().to_numpy(dtype=float)
            finite = seed_vals[np.isfinite(seed_vals)]
            S_eff = int(len(finite))
            row[f"{m}_n_seed_finite"] = S_eff

            if nb < 1 or S_eff < 2:
                row[f"{m}_ci_low"] = float("nan")
                row[f"{m}_ci_high"] = float("nan")
            else:
                boot = np.empty(nb, dtype=float)
                for bi in range(nb):
                    idx = rng.integers(0, S_eff, size=S_eff)
                    boot[bi] = float(np.mean(finite[idx]))
                row[f"{m}_ci_low"] = float(np.quantile(boot, alpha / 2))
                row[f"{m}_ci_high"] = float(np.quantile(boot, 1 - alpha / 2))

        out_rows.append(row)

    return pd.DataFrame(out_rows)

# -----------------------------
# HF wrappers
# -----------------------------
def _set_hf_endpoint(endpoint: str):
    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint
        os.environ["HUGGINGFACE_HUB_BASE_URL"] = endpoint


def hf_load_dataset(name: str, cache_dir: str, token: Optional[str], hf_endpoint: Optional[str] = None):
    endpoints = []
    if hf_endpoint:
        endpoints.append(hf_endpoint)
    endpoints.append(os.environ.get("HF_ENDPOINT", ""))
    endpoints.append("https://huggingface.co")

    last_err = None
    for ep in endpoints:
        try:
            if ep:
                _set_hf_endpoint(ep)
            try:
                return load_dataset(name, cache_dir=cache_dir, token=token)
            except TypeError:
                return load_dataset(name, cache_dir=cache_dir, use_auth_token=token)
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Failed to load dataset {name}. Last error: {last_err}")


def hf_from_pretrained(cls, model_name: str, cache_dir: str, token: Optional[str], hf_endpoint: Optional[str] = None):
    endpoints = []
    if hf_endpoint:
        endpoints.append(hf_endpoint)
    endpoints.append(os.environ.get("HF_ENDPOINT", ""))
    endpoints.append("https://huggingface.co")

    last_err = None
    for ep in endpoints:
        try:
            if ep:
                _set_hf_endpoint(ep)
            try:
                return cls.from_pretrained(model_name, cache_dir=cache_dir, token=token)
            except TypeError:
                return cls.from_pretrained(model_name, cache_dir=cache_dir, use_auth_token=token)
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Failed from_pretrained({model_name}) for {cls}. Last error: {last_err}")


def _hf_repo_candidates(ds_name: str) -> List[str]:
    n = (ds_name or "").lower().strip()
    if n in ["civilcomments", "civil_comments"]:
        return [
            "pietrolesci/civilcomments-wilds",
        ]
    if n in ["jigsaw", "jigsaw_unintended_bias"]:
        return [
            "james-burton/jigsaw_unintended_bias100K",
        ]
    raise ValueError(f"Unknown dataset: {ds_name}")


def hf_load_dataset_with_fallback(repo_candidates: List[str], cache_dir: str, token: Optional[str], hf_endpoint: Optional[str] = None):
    errs = []
    for repo in repo_candidates:
        try:
            ds = hf_load_dataset(repo, cache_dir=cache_dir, token=token, hf_endpoint=hf_endpoint)
            print(f"[HF] loaded dataset repo: {repo}")
            return ds, repo
        except Exception as e:
            errs.append(f"{repo}: {repr(e)}")
    raise RuntimeError("Failed to load any dataset repo candidate:\n" + "\n".join(errs))


# -----------------------------
# Dataset loading
# -----------------------------
def _infer_text_col(df: pd.DataFrame) -> str:
    for c in ["comment_text", "text", "comment", "sentence", "content"]:
        if c in df.columns:
            return c
    obj = [c for c in df.columns if df[c].dtype == object]
    if obj:
        return obj[0]
    return df.columns[0]


def _infer_label_col(df: pd.DataFrame) -> str:
    for c in ["toxicity", "target", "label", "y"]:
        if c in df.columns:
            return c
    raise RuntimeError("Could not infer label column (expected one of toxicity/target/label/y).")


def _coerce_binary_label(x):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return 0
    if isinstance(x, (np.integer, int)):
        return int(x != 0)
    if isinstance(x, (np.floating, float)):
        return int(float(x) >= 0.5)
    if isinstance(x, (bool, np.bool_)):
        return int(bool(x))
    s = str(x).strip().lower()
    if s in ["toxic", "1", "true", "yes"]:
        return 1
    return 0


def _gender_from_identity_cols(row: pd.Series, thresh: float) -> int:
    m = row.get("male", np.nan)
    f = row.get("female", np.nan)
    m = float(m) if (m is not None and np.isfinite(m)) else 0.0
    f = float(f) if (f is not None and np.isfinite(f)) else 0.0
    is_m = (m >= thresh)
    is_f = (f >= thresh)
    if is_m and (not is_f):
        return 1
    if is_f and (not is_m):
        return 0
    return -1


def _pick_split_name(ds: DatasetDict, candidates: List[str]) -> Optional[str]:
    keys = list(ds.keys())
    low = {k.lower(): k for k in keys}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    return None


def load_text_dataset_splits(seed: int, hf_endpoint: Optional[str] = None) -> Dict[str, Dict[str, object]]:
    hf_token = os.environ.get("HF_TOKEN", None)
    cache_dir = args.CACHE_DIR

    id_name = args.dataset
    ood_name = args.ood_dataset or args.dataset

    ds_id, id_repo_used = hf_load_dataset_with_fallback(
        _hf_repo_candidates(id_name),
        cache_dir=cache_dir,
        token=hf_token,
        hf_endpoint=hf_endpoint,
    )

    if ood_name.lower().strip() == id_name.lower().strip():
        ds_ood, ood_repo_used = ds_id, id_repo_used
    else:
        ds_ood, ood_repo_used = hf_load_dataset_with_fallback(
            _hf_repo_candidates(ood_name),
            cache_dir=cache_dir,
            token=hf_token,
            hf_endpoint=hf_endpoint,
        )

    tr_name = _pick_split_name(ds_id, ["train", "training"])
    va_name = _pick_split_name(ds_id, ["validation", "val", "valid", "dev", "development"])
    if tr_name is None:
        raise RuntimeError(f"No train-like split found for ID dataset. Available: {list(ds_id.keys())}")

    df_tr = ds_id[tr_name].to_pandas()

    if va_name is None:
        idx = np.arange(len(df_tr))
        rng = np.random.default_rng(seed + 111)
        rng.shuffle(idx)
        n_val = max(1000, int(0.10 * len(idx)))
        n_val = min(n_val, len(idx) - 1)
        val_idx = idx[:n_val]
        tr_idx = idx[n_val:]
        df_va = df_tr.iloc[val_idx].reset_index(drop=True)
        df_tr = df_tr.iloc[tr_idx].reset_index(drop=True)
        va_name_eff = "CREATED_FROM_TRAIN"
    else:
        df_va = ds_id[va_name].to_pandas()
        va_name_eff = va_name

    te_name = _pick_split_name(ds_ood, ["test", "testing", "eval", "evaluation"])
    if te_name is None:
        te_name = _pick_split_name(ds_ood, ["validation", "val", "valid", "dev", "development"])
    if te_name is None:
        te_name = _pick_split_name(ds_ood, ["train", "training"])
    if te_name is None:
        te_name = list(ds_ood.keys())[0]
    df_ood = ds_ood[te_name].to_pandas()

    text_col_id = _infer_text_col(df_tr)
    label_col_id = _infer_label_col(df_tr)
    text_col_ood = _infer_text_col(df_ood)
    label_col_ood = _infer_label_col(df_ood)

    def extract(df: pd.DataFrame, text_col: str, label_col: str):
        texts = df[text_col].astype(str).tolist()
        y = np.array([_coerce_binary_label(v) for v in df[label_col].tolist()], dtype=np.int64)
        if ("male" in df.columns) and ("female" in df.columns):
            g = np.array([_gender_from_identity_cols(df.iloc[i], args.IDENTITY_THRESH) for i in range(len(df))], dtype=np.int64)
        else:
            g = -np.ones(len(df), dtype=np.int64)
        return texts, y, g

    tr_text, tr_y, tr_g = extract(df_tr, text_col_id, label_col_id)
    va_text, va_y, va_g = extract(df_va, text_col_id, label_col_id)
    ood_text, ood_y, ood_g = extract(df_ood, text_col_ood, label_col_ood)

    def drop_unknown(texts, y, g):
        m = g >= 0
        texts2 = [t for t, keep in zip(texts, m) if keep]
        return texts2, y[m], g[m]

    tr_text, tr_y, tr_g = drop_unknown(tr_text, tr_y, tr_g)
    va_text, va_y, va_g = drop_unknown(va_text, va_y, va_g)
    ood_text, ood_y, ood_g = drop_unknown(ood_text, ood_y, ood_g)

    rng = np.random.default_rng(seed + 12345)

    idx = np.arange(len(tr_y))
    rng.shuffle(idx)
    n_id_test = int(round(0.10 * len(idx)))
    n_id_test = min(n_id_test, max(1, len(idx) - 1))
    id_test_idx = idx[:n_id_test]
    tr_idx = idx[n_id_test:]

    idx_ood = np.arange(len(ood_y))
    rng.shuffle(idx_ood)
    n_ood_val = int(round(0.10 * len(idx_ood)))
    n_ood_val = min(n_ood_val, max(1, len(idx_ood) - 1))
    ood_val_idx = idx_ood[:n_ood_val]
    ood_test_idx = idx_ood[n_ood_val:]

    def take(texts, y, g, ix):
        return {"text": [texts[i] for i in ix], "y": y[ix].astype(np.int64), "g": g[ix].astype(np.int64)}

    splits = {
        "train": take(tr_text, tr_y, tr_g, tr_idx),
        "validation": {"text": va_text, "y": va_y.astype(np.int64), "g": va_g.astype(np.int64)},
        "id_test": take(tr_text, tr_y, tr_g, id_test_idx),
        "ood_validation": take(ood_text, ood_y, ood_g, ood_val_idx),
        "ood_test": take(ood_text, ood_y, ood_g, ood_test_idx),
        "meta": {
            "id_dataset": id_name,
            "ood_dataset": ood_name,
            "id_repo_used": id_repo_used,
            "ood_repo_used": ood_repo_used,
            "id_train_split": tr_name,
            "id_val_split": va_name_eff,
            "ood_pool_split": te_name,
            "text_col": text_col_id,
            "label_col": label_col_id,
            "ood_text_col": text_col_ood,
            "ood_label_col": label_col_ood,
        },
    }

    def subsample(split, max_n, seed_offset):
        if max_n is None or max_n <= 0:
            return split
        n = len(split["y"])
        if n <= max_n:
            return split
        rr = np.random.default_rng(seed + seed_offset)
        ix = rr.choice(n, size=int(max_n), replace=False)
        ix = np.sort(ix)
        return {"text": [split["text"][i] for i in ix], "y": split["y"][ix], "g": split["g"][ix]}

    splits["train"] = subsample(splits["train"], args.MAX_TRAIN, 1)
    splits["validation"] = subsample(splits["validation"], args.MAX_VAL, 2)
    splits["id_test"] = subsample(splits["id_test"], args.MAX_ID_TEST, 3)
    splits["ood_validation"] = subsample(splits["ood_validation"], args.MAX_OOD_VAL, 4)
    splits["ood_test"] = subsample(splits["ood_test"], args.MAX_OOD_TEST, 5)

    for split_name in ["train", "validation", "id_test", "ood_validation", "ood_test"]:
        n_split = len(splits[split_name]["y"])
        if n_split == 0:
            raise RuntimeError(
                f"Split '{split_name}' is empty after group filtering. "
                f"id_repo_used={id_repo_used}, ood_repo_used={ood_repo_used}."
            )

    return splits


# -----------------------------
# Dataset stats
# -----------------------------
def summarize_split(seed: int, exp: str, split_name: str, split: Dict[str, object]) -> Dict[str, object]:
    y = np.asarray(split["y"]).astype(int)
    g = np.asarray(split["g"]).astype(int)
    n = int(len(y))
    n_pos = int(y.sum())
    n_neg = int(n - n_pos)
    n_male = int((g == 1).sum())
    n_female = int((g == 0).sum())
    return {
        "seed": int(seed),
        "exp": exp,
        "split": split_name,
        "n": n,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "pos_rate": float(n_pos / n) if n > 0 else float("nan"),
        "n_male": n_male,
        "n_female": n_female,
        "male_frac": float(n_male / n) if n > 0 else float("nan"),
        "female_frac": float(n_female / n) if n > 0 else float("nan"),
        "n_pos_male": int(((y == 1) & (g == 1)).sum()),
        "n_pos_female": int(((y == 1) & (g == 0)).sum()),
    }

# -----------------------------
# Tokenization/model
# -----------------------------
def tokenize_texts(tokenizer, texts: List[str], max_len: int):
    if texts is None or len(texts) == 0:
        raise RuntimeError("tokenize_texts received an empty text list.")
    return tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=int(max_len),
        return_tensors="pt"
    )


class EncDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, y):
        self.enc = encodings
        self.y = torch.from_numpy(np.asarray(y).astype(np.int64)).long()

    def __len__(self):
        return int(self.y.numel())

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.enc.items()}
        return item, self.y[idx]


class BertBinary(nn.Module):
    def __init__(self, model_name: str, cache_dir: str, token: Optional[str], hf_endpoint: Optional[str] = None):
        super().__init__()
        self.enc = hf_from_pretrained(AutoModel, model_name, cache_dir=cache_dir, token=token, hf_endpoint=hf_endpoint)
        hid = self.enc.config.hidden_size
        self.drop = nn.Dropout(0.2)
        self.out = nn.Linear(hid, 2)

    def forward(self, batch):
        o = self.enc(**batch)
        h = o.last_hidden_state[:, 0, :]
        h = self.drop(h)
        logits = self.out(h)
        return h, logits


def probs_from_logits(logits):
    return F.softmax(logits, dim=-1)


def conf_from_logits(logits):
    return probs_from_logits(logits).max(dim=-1).values


def entropy_from_logits(logits):
    p = probs_from_logits(logits).clamp_min(1e-8)
    return -(p * p.log()).sum(dim=-1)


def margin_from_logits(logits):
    top2 = torch.topk(logits, k=2, dim=-1).values
    return top2[:, 0] - top2[:, 1]


def energy_from_logits(logits):
    return -torch.logsumexp(logits, dim=-1)


class TempScaler(nn.Module):
    def __init__(self, init_T: float = 1.0):
        super().__init__()
        self.logT = nn.Parameter(torch.tensor([math.log(init_T)], dtype=torch.float32))

    def forward(self, logits):
        T = torch.exp(self.logT).clamp(1e-3, 1e3)
        return logits / T


@torch.no_grad()
def eval_cls(model, loader, device, temp_scaler: Optional[TempScaler] = None):
    model.eval()
    if temp_scaler is not None:
        temp_scaler.eval()
    tot, correct, nll_sum = 0, 0, 0.0
    for batch, y in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        y = y.to(device)
        _, logits = model(batch)
        if temp_scaler is not None:
            logits = temp_scaler(logits)
        nll_sum += float(F.cross_entropy(logits, y, reduction="sum").item())
        correct += int((logits.argmax(dim=-1) == y).sum().item())
        tot += int(y.numel())
    return {"acc": correct / max(1, tot), "nll": nll_sum / max(1, tot)}


def train_cls_with_early_stopping(model, train_loader, val_loader, device):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.LR_CLS, weight_decay=args.WD)

    best_state = None
    best_epoch = 0
    bad = 0

    metric = args.EARLY_STOP_METRIC.lower().strip()
    if metric not in ["nll", "acc"]:
        metric = "nll"

    if metric == "nll":
        best_val = float("inf")

        def is_better(v):
            return (best_val - v) > float(args.EARLY_STOP_MIN_DELTA)
    else:
        best_val = -float("inf")

        def is_better(v):
            return (v - best_val) > float(args.EARLY_STOP_MIN_DELTA)

    for ep in range(1, int(args.EPOCHS_CLS) + 1):
        model.train()
        losses = []
        for batch, y in tqdm(train_loader, desc=f"[CLS] ep {ep}/{args.EPOCHS_CLS}", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            y = y.to(device)
            _, logits = model(batch)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))

        m = eval_cls(model, val_loader, device)
        print(f"[CLS] ep={ep:02d} train_loss={np.mean(losses):.4f} val_acc={m['acc']:.4f} val_nll={m['nll']:.4f}")

        cur = float(m[metric])
        if is_better(cur):
            best_val = cur
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = ep
            bad = 0
        else:
            bad += 1
            if bad >= int(args.EARLY_STOP_PATIENCE):
                print(f"[CLS] Early stopping at ep={ep:02d}. Best ep={best_epoch:02d} best_{metric}={best_val:.6f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"[CLS] Restored best checkpoint from ep={best_epoch:02d} (best_{metric}={best_val:.6f})")
    return model


def fit_temperature_scaler(model, calib_loader, device) -> TempScaler:
    scaler = TempScaler(init_T=1.0).to(device)
    opt = torch.optim.LBFGS(scaler.parameters(), max_iter=args.TS_MAX_ITERS, line_search_fn="strong_wolfe")

    model.eval()
    logits_list, y_list = [], []
    with torch.no_grad():
        for batch, y in calib_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            _, logits = model(batch)
            logits_list.append(logits.detach())
            y_list.append(y.to(device))
    logits_all = torch.cat(logits_list, dim=0)
    y_all = torch.cat(y_list, dim=0)

    def closure():
        opt.zero_grad(set_to_none=True)
        scaled = scaler(logits_all)
        loss = F.cross_entropy(scaled, y_all)
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        T = float(torch.exp(scaler.logT).item())
    print(f"[TempScaling] fitted T={T:.4f}")
    return scaler


@torch.no_grad()
def make_feat_batch(model, batch, temp_scaler: Optional[TempScaler] = None):
    h, logits_raw = model(batch)
    logits = temp_scaler(logits_raw) if temp_scaler is not None else logits_raw
    conf = conf_from_logits(logits)
    ent = entropy_from_logits(logits)
    marg = margin_from_logits(logits)
    feats = torch.cat([h, logits, marg.unsqueeze(-1), ent.unsqueeze(-1)], dim=-1)
    return feats, logits_raw, logits, conf, marg, ent


# -----------------------------
# Drift operators for ablations
# -----------------------------
def _tokenize_simple(s: str) -> List[str]:
    if not isinstance(s, str):
        s = str(s)
    toks = s.split()
    return toks if len(toks) else [s]


def swap_adjacent_tokens(text: str, severity: float, rng: np.random.Generator) -> str:
    toks = _tokenize_simple(text)
    if len(toks) < 2:
        return " ".join(toks)
    out = toks[:]
    local_p = float(np.clip(0.15 + 0.70 * severity, 0.0, 0.95))
    i = 0
    while i < len(out) - 1:
        if rng.random() < local_p:
            out[i], out[i + 1] = out[i + 1], out[i]
            i += 2
        else:
            i += 1
    return " ".join(out)


def permute_local_span(text: str, severity: float, rng: np.random.Generator) -> str:
    toks = _tokenize_simple(text)
    n = len(toks)
    if n < 3:
        return " ".join(toks)
    out = toks[:]
    max_span = int(min(8, max(3, round(3 + 4 * severity))))
    span = int(rng.integers(3, max_span + 1))
    span = min(span, n)
    start = int(rng.integers(0, max(1, n - span + 1)))
    chunk = out[start:start + span]
    if len(chunk) >= 3:
        rng.shuffle(chunk)
    out[start:start + span] = chunk
    return " ".join(out)


def apply_combined_text_drift(
    cfg: ExperimentConfig,
    texts: List[str],
    prob: float,
    rng: np.random.Generator,
) -> Tuple[List[str], Dict[str, np.ndarray]]:
    prob = float(np.clip(prob, 0.0, 1.0))
    out = []
    swap_mask = np.zeros(len(texts), dtype=np.int8)
    perm_mask = np.zeros(len(texts), dtype=np.int8)

    for i, text in enumerate(texts):
        t = text
        if cfg.use_swap_drift and rng.random() < prob:
            t = swap_adjacent_tokens(t, severity=prob, rng=rng)
            swap_mask[i] = 1
        if cfg.use_perm_drift and rng.random() < prob:
            t = permute_local_span(t, severity=prob, rng=rng)
            perm_mask[i] = 1
        out.append(t)

    return out, {"swap": swap_mask, "perm": perm_mask, "any": ((swap_mask + perm_mask) > 0).astype(np.int8)}


# -----------------------------
# Stability features
# -----------------------------
def _get_punct_token_ids(tokenizer) -> List[int]:
    candidates = [".", ",", "!", "?", ";", ":", "-", "_", "(", ")", "[", "]"]
    ids = []
    for t in candidates:
        try:
            tid = tokenizer.convert_tokens_to_ids(t)
            if isinstance(tid, int) and tid >= 0 and tid not in tokenizer.all_special_ids:
                ids.append(tid)
        except Exception:
            continue
    return sorted(list(set(ids)))


@torch.no_grad()
def compute_stability_features_from_encodings(
    model: nn.Module,
    temp_scaler: Optional[TempScaler],
    tokenizer,
    encodings: Dict[str, torch.Tensor],
    y: np.ndarray,
    device: torch.device,
    K: int,
    seed: int,
    token_drop_prob: float,
    token_noise_prob: float,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    if temp_scaler is not None:
        temp_scaler.eval()

    rng = np.random.default_rng(seed + 2025)

    input_ids_all = encodings["input_ids"].cpu().numpy().astype(np.int64)
    attn_all = encodings["attention_mask"].cpu().numpy().astype(np.int64)
    n, _ = input_ids_all.shape

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    special_ids = set(getattr(tokenizer, "all_special_ids", []))
    punct_ids = _get_punct_token_ids(tokenizer)
    punct_ids_arr = np.array(punct_ids, dtype=np.int64) if len(punct_ids) else None

    conf0 = np.empty(n, dtype=np.float32)
    pred0 = np.empty(n, dtype=np.int64)

    for s, e in batch_iterable(n, batch_size):
        batch = {
            "input_ids": torch.from_numpy(input_ids_all[s:e]).to(device),
            "attention_mask": torch.from_numpy(attn_all[s:e]).to(device),
        }
        if "token_type_ids" in encodings:
            batch["token_type_ids"] = encodings["token_type_ids"][s:e].to(device)

        _, lg_raw = model(batch)
        lg = temp_scaler(lg_raw) if temp_scaler is not None else lg_raw
        conf0[s:e] = conf_from_logits(lg).detach().cpu().numpy().astype(np.float32)
        pred0[s:e] = lg.argmax(dim=-1).detach().cpu().numpy().astype(np.int64)

    conf_ks = np.empty((n, K), dtype=np.float32)
    flip_ks = np.empty((n, K), dtype=np.float32)
    margin_ks = np.empty((n, K), dtype=np.float32)
    ent_ks = np.empty((n, K), dtype=np.float32)

    for kk in range(K):
        for s, e in batch_iterable(n, batch_size):
            ids0 = input_ids_all[s:e].copy()
            am0 = attn_all[s:e].copy()

            special_mask = np.isin(ids0, np.fromiter(special_ids, dtype=np.int64)) if len(special_ids) else np.zeros_like(ids0, dtype=bool)
            cand = (am0 == 1) & (~special_mask)

            if token_drop_prob > 0:
                drop = (rng.random(size=ids0.shape) < float(token_drop_prob)) & cand
                am0[drop] = 0
                ids0[drop] = pad_id

                cand_count = cand.sum(axis=1)
                kept_count = ((am0 == 1) & cand).sum(axis=1)
                need_fix = (cand_count > 0) & (kept_count == 0)
                if np.any(need_fix):
                    for i in np.where(need_fix)[0]:
                        j = int(np.where(cand[i])[0][0])
                        am0[i, j] = 1
                        ids0[i, j] = input_ids_all[s + i, j]

            if token_noise_prob > 0 and punct_ids_arr is not None and len(punct_ids_arr) > 0:
                cand2 = (am0 == 1) & (~special_mask)
                noise = (rng.random(size=ids0.shape) < float(token_noise_prob)) & cand2
                if np.any(noise):
                    choices = punct_ids_arr[rng.integers(0, len(punct_ids_arr), size=ids0.shape)]
                    ids0[noise] = choices[noise]

            batch = {
                "input_ids": torch.from_numpy(ids0).to(device),
                "attention_mask": torch.from_numpy(am0).to(device),
            }
            if "token_type_ids" in encodings:
                batch["token_type_ids"] = encodings["token_type_ids"][s:e].to(device)

            _, lg_raw = model(batch)
            lg = temp_scaler(lg_raw) if temp_scaler is not None else lg_raw

            conf_ks[s:e, kk] = conf_from_logits(lg).detach().cpu().numpy().astype(np.float32)
            flip_ks[s:e, kk] = (lg.argmax(dim=-1).detach().cpu().numpy().astype(np.int64) != pred0[s:e]).astype(np.float32)
            margin_ks[s:e, kk] = margin_from_logits(lg).detach().cpu().numpy().astype(np.float32)
            ent_ks[s:e, kk] = entropy_from_logits(lg).detach().cpu().numpy().astype(np.float32)

    conf_drop_max = (conf0 - np.min(conf_ks, axis=1)).astype(np.float32)
    flip_rate = np.mean(flip_ks, axis=1).astype(np.float32)
    margin_std = np.std(margin_ks, axis=1).astype(np.float32)
    ent_mean = np.mean(ent_ks, axis=1).astype(np.float32)

    return np.stack([conf_drop_max, flip_rate, margin_std, ent_mean], axis=1)


# -----------------------------
# Monitoring labels and metrics
# -----------------------------
def overconfident_error_label(err: np.ndarray, conf: np.ndarray, tau: float) -> np.ndarray:
    err = np.asarray(err).astype(int).reshape(-1)
    conf = np.asarray(conf).astype(float).reshape(-1)
    return ((err == 1) & (conf >= float(tau))).astype(int)


def _k_from_budget(budget: float, n: int) -> int:
    budget = float(np.clip(budget, 0.0, 1.0))
    return int(np.floor(budget * float(n)))


# -----------------------------
# Fairness policy selection functions
# -----------------------------
def _allocate_parity_quotas(g_h: np.ndarray, budget: float) -> Dict[int, int]:
    uniq, counts = np.unique(g_h, return_counts=True)
    n_total = int(counts.sum())
    if n_total <= 0:
        return {}
    K = _k_from_budget(budget, n_total)
    if K <= 0:
        return {int(g): 0 for g in uniq}

    exact = (float(K) * counts.astype(float) / float(n_total))
    base = np.floor(exact).astype(int)
    frac = exact - base
    base = np.clip(base, 0, counts)

    R = K - int(base.sum())
    if R > 0:
        order = np.argsort(-frac)
        for i in order[:R]:
            if base[i] < counts[i]:
                base[i] += 1
    elif R < 0:
        order = np.argsort(frac)
        need = -R
        for i in order:
            if need == 0:
                break
            if base[i] > 0:
                base[i] -= 1
                need -= 1
    return {int(g): int(k) for g, k in zip(uniq, base)}


def _select_alerts_global(r_h: np.ndarray, budget: float) -> np.ndarray:
    n = len(r_h)
    if n == 0:
        return np.zeros(0, dtype=bool)
    K = _k_from_budget(budget, n)
    alerted = np.zeros(n, dtype=bool)
    if K <= 0:
        return alerted
    order = np.argsort(-r_h)
    alerted[order[:K]] = True
    return alerted


def _select_alerts_parity(r_h: np.ndarray, g_h: np.ndarray, budget: float) -> np.ndarray:
    n = len(r_h)
    if n == 0:
        return np.zeros(0, dtype=bool)
    quotas = _allocate_parity_quotas(g_h, budget)
    m = np.zeros(n, dtype=bool)
    for gg, k in quotas.items():
        idx = np.where(g_h == gg)[0]
        if len(idx) == 0 or k <= 0:
            continue
        order = idx[np.argsort(-r_h[idx])]
        m[order[:k]] = True
    return m


def _select_alerts_bounded_topk(r_h: np.ndarray, g_h: np.ndarray, budget: float, delta: float) -> np.ndarray:
    n = len(r_h)
    if n == 0:
        return np.zeros(0, dtype=bool)

    uniq, counts = np.unique(g_h, return_counts=True)
    n_total = int(counts.sum())
    K = _k_from_budget(budget, n_total)
    if K <= 0:
        return np.zeros(n, dtype=bool)

    delta = float(max(0.0, delta))
    bmin = max(0.0, float(budget) - delta)
    bmax = min(1.0, float(budget) + delta)

    Kmin = np.floor(bmin * counts.astype(float)).astype(int)
    Kmax = np.ceil(bmax * counts.astype(float)).astype(int)
    Kmin = np.clip(Kmin, 0, counts)
    Kmax = np.clip(Kmax, Kmin, counts)

    m = np.zeros(n, dtype=bool)
    picked = {int(g): 0 for g in uniq}

    for gg, kmin in zip(uniq, Kmin):
        gg = int(gg)
        if int(kmin) <= 0:
            continue
        idx = np.where(g_h == gg)[0]
        order = idx[np.argsort(-r_h[idx])]
        take = int(min(int(kmin), len(order)))
        if take > 0:
            m[order[:take]] = True
            picked[gg] += take

    remaining = K - int(m.sum())
    if remaining > 0:
        global_order = np.argsort(-r_h)
        for i in global_order:
            if remaining == 0:
                break
            if m[i]:
                continue
            gg = int(g_h[i])
            gi = int(np.where(uniq == gg)[0][0])
            if picked[gg] < int(Kmax[gi]):
                m[i] = True
                picked[gg] += 1
                remaining -= 1

    if remaining > 0:
        global_order = np.argsort(-r_h)
        for i in global_order:
            if remaining == 0:
                break
            if m[i]:
                continue
            m[i] = True
            remaining -= 1

    return m


def _policy_alerts(r_h: np.ndarray, g_h: np.ndarray, budget: float, policy: str, delta: float = 0.0) -> np.ndarray:
    if policy == "global":
        return _select_alerts_global(r_h, budget)
    if policy == "parity":
        return _select_alerts_parity(r_h, g_h, budget)
    if policy == "bounded":
        return _select_alerts_bounded_topk(r_h, g_h, budget, delta)
    raise ValueError(f"Unknown policy {policy}")


def fit_red_like_lr(features: np.ndarray, z_target: np.ndarray, seed: int) -> LogisticRegression:
    X = features.astype(np.float32)
    y = np.asarray(z_target).astype(int).reshape(-1)
    if len(np.unique(y)) < 2:
        lr = LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=1)
        lr.fit(np.vstack([X, X]), np.array([0, 1]))
        return lr
    X_tr, X_ev, y_tr, y_ev = train_test_split(X, y, test_size=0.25, random_state=seed + 17, stratify=y)
    lr = LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=1)
    lr.fit(X_tr, y_tr)
    print(f"[RED-like LR] AUC={safe_auc(y_ev, lr.predict_proba(X_ev)[:, 1]):.4f}")
    return lr


def calibrate_isotonic(scores: np.ndarray, labels: np.ndarray) -> Optional[IsotonicRegression]:
    y = np.asarray(labels).astype(int).reshape(-1)
    s = np.asarray(scores).astype(float).reshape(-1)
    m = np.isfinite(s)
    y = y[m]
    s = s[m]
    if len(y) < 50 or len(np.unique(y)) < 2:
        return None
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(s, y)
    return iso


# -----------------------------
# Detectors
# -----------------------------
class TCPHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class KOFDetector(nn.Module):
    def __init__(self, in_dim: int, hidden: int, n_tau: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.head = nn.Linear(hidden, n_tau)

    def forward(self, x):
        return self.head(self.net(x))


def make_z_multi_tau(err: torch.Tensor, conf: torch.Tensor, tau_list: List[float]) -> torch.Tensor:
    return torch.stack([(((err > 0.5) & (conf >= float(t))).float()) for t in tau_list], dim=1)


@torch.no_grad()
def kof_head_probs(kof_logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-kof_logits))


def fit_tau_combo_lr(head_probs: np.ndarray, z_target: np.ndarray, seed: int) -> LogisticRegression:
    X = head_probs.astype(np.float32)
    y = np.asarray(z_target).astype(int).reshape(-1)
    if len(np.unique(y)) < 2:
        lr = LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=1)
        lr.fit(np.vstack([X, X]), np.array([0, 1]))
        return lr
    X_tr, X_ev, y_tr, y_ev = train_test_split(X, y, test_size=0.25, random_state=seed + 101, stratify=y)
    lr = LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=1)
    lr.fit(X_tr, y_tr)
    print(f"[TauCombo LR] AUC={safe_auc(y_ev, lr.predict_proba(X_ev)[:, 1]):.4f}")
    return lr


def train_tcp_correctness(tcp: nn.Module, feats_loader, device):
    tcp.to(device)
    opt = torch.optim.AdamW(tcp.parameters(), lr=args.LR_DETECT)
    bce = nn.BCEWithLogitsLoss()

    for ep in range(1, args.EPOCHS_DETECT + 1):
        tcp.train()
        losses = []
        for feats, y, pred in tqdm(feats_loader, desc=f"[TCP] ep {ep}/{args.EPOCHS_DETECT}", leave=False):
            feats = feats.to(device)
            y = y.to(device)
            pred = pred.to(device)
            corr = (pred == y).float()
            loss = bce(tcp(feats), corr)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        print(f"[TCP] ep={ep:02d} loss={np.mean(losses):.6f}")


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@torch.no_grad()
def _tcpm_val_auc(tcpm: nn.Module, val_feats: torch.Tensor, val_conf: np.ndarray, val_z: np.ndarray, device: torch.device) -> float:
    tcpm.eval()
    logits = tcpm(val_feats.to(device)).detach().cpu().numpy().reshape(-1)
    risk = _sigmoid_np(logits).astype(np.float32)
    if args.TCPM_APPLY_TAU_GATE_AT_INFERENCE:
        risk[val_conf < float(args.TAU_TARGET)] = -1e9
    high = (val_conf >= float(args.HIGH_CONF_REGION))
    return safe_auc(val_z[high], risk[high])


def train_tcp_monitoring_strong(
    tcpm: nn.Module,
    feats_train: torch.Tensor,
    z_train: torch.Tensor,
    device: torch.device,
    val_feats: Optional[torch.Tensor] = None,
    val_conf: Optional[np.ndarray] = None,
    val_z: Optional[np.ndarray] = None,
):
    tcpm.to(device)
    opt = torch.optim.AdamW(tcpm.parameters(), lr=args.LR_DETECT)

    if args.TCPM_USE_POS_WEIGHT:
        z_np = z_train.detach().cpu().numpy().astype(np.int64)
        pos = float(z_np.sum())
        neg = float(len(z_np) - pos)
        pw = (neg / max(pos, 1.0)) if pos > 0 else 1.0
        pw = float(np.clip(pw, 1.0, 1e4))
        pos_weight = torch.tensor([pw], dtype=torch.float32, device=device)
        crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"[TCP-M] pos={int(pos)} neg={int(neg)} pos_weight={pw:.3f}")
    else:
        crit = nn.BCEWithLogitsLoss()

    class FeatsForTCPM(torch.utils.data.Dataset):
        def __init__(self, feats: torch.Tensor, z: torch.Tensor):
            self.feats = feats
            self.z = z

        def __len__(self):
            return int(self.z.numel())

        def __getitem__(self, i):
            return self.feats[i], self.z[i]

    loader = torch.utils.data.DataLoader(FeatsForTCPM(feats_train, z_train), batch_size=1024, shuffle=True)

    best_state = None
    best_auc = -1.0
    bad = 0

    for ep in range(1, int(args.EPOCHS_TCPM) + 1):
        tcpm.train()
        losses = []
        for feats, z in tqdm(loader, desc=f"[TCP-M] ep {ep}/{args.EPOCHS_TCPM}", leave=False):
            feats = feats.to(device)
            z = z.to(device).float()
            loss = crit(tcpm(feats), z)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))

        auc_val = float("nan")
        if val_feats is not None and val_conf is not None and val_z is not None:
            auc_val = _tcpm_val_auc(tcpm, val_feats, val_conf, val_z, device)

        print(f"[TCP-M] ep={ep:02d} loss={np.mean(losses) if losses else float('nan'):.6f} val_HC_AUC={auc_val:.4f}")

        if np.isfinite(auc_val) and (auc_val - best_auc) > float(args.TCPM_EARLY_STOP_MIN_DELTA):
            best_auc = auc_val
            best_state = {k: v.detach().cpu().clone() for k, v in tcpm.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= int(args.TCPM_EARLY_STOP_PATIENCE):
                print(f"[TCP-M] Early stopping at ep={ep:02d}, best_val_HC_AUC={best_auc:.4f}")
                break

    if best_state is not None:
        tcpm.load_state_dict(best_state)
        print(f"[TCP-M] Restored best checkpoint (val_HC_AUC={best_auc:.4f})")


@torch.no_grad()
def compute_kof_pos_weight(pool_y: torch.Tensor, pool_logits_cal: torch.Tensor) -> np.ndarray:
    conf = conf_from_logits(pool_logits_cal)
    pred = pool_logits_cal.argmax(dim=-1)
    err = (pred != pool_y).float()
    z = make_z_multi_tau(err, conf, args.TAU_LIST).cpu().numpy()
    z_sum = z.sum(axis=0)
    n_sum = z.shape[0]
    pos = np.maximum(z_sum, 1.0)
    neg = np.maximum(n_sum - z_sum, 1.0)
    return (neg / pos).astype(np.float32)


def train_kof_detector(kof: nn.Module, feats_loader, device, pos_weight_vec: np.ndarray):
    kof.to(device)
    posw = torch.from_numpy(pos_weight_vec).to(device)
    bce = nn.BCEWithLogitsLoss(reduction="none", pos_weight=posw)
    opt = torch.optim.AdamW(kof.parameters(), lr=args.LR_DETECT)

    tau_arr = np.array(args.TAU_LIST, dtype=float)
    ti = int(np.argmin(np.abs(tau_arr - float(args.TAU_TARGET))))
    head_w = np.ones(len(args.TAU_LIST), dtype=np.float32)
    head_w[ti] = 2.5
    head_w_t = torch.from_numpy(head_w).to(device).view(1, -1)

    for ep in range(1, args.EPOCHS_DETECT + 1):
        kof.train()
        losses = []
        for feats, z_multi in tqdm(feats_loader, desc=f"[KOF] ep {ep}/{args.EPOCHS_DETECT}", leave=False):
            feats = feats.to(device)
            z_multi = z_multi.to(device).float()
            qlog = kof(feats)
            lv = bce(qlog, z_multi)
            loss = (lv * head_w_t).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        print(f"[KOF] ep={ep:02d} loss={np.mean(losses):.6f}")


def collect_eval_features(model, temp_scaler, encodings, y, device, batch_size):
    ds = EncDataset(encodings, y)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    feats_list, logits_list, conf_list, err_list, ent_list, marg_list, energy_list = [], [], [], [], [], [], []
    with torch.no_grad():
        for batch, yb in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            yb = yb.to(device)
            feats, _lg_raw, lg_cal, conf, marg, ent = make_feat_batch(model, batch, temp_scaler=temp_scaler)
            err = (lg_cal.argmax(dim=-1) != yb).long()

            feats_list.append(feats.cpu())
            logits_list.append(lg_cal.cpu())
            conf_list.append(conf.cpu())
            err_list.append(err.cpu())
            ent_list.append(ent.cpu())
            marg_list.append(marg.cpu())
            energy_list.append(energy_from_logits(lg_cal).cpu())

    feats = torch.cat(feats_list, dim=0)
    logits_cal = torch.cat(logits_list, dim=0)
    conf = torch.cat(conf_list, dim=0).numpy()
    err = torch.cat(err_list, dim=0).numpy().astype(int)
    ent = torch.cat(ent_list, dim=0).numpy()
    marg = torch.cat(marg_list, dim=0).numpy()
    energy = torch.cat(energy_list, dim=0).numpy()
    pred = logits_cal.argmax(dim=-1).numpy().astype(int)
    return feats, logits_cal, conf, err, pred, ent, marg, energy


# -----------------------------
# Fairness helpers
# -----------------------------
def _group_name(gv: int) -> str:
    return "male" if int(gv) == 1 else "female"


def eval_one_model_with_groups(
    seed: int,
    exp: str,
    rho: float,
    model_name: str,
    conf: np.ndarray,
    err: np.ndarray,
    group: np.ndarray,
    risk_score: np.ndarray,
    budgets: List[float],
    high_conf_region: float,
    tau_target: float,
    policy: str = "global",
    delta: float = 0.0,
):
    conf = np.asarray(conf).reshape(-1).astype(float)
    err = np.asarray(err).reshape(-1).astype(int)
    group = np.asarray(group).reshape(-1).astype(int)
    risk = np.asarray(risk_score).reshape(-1).astype(float)

    high = conf >= float(high_conf_region)
    high_tau = conf >= float(tau_target)

    hc_auc = safe_auc(overconfident_error_label(err, conf, tau_target)[high], risk[high]) if high.sum() > 0 else float("nan")
    hc_auc_tau = safe_auc(err[high_tau], risk[high_tau]) if high_tau.sum() > 0 else float("nan")

    z = overconfident_error_label(err, conf, tau_target)
    z_h = z[high]
    r_h = risk[high]
    g_h = group[high]
    conf_h = conf[high]
    err_h = err[high]

    overall_row = {
        "seed": int(seed),
        "exp": exp,
        "rho": float(rho),
        "model": model_name,
        "hc_auc": float(hc_auc),
        "hc_auc_tau": float(hc_auc_tau),
        "n_test": int(len(conf)),
        "high_conf_n": int(high.sum()),
        "high_conf_tau_n": int(high_tau.sum()),
        "err_count": int(err.sum()),
        "err_rate": float(err.mean()) if len(err) > 0 else float("nan"),
        "z_count": int(z.sum()),
        "z_rate": float(z.mean()) if len(z) > 0 else float("nan"),
        "z_count_high": int(z_h.sum()),
        "z_rate_high": float(z_h.mean()) if len(z_h) > 0 else float("nan"),
        "male_n_high": int((g_h == 1).sum()),
        "female_n_high": int((g_h == 0).sum()),
    }

    alert_rows = []
    fair_rows = []
    group_rows = []

    for b in budgets:
        alerted = _policy_alerts(r_h, g_h, b, policy=policy, delta=delta)
        n_alert = int(alerted.sum())
        prec = float(z_h[alerted].mean()) if n_alert > 0 else float("nan")
        rec = float(z_h[alerted].sum() / max(int(z_h.sum()), 1)) if int(z_h.sum()) > 0 else 0.0
        base_rate = float(z_h.mean()) if len(z_h) > 0 else float("nan")
        lift = float(prec / base_rate) if (np.isfinite(prec) and np.isfinite(base_rate) and base_rate > 0) else float("nan")

        tau_mask = conf_h >= float(tau_target)
        n_alert_tau = int((alerted & tau_mask).sum()) if n_alert > 0 else 0
        n_alert_mid = int((alerted & (~tau_mask)).sum()) if n_alert > 0 else 0
        frac_alert_tau = float(n_alert_tau / n_alert) if n_alert > 0 else float("nan")
        frac_alert_mid = float(n_alert_mid / n_alert) if n_alert > 0 else float("nan")

        grp_metrics = {}
        for gv in [0, 1]:
            gm = (g_h == gv)
            n_g = int(gm.sum())
            z_g = z_h[gm]
            alerted_g = alerted[gm]
            n_alert_g = int(alerted_g.sum())
            n_pos_g = int(z_g.sum())
            n_neg_g = int(len(z_g) - n_pos_g)

            precision_g = float(z_g[alerted_g].mean()) if n_alert_g > 0 else float("nan")
            recall_g = float(z_g[alerted_g].sum() / n_pos_g) if n_pos_g > 0 else 0.0
            alert_rate_g = float(n_alert_g / n_g) if n_g > 0 else float("nan")
            tpr_g = recall_g
            fp_g = int(((alerted_g == 1) & (z_g == 0)).sum())
            fpr_g = float(fp_g / n_neg_g) if n_neg_g > 0 else 0.0

            grp_metrics[gv] = {
                "n_group_high": n_g,
                "n_pos_z_high": n_pos_g,
                "n_neg_z_high": n_neg_g,
                "n_alert": n_alert_g,
                "alert_rate": alert_rate_g,
                "precision": precision_g,
                "recall": recall_g,
                "tpr": tpr_g,
                "fpr": fpr_g,
            }

            group_rows.append({
                "seed": int(seed),
                "exp": exp,
                "rho": float(rho),
                "model": model_name,
                "budget": float(b),
                "policy": policy,
                "delta": delta if policy == "bounded" else -1.0,
                "group": _group_name(gv),
                "n_group_high": n_g,
                "n_pos_z_high": n_pos_g,
                "n_neg_z_high": n_neg_g,
                "n_alert": n_alert_g,
                "alert_rate": alert_rate_g,
                "precision": precision_g,
                "recall": recall_g,
                "equal_opportunity_tpr": tpr_g,
                "fpr": fpr_g,
            })

        male = grp_metrics[1]
        female = grp_metrics[0]

        alert_rate_gap = float(abs(male["alert_rate"] - female["alert_rate"])) if np.isfinite(male["alert_rate"]) and np.isfinite(female["alert_rate"]) else float("nan")
        alert_rate_ratio = float(
            max(male["alert_rate"], female["alert_rate"]) / max(min(male["alert_rate"], female["alert_rate"]), 1e-12)
        ) if np.isfinite(male["alert_rate"]) and np.isfinite(female["alert_rate"]) else float("nan")
        gap_precision = float(abs(male["precision"] - female["precision"])) if np.isfinite(male["precision"]) and np.isfinite(female["precision"]) else float("nan")
        gap_recall = float(abs(male["recall"] - female["recall"])) if np.isfinite(male["recall"]) and np.isfinite(female["recall"]) else float("nan")
        gap_tpr = float(abs(male["tpr"] - female["tpr"])) if np.isfinite(male["tpr"]) and np.isfinite(female["tpr"]) else float("nan")
        gap_fpr = float(abs(male["fpr"] - female["fpr"])) if np.isfinite(male["fpr"]) and np.isfinite(female["fpr"]) else float("nan")

        fair_rows.append({
            "seed": int(seed),
            "exp": exp,
            "rho": float(rho),
            "model": model_name,
            "budget": float(b),
            "policy": policy,
            "delta": delta if policy == "bounded" else -1.0,
            "alert_rate_gap": alert_rate_gap,
            "alert_rate_ratio": alert_rate_ratio,
            "gap_precision": gap_precision,
            "gap_recall": gap_recall,
            "equal_opportunity_gap": gap_tpr,
            "equalized_odds_fpr_gap": gap_fpr,
            "male_alert_rate": male["alert_rate"],
            "female_alert_rate": female["alert_rate"],
            "male_precision": male["precision"],
            "female_precision": female["precision"],
            "male_recall": male["recall"],
            "female_recall": female["recall"],
            "male_fpr": male["fpr"],
            "female_fpr": female["fpr"],
        })

        alert_rows.append({
            "seed": int(seed),
            "exp": exp,
            "rho": float(rho),
            "model": model_name,
            "budget": float(b),
            "policy": policy,
            "delta": delta if policy == "bounded" else -1.0,
            "precision": prec,
            "recall": rec,
            "lift": lift,
            "z_rate_high": base_rate,
            "n_alert": n_alert,
            "n_high": int(len(z_h)),
            "n_alert_tau": int(n_alert_tau),
            "n_alert_mid": int(n_alert_mid),
            "frac_alert_tau": float(frac_alert_tau),
            "frac_alert_mid": float(frac_alert_mid),
            "worst_precision": float(np.nanmin([male["precision"], female["precision"]])) if np.isfinite([male["precision"], female["precision"]]).any() else float("nan"),
            "worst_recall": float(np.nanmin([male["recall"], female["recall"]])) if np.isfinite([male["recall"], female["recall"]]).any() else float("nan"),
            "n_groups_total": 2,
        })

    overall_df_row = overall_row
    return overall_df_row, alert_rows, fair_rows, group_rows

# -----------------------------
# One seed / one experiment
# -----------------------------
def run_one_seed_experiment(seed: int, cfg: ExperimentConfig, hf_endpoint: Optional[str] = None):
    set_seed(seed)
    set_determinism(args.DETERMINISTIC)
    device = get_device(args.DEVICE)

    print("\n" + "=" * 80)
    print(f"RUN seed={seed} exp={cfg.name} dataset={dataset_label()} device={device}")
    print(f"swap={cfg.use_swap_drift} perm={cfg.use_perm_drift}")
    print("=" * 80)

    splits = load_text_dataset_splits(seed, hf_endpoint=hf_endpoint)
    d_train = splits["train"]
    d_val = splits["validation"]
    d_id_test = splits["id_test"]
    d_ood_val = splits["ood_validation"]
    d_ood_test = splits["ood_test"]

    dataset_rows = []
    for sn, sp in [
        ("train", d_train),
        ("validation", d_val),
        ("id_test", d_id_test),
        ("ood_validation", d_ood_val),
        ("ood_test", d_ood_test),
    ]:
        dataset_rows.append(summarize_split(seed, cfg.name, sn, sp))

    idx = np.arange(len(d_val["y"]))
    np.random.default_rng(seed + 2017).shuffle(idx)
    n_vs = int(0.5 * len(idx))
    vs_idx = idx[:n_vs]
    vc_idx = idx[n_vs:]

    def take_split(d, ix):
        return {"text": [d["text"][i] for i in ix], "y": d["y"][ix], "g": d["g"][ix]}

    val_select = take_split(d_val, vs_idx)
    val_calib = take_split(d_val, vc_idx)

    hf_token = os.environ.get("HF_TOKEN", None)
    tokenizer = hf_from_pretrained(AutoTokenizer, args.MODEL_NAME, cache_dir=args.CACHE_DIR, token=hf_token, hf_endpoint=hf_endpoint)

    enc_tr = tokenize_texts(tokenizer, d_train["text"], max_len=args.MAX_LEN)
    enc_vs = tokenize_texts(tokenizer, val_select["text"], max_len=args.MAX_LEN)
    enc_vc = tokenize_texts(tokenizer, val_calib["text"], max_len=args.MAX_LEN)

    pin = torch.cuda.is_available()
    train_loader = torch.utils.data.DataLoader(EncDataset(enc_tr, d_train["y"]), batch_size=args.BATCH_SIZE, shuffle=True, num_workers=args.NUM_WORKERS, pin_memory=pin)
    vs_loader = torch.utils.data.DataLoader(EncDataset(enc_vs, val_select["y"]), batch_size=args.BATCH_SIZE, shuffle=False, num_workers=args.NUM_WORKERS, pin_memory=pin)
    vc_loader = torch.utils.data.DataLoader(EncDataset(enc_vc, val_calib["y"]), batch_size=args.BATCH_SIZE, shuffle=False, num_workers=args.NUM_WORKERS, pin_memory=pin)

    clf = BertBinary(args.MODEL_NAME, cache_dir=args.CACHE_DIR, token=hf_token, hf_endpoint=hf_endpoint).to(device)
    clf = train_cls_with_early_stopping(clf, train_loader, vc_loader, device)
    temp_scaler = fit_temperature_scaler(clf, vc_loader, device) if args.USE_TEMP_SCALING else None

    vs_feats, _vs_logits, vs_conf, vs_err, vs_pred, vs_ent, vs_marg, vs_energy = collect_eval_features(
        clf, temp_scaler, enc_vs, val_select["y"], device, args.BATCH_SIZE
    )
    z_vs = overconfident_error_label(vs_err, vs_conf, args.TAU_TARGET)

    red_lr = fit_red_like_lr(np.stack([vs_conf, vs_ent, vs_marg], axis=1), z_vs, seed)

    # pool for TCP/TCP-M/KOF
    pool_text = []
    pool_y_parts = []
    pool_text.extend(val_calib["text"])
    pool_y_parts.append(val_calib["y"])
    pool_text.extend(d_ood_val["text"])
    pool_y_parts.append(d_ood_val["y"])

    drift_levels = [0.5, 0.8, 1.0]
    for i, dl in enumerate(drift_levels):
        texts_d, _ = apply_combined_text_drift(
            cfg,
            d_ood_val["text"],
            prob=float(np.clip(dl, 0.0, 1.0)),
            rng=np.random.default_rng(seed + 700 + i),
        )
        pool_text.extend(texts_d)
        pool_y_parts.append(d_ood_val["y"])

        texts_id_d, _ = apply_combined_text_drift(
            cfg,
            val_calib["text"],
            prob=float(np.clip(min(1.0, args.ID_DRIFT_SCALE * dl), 0.0, 1.0)),
            rng=np.random.default_rng(seed + 800 + i),
        )
        pool_text.extend(texts_id_d)
        pool_y_parts.append(val_calib["y"])

    pool_y = np.concatenate(pool_y_parts).astype(np.int64)
    enc_pool = tokenize_texts(tokenizer, pool_text, max_len=args.MAX_LEN)
    pool_feats, pool_logits_cal, pool_conf, pool_err, pool_pred, pool_ent, pool_marg, pool_energy = collect_eval_features(
        clf, temp_scaler, enc_pool, pool_y, device, args.BATCH_SIZE
    )
    pool_y_t = torch.from_numpy(pool_y).long()
    pool_conf_t = torch.from_numpy(pool_conf).float()
    pool_err_t = torch.from_numpy(pool_err).float()
    pool_z_tau = torch.from_numpy(overconfident_error_label(pool_err, pool_conf, args.TAU_TARGET)).float()

    stab_pool = None
    stab_vs = None
    if args.USE_STABILITY:
        stab_pool = torch.from_numpy(compute_stability_features_from_encodings(
            clf, temp_scaler, tokenizer, enc_pool, pool_y, device,
            K=args.STAB_K, seed=seed,
            token_drop_prob=args.STAB_TOKEN_DROP_PROB,
            token_noise_prob=args.STAB_CHAR_NOISE_PROB,
            batch_size=args.BATCH_SIZE * max(1, args.STAB_BATCH_MULT),
        )).float()

        stab_vs = torch.from_numpy(compute_stability_features_from_encodings(
            clf, temp_scaler, tokenizer, enc_vs, val_select["y"], device,
            K=args.STAB_K, seed=seed + 1,
            token_drop_prob=args.STAB_TOKEN_DROP_PROB,
            token_noise_prob=args.STAB_CHAR_NOISE_PROB,
            batch_size=args.BATCH_SIZE * max(1, args.STAB_BATCH_MULT),
        )).float()

    # TCP
    tcp = TCPHead(in_dim=pool_feats.shape[1], hidden=args.KOF_HIDDEN).to(device)

    class _TCPDS(torch.utils.data.Dataset):
        def __init__(self, feats, y, pred):
            self.feats, self.y, self.pred = feats, y, pred

        def __len__(self):
            return int(self.y.numel())

        def __getitem__(self, i):
            return self.feats[i], self.y[i], self.pred[i]

    train_tcp_correctness(
        tcp,
        torch.utils.data.DataLoader(_TCPDS(pool_feats, pool_y_t, torch.from_numpy(pool_pred).long()), batch_size=1024, shuffle=True),
        device
    )

    # TCP-M
    if args.TCPM_TRAIN_ON_TAU_SLICE:
        tcpm_mask = (pool_conf >= float(args.TAU_TARGET))
    else:
        tcpm_mask = (pool_conf >= float(args.HIGH_CONF_REGION))

    feats_tcpm = pool_feats[tcpm_mask]
    z_tcpm = pool_z_tau[tcpm_mask]
    tcpm = TCPHead(in_dim=pool_feats.shape[1], hidden=args.KOF_HIDDEN).to(device)
    train_tcp_monitoring_strong(
        tcpm,
        feats_train=feats_tcpm,
        z_train=z_tcpm,
        device=device,
        val_feats=vs_feats,
        val_conf=vs_conf,
        val_z=z_vs,
    )

    # KOF
    tau_arr = np.array(args.TAU_LIST, dtype=float)
    ti = int(np.argmin(np.abs(tau_arr - float(args.TAU_TARGET))))
    with torch.no_grad():
        z_multi = make_z_multi_tau(pool_err_t, pool_conf_t, args.TAU_LIST).cpu()

    class _KOFDS(torch.utils.data.Dataset):
        def __init__(self, feats, zmulti):
            self.feats = feats
            self.zm = zmulti

        def __len__(self):
            return int(self.zm.shape[0])

        def __getitem__(self, i):
            return self.feats[i], self.zm[i]

    kof_nostab = KOFDetector(in_dim=pool_feats.shape[1], hidden=args.KOF_HIDDEN, n_tau=len(args.TAU_LIST)).to(device)
    posw = compute_kof_pos_weight(pool_y_t.to(device), pool_logits_cal.to(device))
    train_kof_detector(
        kof_nostab,
        torch.utils.data.DataLoader(_KOFDS(pool_feats, z_multi), batch_size=1024, shuffle=True),
        device,
        posw,
    )

    kof_stab = None
    if args.USE_STABILITY and (stab_pool is not None):
        kof_in = torch.cat([pool_feats, stab_pool], dim=1)
        kof_stab = KOFDetector(in_dim=kof_in.shape[1], hidden=args.KOF_HIDDEN, n_tau=len(args.TAU_LIST)).to(device)
        train_kof_detector(
            kof_stab,
            torch.utils.data.DataLoader(_KOFDS(kof_in, z_multi), batch_size=1024, shuffle=True),
            device,
            posw,
        )

    with torch.no_grad():
        lg0 = kof_nostab(vs_feats.to(device)).cpu().numpy()
        ph0 = kof_head_probs(lg0)
        if kof_stab is not None and stab_vs is not None:
            lgS = kof_stab(torch.cat([vs_feats, stab_vs], dim=1).to(device)).cpu().numpy()
            phS = kof_head_probs(lgS)
        else:
            phS = None

    high_vs = (vs_conf >= float(args.HIGH_CONF_REGION))

    def select_kof_variant(ph: np.ndarray, name: str):
        risk_head = ph[:, ti]
        risk_mean = ph.mean(axis=1)
        combo_lr = fit_tau_combo_lr(ph[high_vs], z_vs[high_vs], seed + (1000 if name == "nostab" else 2000))
        risk_combo = combo_lr.predict_proba(ph.astype(np.float32))[:, 1]
        best_key, best_auc = None, -1
        for k, r in {"head": risk_head, "mean": risk_mean, "combo": risk_combo}.items():
            auc = safe_auc(z_vs[high_vs], r[high_vs])
            print(f"[SelectVariant KOF-{name}] {k}: HC-AUC={auc:.4f}")
            if np.isfinite(auc) and auc > best_auc:
                best_auc, best_key = auc, k
        best_key = best_key or "head"
        iso = calibrate_isotonic({"head": risk_head, "mean": risk_mean, "combo": risk_combo}[best_key][high_vs], z_vs[high_vs])
        return best_key, combo_lr, iso

    best_nostab, combo_lr_nostab, iso_nostab = select_kof_variant(ph0, "nostab")
    if phS is not None:
        best_stab, combo_lr_stab, iso_stab = select_kof_variant(phS, "stab")
    else:
        best_stab, combo_lr_stab, iso_stab = None, None, None

    # ============================================================
    # Mod 1: Post-hoc calibration for TCP-M (kept as in original)
    # ============================================================
    with torch.no_grad():
        risk_tcpm_val = _sigmoid_np(tcpm(vs_feats.to(device)).cpu().numpy())
    g_val = val_select["g"]
    calibrators_tcpm = {}
    for gv in [0, 1]:
        mask = (g_val == gv)
        scores = risk_tcpm_val[mask]
        labels = z_vs[mask]
        if len(scores) > 50 and len(np.unique(labels)) > 1:
            iso = IsotonicRegression(out_of_bounds='clip')
            iso.fit(scores, labels)
            calibrators_tcpm[gv] = iso
        else:
            calibrators_tcpm[gv] = None

    # ============================================================
    # Mod 2: Group-specific confidence thresholds (kept as in original)
    # ============================================================
    target_high_frac = 0.10
    tau_groups = {}
    for gv in [0, 1]:
        mask = (g_val == gv)
        conf_g = vs_conf[mask]
        if len(conf_g) > 0:
            tau_groups[gv] = np.percentile(conf_g, 100 * (1 - target_high_frac))
        else:
            tau_groups[gv] = args.TAU_TARGET

    # ============================================================
    # Evaluation across rho
    # ============================================================
    overall_rows, alert_rows, fair_rows, group_rows, target_stat_rows = [], [], [], [], []

    id_text, id_y, id_g = d_id_test["text"], d_id_test["y"], d_id_test["g"]
    ood_text, ood_y, ood_g = d_ood_test["text"], d_ood_test["y"], d_ood_test["g"]
    N_TARGET = int(min(args.N_TARGET, len(id_y) + len(ood_y)))

    def sample_target_mix(rho: float, rng2: np.random.Generator):
        rho = float(np.clip(rho, 0.0, 1.0))
        n_ood = int(round(N_TARGET * rho))
        n_idd = N_TARGET - n_ood

        n_idd = min(n_idd, len(id_y))
        n_ood = min(n_ood, len(ood_y))

        idx_id = rng2.choice(len(id_y), size=n_idd, replace=False) if n_idd > 0 else np.array([], dtype=int)
        idx_oo = rng2.choice(len(ood_y), size=n_ood, replace=False) if n_ood > 0 else np.array([], dtype=int)

        tid = [id_text[i] for i in idx_id]
        yid = id_y[idx_id] if n_idd > 0 else np.array([], dtype=np.int64)
        gid = id_g[idx_id] if n_idd > 0 else np.array([], dtype=np.int64)

        too = [ood_text[i] for i in idx_oo]
        yoo = ood_y[idx_oo] if n_ood > 0 else np.array([], dtype=np.int64)
        goo = ood_g[idx_oo] if n_ood > 0 else np.array([], dtype=np.int64)

        p = p_drift_from_rho(rho)
        too_d, _ = apply_combined_text_drift(cfg, too, prob=p, rng=np.random.default_rng(seed + int(1000 * rho) + 101))

        if args.DRIFT_ON_ID_AT_HIGH_RHO:
            pid = float(np.clip(args.ID_DRIFT_SCALE * rho, 0.0, 1.0))
            tid_d, _ = apply_combined_text_drift(cfg, tid, prob=pid, rng=np.random.default_rng(seed + int(1000 * rho) + 202))
        else:
            tid_d = tid

        all_text = tid_d + too_d
        all_y = np.concatenate([yid, yoo]).astype(np.int64)
        all_g = np.concatenate([gid, goo]).astype(np.int64)

        perm = rng2.permutation(len(all_y))
        all_text = [all_text[i] for i in perm]
        return all_text, all_y[perm], all_g[perm]

    for rho in args.RHO_VALUES:
        texts_rho, y_rho, g_rho = sample_target_mix(float(rho), np.random.default_rng(seed + int(1000 * rho) + 999))
        enc_rho = tokenize_texts(tokenizer, texts_rho, max_len=args.MAX_LEN)
        feats_rho, logits_cal_rho, conf_rho, err_rho, pred_rho, ent_rho, marg_rho, energy_rho = collect_eval_features(
            clf, temp_scaler, enc_rho, y_rho, device, args.BATCH_SIZE
        )

        z_rho = overconfident_error_label(err_rho, conf_rho, args.TAU_TARGET)
        high_rho = (conf_rho >= float(args.HIGH_CONF_REGION))
        target_stat_rows.append({
            "seed": int(seed),
            "exp": cfg.name,
            "rho": float(rho),
            "n_test": int(len(y_rho)),
            "n_high": int(high_rho.sum()),
            "n_high_tau": int((conf_rho >= float(args.TAU_TARGET)).sum()),
            "n_err": int(err_rho.sum()),
            "err_rate": float(err_rho.mean()) if len(err_rho) > 0 else float("nan"),
            "n_z": int(z_rho.sum()),
            "z_rate": float(z_rho.mean()) if len(z_rho) > 0 else float("nan"),
            "n_z_high": int(z_rho[high_rho].sum()) if high_rho.sum() > 0 else 0,
            "z_rate_high": float(z_rho[high_rho].mean()) if high_rho.sum() > 0 else float("nan"),
            "n_male": int((g_rho == 1).sum()),
            "n_female": int((g_rho == 0).sum()),
            "n_male_high": int(((g_rho == 1) & high_rho).sum()),
            "n_female_high": int(((g_rho == 0) & high_rho).sum()),
            "n_z_male_high": int(((g_rho == 1) & high_rho & (z_rho == 1)).sum()),
            "n_z_female_high": int(((g_rho == 0) & high_rho & (z_rho == 1)).sum()),
        })

        # compute risks
        risk_entropy = ent_rho
        risk_energy = energy_rho
        risk_red_lr = red_lr.predict_proba(np.stack([conf_rho, ent_rho, marg_rho], axis=1).astype(np.float32))[:, 1]

        with torch.no_grad():
            risk_tcp = _sigmoid_np(tcp(feats_rho.to(device)).cpu().numpy())
            risk_tcpm = _sigmoid_np(tcpm(feats_rho.to(device)).cpu().numpy())

        risk_tcp_gate = risk_tcp.copy()
        risk_tcp_gate[conf_rho < float(args.TAU_TARGET)] = -1e9

        # ----- Apply post-hoc calibration to TCP-M risk scores -----
        risk_tcpm_cal = risk_tcpm.copy()
        risk_tcpm_gate_cal = risk_tcpm.copy()
        for gv in [0, 1]:
            mask = (g_rho == gv)
            if calibrators_tcpm.get(gv) is not None and mask.sum() > 0:
                risk_tcpm_cal[mask] = calibrators_tcpm[gv].predict(risk_tcpm[mask])
                risk_tcpm_gate_cal[mask] = calibrators_tcpm[gv].predict(risk_tcpm[mask])
        for gv in [0, 1]:
            mask = (g_rho == gv)
            tau_g = tau_groups.get(gv, args.TAU_TARGET)
            risk_tcpm_gate_cal[mask & (conf_rho < tau_g)] = -1e9

        risk_tcpm_gate = risk_tcpm.copy()
        risk_tcpm_gate[conf_rho < float(args.TAU_TARGET)] = -1e9

        with torch.no_grad():
            lg_nostab = kof_nostab(feats_rho.to(device)).cpu().numpy()
            ph_nostab = kof_head_probs(lg_nostab)

        if best_nostab == "head":
            risk_kof_nostab = ph_nostab[:, ti]
        elif best_nostab == "mean":
            risk_kof_nostab = ph_nostab.mean(axis=1)
        else:
            risk_kof_nostab = combo_lr_nostab.predict_proba(ph_nostab.astype(np.float32))[:, 1]
        if iso_nostab is not None:
            risk_kof_nostab = iso_nostab.predict(risk_kof_nostab)

        risk_kof_stab = None
        if kof_stab is not None and args.USE_STABILITY:
            stab_rho = torch.from_numpy(compute_stability_features_from_encodings(
                clf, temp_scaler, tokenizer, enc_rho, y_rho, device,
                K=args.STAB_K, seed=seed + int(1000 * rho) + 9,
                token_drop_prob=args.STAB_TOKEN_DROP_PROB,
                token_noise_prob=args.STAB_CHAR_NOISE_PROB,
                batch_size=args.BATCH_SIZE * max(1, args.STAB_BATCH_MULT),
            )).float()
            with torch.no_grad():
                lg_stab = kof_stab(torch.cat([feats_rho, stab_rho], dim=1).to(device)).cpu().numpy()
                ph_stab = kof_head_probs(lg_stab)

            if best_stab == "head":
                risk_kof_stab = ph_stab[:, ti]
            elif best_stab == "mean":
                risk_kof_stab = ph_stab.mean(axis=1)
            else:
                risk_kof_stab = combo_lr_stab.predict_proba(ph_stab.astype(np.float32))[:, 1]
            if iso_stab is not None:
                risk_kof_stab = iso_stab.predict(risk_kof_stab)

        risk_kof_stab_gate = risk_kof_stab.copy() if risk_kof_stab is not None else None
        if risk_kof_stab_gate is not None:
            risk_kof_stab_gate[conf_rho < float(args.TAU_TARGET)] = -1e9

        model2risk = {
            "entropy": risk_entropy,
            "energy": risk_energy,
            "red_lr": risk_red_lr,
            "tcp": risk_tcp,
            "tcp_gate_tau": risk_tcp_gate,
            "tcp_m": risk_tcpm_cal,
            "tcp_m_gate_tau": risk_tcpm_gate_cal,
            "kof_nostab": risk_kof_nostab,
        }
        if risk_kof_stab is not None:
            model2risk["kof_stab"] = risk_kof_stab
            model2risk["kof_stab_gate_tau"] = risk_kof_stab_gate

        # --- FAIRNESS POLICY ADDITIONS: evaluate each model under multiple policies ---
        for mname, risk in model2risk.items():
            if args.ENABLE_FAIRNESS_MITIGATION and len(args.MITIGATE_MODELS) > 0:
                run_all_policies = (mname in args.MITIGATE_MODELS)
            else:
                run_all_policies = True

            policies_to_run = ["global"]
            if run_all_policies:
                policies_to_run = args.FAIRNESS_POLICIES  # includes global, parity, bounded

            for policy in policies_to_run:
                if policy == "bounded":
                    deltas_to_use = args.FAIR_TOPK_DELTAS
                else:
                    deltas_to_use = [0.0]  # delta only used for bounded

                for delta in deltas_to_use:
                    o_row, a_rows, f_rows, g_rows = eval_one_model_with_groups(
                        seed=seed,
                        exp=cfg.name,
                        rho=float(rho),
                        model_name=mname,
                        conf=conf_rho,
                        err=err_rho,
                        group=g_rho,
                        risk_score=risk,
                        budgets=args.ALERT_BUDGETS,
                        high_conf_region=args.HIGH_CONF_REGION,
                        tau_target=args.TAU_TARGET,
                        policy=policy,
                        delta=delta,
                    )
                    if policy == "global" and delta == 0.0:
                        overall_rows.append(o_row)
                    alert_rows.extend(a_rows)
                    fair_rows.extend(f_rows)
                    group_rows.extend(g_rows)

    overall_df = pd.DataFrame(overall_rows)
    if len(overall_df):
        piv = overall_df.pivot_table(index=["seed", "rho", "exp"], columns="model", values=["hc_auc", "hc_auc_tau"],
                                     aggfunc="mean")
        piv.columns = [f"{a}_{b}" for a, b in piv.columns]
        piv = piv.reset_index()
        if "high_conf_n" in overall_df.columns:
            hc_n = overall_df.groupby(["seed", "rho", "exp"], as_index=False)["high_conf_n"].mean()
            piv = piv.merge(hc_n, on=["seed", "rho", "exp"], how="left")
        overall_df = piv

    alerts_df = pd.DataFrame(alert_rows)
    fair_df = pd.DataFrame(fair_rows)
    group_df = pd.DataFrame(group_rows)

    meta = {"seed": seed}
    meta.update(splits.get("meta", {}))

    return overall_df, alerts_df, fair_df, group_df, pd.DataFrame(dataset_rows), pd.DataFrame(target_stat_rows), meta

# -----------------------------
# Ablation summary
# -----------------------------
def build_ablation_summary(raw_overall: pd.DataFrame, raw_alerts: pd.DataFrame, budget: float = 0.05) -> pd.DataFrame:
    model_to_overall_col = {
        "tcp_gate_tau": "hc_auc_tcp_gate_tau",
        "tcp_m_gate_tau": "hc_auc_tcp_m_gate_tau",
        "kof_stab_gate_tau": "hc_auc_kof_stab_gate_tau",
        "red_lr": "hc_auc_red_lr",
    }

    rows = []
    for model, overall_col in model_to_overall_col.items():
        if overall_col not in raw_overall.columns:
            continue

        per_seed_hc = raw_overall.groupby(["exp", "seed"], as_index=False)[overall_col].mean()
        per_seed_hc.rename(columns={overall_col: "avg_hc_auc"}, inplace=True)

        sub_alert = raw_alerts[(raw_alerts["model"] == model) & (np.isclose(raw_alerts["budget"], float(budget)))].copy()
        per_seed_prec = sub_alert.groupby(["exp", "seed"], as_index=False)["precision"].mean()
        per_seed_prec.rename(columns={"precision": "avg_precision_at_budget"}, inplace=True)

        sub_rho1 = sub_alert[np.isclose(sub_alert["rho"], 1.0)].copy()
        per_seed_rho1 = sub_rho1.groupby(["exp", "seed"], as_index=False)["precision"].mean()
        per_seed_rho1.rename(columns={"precision": "precision_at_rho_1"}, inplace=True)

        merged = per_seed_hc.merge(per_seed_prec, on=["exp", "seed"], how="outer").merge(per_seed_rho1, on=["exp", "seed"], how="outer")

        for exp, sub in merged.groupby("exp"):
            def _fmt(col):
                vals = sub[col].to_numpy(dtype=float)
                vals = vals[np.isfinite(vals)]
                if len(vals) == 0:
                    return float("nan"), float("nan")
                return float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")

            hc_m, hc_s = _fmt("avg_hc_auc")
            ap_m, ap_s = _fmt("avg_precision_at_budget")
            p1_m, p1_s = _fmt("precision_at_rho_1")

            rows.append({
                "exp": exp,
                "model": model,
                "avg_hc_auc_mean": hc_m,
                "avg_hc_auc_std": hc_s,
                "avg_precision_at_budget_mean": ap_m,
                "avg_precision_at_budget_std": ap_s,
                "precision_at_rho_1_mean": p1_m,
                "precision_at_rho_1_std": p1_s,
            })

    return pd.DataFrame(rows)

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=args.dataset, choices=["civilcomments", "jigsaw"])
    parser.add_argument("--ood_dataset", type=str, default=(args.ood_dataset or ""), choices=["", "civilcomments", "jigsaw"])
    parser.add_argument("--outdir", type=str, default=args.OUTDIR)
    parser.add_argument("--cache_dir", type=str, default=args.CACHE_DIR)
    parser.add_argument("--device", type=str, default=args.DEVICE)
    parser.add_argument("--max_train", type=int, default=args.MAX_TRAIN)
    parser.add_argument("--max_val", type=int, default=args.MAX_VAL)
    parser.add_argument("--n_target", type=int, default=args.N_TARGET)
    parser.add_argument("--epochs_cls", type=int, default=args.EPOCHS_CLS)
    parser.add_argument("--epochs_detect", type=int, default=args.EPOCHS_DETECT)
    parser.add_argument("--epochs_tcpm", type=int, default=args.EPOCHS_TCPM)
    parser.add_argument("--batch_size", type=int, default=args.BATCH_SIZE)
    parser.add_argument("--max_len", type=int, default=args.MAX_LEN)
    parser.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN", "") or "")
    parser.add_argument("--hf_endpoint", type=str, default=os.environ.get("HF_ENDPOINT", ""))
    parser.add_argument("--enable_full_exp", action="store_true", default=False)

    opt, _unknown = parser.parse_known_args()

    args.dataset = opt.dataset
    args.ood_dataset = (opt.ood_dataset.strip() if isinstance(opt.ood_dataset, str) else opt.ood_dataset) or None
    args.OUTDIR = opt.outdir
    args.CACHE_DIR = opt.cache_dir
    args.DEVICE = opt.device
    args.MAX_TRAIN = opt.max_train
    args.MAX_VAL = opt.max_val
    args.N_TARGET = int(opt.n_target)
    args.EPOCHS_CLS = opt.epochs_cls
    args.EPOCHS_DETECT = opt.epochs_detect
    args.EPOCHS_TCPM = opt.epochs_tcpm
    args.BATCH_SIZE = opt.batch_size
    args.MAX_LEN = opt.max_len

    if opt.hf_token:
        os.environ["HF_TOKEN"] = opt.hf_token
    if opt.hf_endpoint:
        _set_hf_endpoint(opt.hf_endpoint)

    if opt.enable_full_exp:
        for i, cfg in enumerate(EXPERIMENTS):
            if cfg.name == "full":
                EXPERIMENTS[i] = ExperimentConfig(name="full", use_swap_drift=True, use_perm_drift=True, enabled=True)

    enabled_experiments = [cfg for cfg in EXPERIMENTS if cfg.enabled]
    if len(enabled_experiments) == 0:
        raise RuntimeError("No experiments are enabled.")

    ensure_outdirs(args.OUTDIR)
    TABLES = os.path.join(args.OUTDIR, "tables")

    print("CONFIG:", json.dumps(asdict(args), indent=2))
    print("HF_TOKEN set:", bool(os.environ.get("HF_TOKEN", "")))
    print("HF_ENDPOINT:", os.environ.get("HF_ENDPOINT", ""))
    print("Enabled experiments:", [cfg.name for cfg in enabled_experiments])

    ALL_OVERALL, ALL_ALERTS, ALL_FAIR, ALL_GROUP, ALL_DSTAT, ALL_TSTAT, ALL_META = [], [], [], [], [], [], []

    for cfg in enabled_experiments:
        for sd in args.SEEDS:
            o, a, fdf, gdf, dstat, tstat, meta = run_one_seed_experiment(sd, cfg, hf_endpoint=opt.hf_endpoint)
            ALL_OVERALL.append(o)
            ALL_ALERTS.append(a)
            ALL_FAIR.append(fdf)
            ALL_GROUP.append(gdf)
            ALL_DSTAT.append(dstat)
            ALL_TSTAT.append(tstat)
            ALL_META.append(meta)

    raw_overall = pd.concat(ALL_OVERALL, axis=0).reset_index(drop=True)
    raw_alerts = pd.concat(ALL_ALERTS, axis=0).reset_index(drop=True)
    raw_fair = pd.concat(ALL_FAIR, axis=0).reset_index(drop=True)
    raw_group = pd.concat(ALL_GROUP, axis=0).reset_index(drop=True)
    raw_dstat = pd.concat(ALL_DSTAT, axis=0).reset_index(drop=True)
    raw_tstat = pd.concat(ALL_TSTAT, axis=0).reset_index(drop=True)
    raw_meta = pd.DataFrame(ALL_META)

    raw_overall.to_csv(os.path.join(TABLES, "raw_monitor_overall.csv"), index=False)
    raw_alerts.to_csv(os.path.join(TABLES, "raw_monitor_alerts.csv"), index=False)
    raw_fair.to_csv(os.path.join(TABLES, "raw_monitor_fairness_alerts.csv"), index=False)
    raw_group.to_csv(os.path.join(TABLES, "raw_monitor_group_alerts.csv"), index=False)
    raw_dstat.to_csv(os.path.join(TABLES, "raw_dataset_statistics.csv"), index=False)
    raw_tstat.to_csv(os.path.join(TABLES, "raw_target_statistics.csv"), index=False)
    raw_meta.to_csv(os.path.join(TABLES, "run_metadata.csv"), index=False)

    overall_metric_cols = [c for c in [
        "hc_auc_tcp", "hc_auc_tau_tcp",
        "hc_auc_tcp_gate_tau", "hc_auc_tau_tcp_gate_tau",
        "hc_auc_tcp_m", "hc_auc_tau_tcp_m",
        "hc_auc_tcp_m_gate_tau", "hc_auc_tau_tcp_m_gate_tau",
        "hc_auc_kof_nostab", "hc_auc_tau_kof_nostab",
        "hc_auc_kof_stab", "hc_auc_tau_kof_stab",
        "hc_auc_kof_stab_gate_tau", "hc_auc_tau_kof_stab_gate_tau",
        "hc_auc_red_lr", "hc_auc_tau_red_lr",
        "hc_auc_entropy", "hc_auc_tau_entropy",
        "hc_auc_energy", "hc_auc_tau_energy",
        "high_conf_n",
    ] if c in raw_overall.columns]

    overall_summary = mean_std_ci_over_seeds(
        raw_overall,
        ["exp", "rho"],
        overall_metric_cols,
        seed_col="seed",
        n_boot=args.BOOTSTRAP_N,
        alpha=args.BOOTSTRAP_ALPHA,
    )
    overall_summary.to_csv(os.path.join(TABLES, "summary_monitor_overall_mean_std_ci.csv"), index=False)

    alert_metric_cols = [
        "precision", "recall", "lift", "z_rate_high",
        "worst_precision", "worst_recall",
        "n_groups_total", "n_alert", "n_high",
        "n_alert_tau", "n_alert_mid",
        "frac_alert_tau", "frac_alert_mid",
    ]
    alerts_summary = mean_std_ci_over_seeds(
        raw_alerts,
        ["exp", "rho", "model", "policy", "delta", "budget"],
        alert_metric_cols,
        seed_col="seed",
        n_boot=args.BOOTSTRAP_N,
        alpha=args.BOOTSTRAP_ALPHA,
    )
    alerts_summary.to_csv(os.path.join(TABLES, "summary_monitor_alerts_mean_std_ci.csv"), index=False)

    fair_metric_cols = [
        "alert_rate_gap", "alert_rate_ratio", "gap_precision", "gap_recall",
        "equal_opportunity_gap", "equalized_odds_fpr_gap",
    ]
    fair_summary = mean_std_ci_over_seeds(
        raw_fair,
        ["exp", "rho", "model", "policy", "delta", "budget"],
        fair_metric_cols,
        seed_col="seed",
        n_boot=args.BOOTSTRAP_N,
        alpha=args.BOOTSTRAP_ALPHA,
    )
    fair_summary.to_csv(os.path.join(TABLES, "summary_monitor_fairness_mean_std_ci.csv"), index=False)

    group_metric_cols = [
        "n_group_high", "n_pos_z_high", "n_neg_z_high",
        "n_alert", "alert_rate", "precision", "recall",
        "equal_opportunity_tpr", "fpr",
    ]
    group_summary = mean_std_ci_over_seeds(
        raw_group,
        ["exp", "rho", "model", "policy", "delta", "budget", "group"],
        group_metric_cols,
        seed_col="seed",
        n_boot=args.BOOTSTRAP_N,
        alpha=args.BOOTSTRAP_ALPHA,
    )
    group_summary.to_csv(os.path.join(TABLES, "summary_monitor_group_mean_std_ci.csv"), index=False)

    dstat_metric_cols = [
        "n", "n_pos", "n_neg", "pos_rate",
        "n_male", "n_female", "male_frac", "female_frac",
        "n_pos_male", "n_pos_female",
    ]
    dstat_summary = mean_std_ci_over_seeds(
        raw_dstat,
        ["exp", "split"],
        dstat_metric_cols,
        seed_col="seed",
        n_boot=args.BOOTSTRAP_N,
        alpha=args.BOOTSTRAP_ALPHA,
    )
    dstat_summary.to_csv(os.path.join(TABLES, "summary_dataset_statistics_mean_std_ci.csv"), index=False)

    tstat_metric_cols = [
        "n_test", "n_high", "n_high_tau", "n_err", "err_rate",
        "n_z", "z_rate", "n_z_high", "z_rate_high",
        "n_male", "n_female", "n_male_high", "n_female_high",
        "n_z_male_high", "n_z_female_high",
    ]
    tstat_summary = mean_std_ci_over_seeds(
        raw_tstat,
        ["exp", "rho"],
        tstat_metric_cols,
        seed_col="seed",
        n_boot=args.BOOTSTRAP_N,
        alpha=args.BOOTSTRAP_ALPHA,
    )
    tstat_summary.to_csv(os.path.join(TABLES, "summary_target_statistics_mean_std_ci.csv"), index=False)

    tradeoff_df = raw_alerts.merge(
        raw_fair[["seed", "exp", "rho", "model", "policy", "delta", "budget", "alert_rate_gap", "gap_recall", "equalized_odds_fpr_gap"]],
        on=["seed", "exp", "rho", "model", "policy", "delta", "budget"],
        how="left",
    )
    tradeoff_summary = mean_std_ci_over_seeds(
        tradeoff_df,
        ["exp", "rho", "model", "policy", "delta", "budget"],
        ["precision", "recall", "lift", "alert_rate_gap", "gap_recall", "equalized_odds_fpr_gap"],
        seed_col="seed",
        n_boot=args.BOOTSTRAP_N,
        alpha=args.BOOTSTRAP_ALPHA,
    )
    tradeoff_summary.to_csv(os.path.join(TABLES, "summary_monitor_tradeoff_mean_std_ci.csv"), index=False)

    ablation_summary = build_ablation_summary(raw_overall, raw_alerts, budget=float(args.ALERT_BUDGETS[0]))
    ablation_summary.to_csv(os.path.join(TABLES, "ablation_summary_budget0.05.csv"), index=False)

    report_md = os.path.join(args.OUTDIR, "report.md")
    with open(report_md, "w", encoding="utf-8") as f:
        f.write("# Text monitoring ablations\n\n")
        f.write(f"- Dataset setting: `{dataset_label()}`\n")
        f.write(f"- Enabled experiments: `{', '.join([cfg.name for cfg in enabled_experiments])}`\n")
        f.write(f"- Full experiment enabled: `{any(cfg.name == 'full' for cfg in enabled_experiments)}`\n\n")
        f.write("## Protocol\n")
        f.write(f"- Seeds: {args.SEEDS[0]}..{args.SEEDS[-1]} ({len(args.SEEDS)} seeds)\n")
        f.write(f"- Monitoring target: z = 1[err & conf >= {args.TAU_TARGET:.2f}]\n")
        f.write(f"- Monitoring slice: conf >= {args.HIGH_CONF_REGION:.2f}\n")
        f.write(f"- Budgets: {args.ALERT_BUDGETS}\n")
        f.write("- Sensitive groups: female / male only (ambiguous dropped)\n")
        f.write("- Text drift operators:\n")
        f.write("  - swap drift: adjacent token swaps\n")
        f.write("  - permutation drift: local span permutation\n")
        f.write("- Fairness policies: global, parity, bounded (if enabled)\n\n")
        f.write("## Args\n```json\n")
        f.write(json.dumps(asdict(args), indent=2))
        f.write("\n```\n\n")

        f.write("## Dataset statistics summary\n")
        f.write(dstat_summary.to_markdown(index=False))
        f.write("\n\n")

        f.write("## Target statistics summary\n")
        f.write(tstat_summary.to_markdown(index=False))
        f.write("\n\n")

        f.write("## Overall summary\n")
        f.write(overall_summary.to_markdown(index=False))
        f.write("\n\n")

        f.write("## Alert summary\n")
        f.write(alerts_summary.to_markdown(index=False))
        f.write("\n\n")

        f.write("## Fairness summary\n")
        f.write(fair_summary.to_markdown(index=False))
        f.write("\n\n")

        f.write("## Tradeoff summary\n")
        f.write(tradeoff_summary.to_markdown(index=False))
        f.write("\n\n")

        f.write("## Compact ablation summary (budget=0.05)\n")
        f.write(ablation_summary.to_markdown(index=False))
        f.write("\n")

    zip_path = args.OUTDIR.rstrip("/").rstrip("\\") + ".zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(args.OUTDIR):
            for fn in files:
                full = os.path.join(root, fn)
                rel = os.path.relpath(full, args.OUTDIR)
                z.write(full, arcname=rel)

    print("\nDONE.")
    print("OUTDIR:", args.OUTDIR)
    print("Zip:", zip_path)


if __name__ == "__main__":
    main()