"""
pipeline.py - Paper 01 (reworked, October 2026)

Does attention over clinical features add anything on a small tabular data
set once it is compared against a matched control, and do the attention
weights agree with SHAP attributions computed the same way for every model?

What this script does, in order (each stage writes CSV/JSON into results/):

  cv           10 x 5 repeated stratified CV on Cleveland (297 complete cases),
               nine models on the SAME 50 splits, preprocessing fitted inside
               each training fold. Kernel SHAP for every model on the first
               three repeats (same explainer, same background, same patients),
               attention weights for the feature-token model.
  sensitivity  The same CV with the original integer coding of nominal
               variables (all 13 features z-scored), all nine models.
  ablation     Head count x attention dropout for the feature-token model,
               inside the same 50 splits.
  cv_ext       The CV again on the reduced feature set used for external
               validation, so internal and external numbers are comparable.
  external     Train on all of Cleveland, test on the Hungarian, Switzerland
               and VA Long Beach sites (10 seeds per model).

Usage:
    python code/pipeline.py --stage all
    python code/pipeline.py --stage cv --smoke      # 1 repeat, short training
Figures are drawn separately by code/make_figures.py from results/.
"""

import argparse
import datetime
import hashlib
import json
import os
import pickle
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Configuration. Every hyperparameter is fixed here, before any run; nothing
# is tuned on the test folds. The ablation stage reports the alternatives.
# ---------------------------------------------------------------------------
SEED = 42
N_SPLITS = 5
N_REPEATS = 10
VAL_FRAC = 0.15          # inner stratified split used only for early stopping
MAX_EPOCHS = 200
PATIENCE = 20            # early stopping on validation loss, best weights restored
BATCH = 16
LR = 1e-3
SHAP_REPEATS = 3         # repeats whose test patients are explained
SHAP_BG = 25             # background rows sampled from the training fold
SHAP_NSAMPLES = 1024
SHAP_CHECK = dict(bg=50, nsamples=4096, seed_offset=7)  # budget/seed check, repeat 0
EXT_SEEDS = 10
INIT_SEEDS = 5           # extra initialisations per neural model, repeat-1 splits
N_BOOT = 1000

COLUMNS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
           "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]
FEATURES = COLUMNS[:-1]
CONTINUOUS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
ORDINAL = ["ca"]                                  # vessels coloured, 0-3
BINARY = ["sex", "fbs", "exang"]
NOMINAL = ["cp", "restecg", "slope", "thal"]
CATS = {"sex": [0, 1], "fbs": [0, 1], "exang": [0, 1],
        "cp": [1, 2, 3, 4], "restecg": [0, 1, 2], "slope": [1, 2, 3],
        "thal": [3, 6, 7]}                        # thal: thallium scan result
SITES = {"cleveland": "Cleveland", "hungarian": "Hungarian",
         "switzerland": "Switzerland", "va": "VA Long Beach"}

MODELS = ["LR", "DT", "RF", "XGB", "MLP", "ST-Attn", "ST-Affine",
          "FT-Attn", "FT-Affine"]
MODEL_NAMES = {
    "LR": "Logistic Regression",
    "DT": "Decision Tree",
    "RF": "Random Forest",
    "XGB": "XGBoost",
    "MLP": "MLP",
    "ST-Attn": "Single-token attention (T = 1)",
    "ST-Affine": "Single-token affine control",
    "FT-Attn": "Feature-token attention",
    "FT-Affine": "Feature-token affine control",
}
NEURAL = ["MLP", "ST-Attn", "ST-Affine", "FT-Attn", "FT-Affine"]
FT_DEFAULT = dict(d=32, heads=4, blocks=2, attn_dropout=0.2, ffn_dropout=0.1)

SMOKE = False

# Checkpoints: one pickle per finished fold / seed job. The folder is keyed by
# a hash of this file, so any code change starts a fresh folder and stale
# results can never be mixed with new ones. Every job is deterministic (fixed
# seeds, single-threaded torch), so a resumed run gives the same numbers as an
# uninterrupted one.
CODE_HASH = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
CACHE = BASE / ".cache" / CODE_HASH


def log(msg):
    print(f"[{datetime.datetime.now():%H:%M:%S}] {msg}", flush=True)


def _keyed(key, fn, *args):
    return key, fn(*args)


def run_jobs(tag, jobs, n_jobs, describe):
    """jobs: ordered list of (key, fn, args). Loads finished jobs from the
    checkpoint folder, runs the rest in parallel, checkpoints each result the
    moment it arrives (write to .tmp, then atomic rename), and logs progress
    with an ETA. Returns results in the order of `jobs`."""
    d = CACHE / ("smoke" if SMOKE else "full") / tag
    d.mkdir(parents=True, exist_ok=True)
    done, todo = {}, []
    for key, fn, args in jobs:
        f = d / f"{key}.pkl"
        if f.exists():
            with open(f, "rb") as fh:
                done[key] = pickle.load(fh)
        else:
            todo.append((key, fn, args))
    log(f"[{tag}] {len(jobs)} jobs: {len(done)} loaded from checkpoint, "
        f"{len(todo)} to run on {n_jobs} workers ({d})")
    if todo:
        t0 = time.time()
        gen = Parallel(n_jobs=n_jobs, return_as="generator_unordered")(
            delayed(_keyed)(key, fn, *args) for key, fn, args in todo)
        for i, (key, res) in enumerate(gen, 1):
            tmp = d / f"{key}.pkl.tmp"
            with open(tmp, "wb") as fh:
                pickle.dump(res, fh)
            os.replace(tmp, d / f"{key}.pkl")
            done[key] = res
            el = time.time() - t0
            log(f"[{tag}] {i}/{len(todo)} {key}: {describe(res)} | "
                f"elapsed {el / 60:.1f} min, ETA {el / i * (len(todo) - i) / 60:.1f} min")
    return [done[key] for key, _, _ in jobs]


def describe_fold(res):
    auc = {m["model"]: m["roc_auc"] for m in res["metrics"]}
    keys = [k for k in ("LR", "FT-Attn", "FT-Affine") if k in auc]
    txt = ", ".join(f"{k} AUC {auc[k]:.3f}" for k in keys) or \
        f"{len(auc)} variants, best AUC {max(auc.values()):.3f}"
    return f"{res['seconds']:.0f}s, {txt}"


# ===========================================================================
# 1. Data
# ===========================================================================

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_site(site):
    """Raw processed UCI file -> DataFrame. '?' and physiologically impossible
    zeros (cholesterol, resting blood pressure) become missing."""
    df = pd.read_csv(DATA / f"processed.{site}.data", names=COLUMNS,
                     na_values="?").apply(pd.to_numeric, errors="coerce")
    for col in ["chol", "trestbps"]:
        df.loc[df[col] == 0, col] = np.nan
    df["target"] = (df["num"] > 0).astype(int)
    return df


def load_cleveland():
    df = load_site("cleveland")
    return df.dropna(subset=FEATURES).reset_index(drop=True)


# ===========================================================================
# 2. Encoding (fitted on training rows only)
# ===========================================================================

class Encoder:
    """Turns raw feature rows (numpy, original UCI codes, NaN allowed) into
    model inputs.

    mode='onehot'   continuous + ordinal z-scored, binary kept 0/1, nominal
                    one-hot (main analysis). Feature tokens: continuous and
                    ordinal get a linear embedding, binary and nominal an
                    embedding table.
    mode='integer'  every feature z-scored as a number, the coding used in the
                    original submission (sensitivity analysis). Every feature
                    is a numeric token.
    """

    def __init__(self, features, mode="onehot"):
        self.features = list(features)
        self.mode = mode
        idx = {f: i for i, f in enumerate(self.features)}
        if mode == "onehot":
            self.scaled = [f for f in self.features if f in CONTINUOUS + ORDINAL]
            self.binary = [f for f in self.features if f in BINARY]
            self.nominal = [f for f in self.features if f in NOMINAL]
        else:
            self.scaled, self.binary, self.nominal = list(self.features), [], []
        self.cat = self.binary + self.nominal
        self.i_scaled = [idx[f] for f in self.scaled]
        self.i_binary = [idx[f] for f in self.binary]
        self.i_nominal = [idx[f] for f in self.nominal]
        self.i_cat = [idx[f] for f in self.cat]
        # token order: numeric tokens first, then categorical tokens
        self.token_names = self.scaled + self.cat
        self.cardinalities = [len(CATS[f]) for f in self.cat]
        # flat column -> original feature (to sum attributions back)
        self.flat_groups = (self.scaled + self.binary +
                            [f for f in self.nominal for _ in CATS[f]])
        self.flat_names = (self.scaled + self.binary +
                           [f"{f}={c}" for f in self.nominal for c in CATS[f]])

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.fill = np.zeros(X.shape[1])
        for j, f in enumerate(self.features):
            col = X[:, j][~np.isnan(X[:, j])]
            if f in CONTINUOUS + ORDINAL and self.mode == "onehot" or \
                    (self.mode == "integer" and f in CONTINUOUS + ORDINAL):
                self.fill[j] = np.median(col)
            else:
                vals, counts = np.unique(col, return_counts=True)
                self.fill[j] = vals[np.argmax(counts)]
        Xf = self._impute(X)
        self.mu = Xf[:, self.i_scaled].mean(0)
        sd = Xf[:, self.i_scaled].std(0)
        self.sd = np.where(sd > 0, sd, 1.0)
        return self

    def _impute(self, X):
        X = np.array(X, dtype=float, copy=True)
        nan = np.isnan(X)
        if nan.any():
            X[nan] = np.take(self.fill, np.where(nan)[1])
        return X

    def _cat_index(self, X):
        out = np.zeros((X.shape[0], len(self.cat)), dtype=np.int64)
        offset = 0
        for k, (j, f) in enumerate(zip(self.i_cat, self.cat)):
            levels = np.array(CATS[f], dtype=float)
            match = X[:, j][:, None] == levels[None, :]
            pos = np.where(match.any(1), match.argmax(1),
                           int(np.argmax(levels == self.fill[j])))
            out[:, k] = pos + offset
            offset += len(levels)
        return out

    def flat(self, X):
        X = self._impute(X)
        parts = [(X[:, self.i_scaled] - self.mu) / self.sd, X[:, self.i_binary]]
        for j, f in zip(self.i_nominal, self.nominal):
            levels = np.array(CATS[f], dtype=float)
            parts.append((X[:, j][:, None] == levels[None, :]).astype(float))
        return np.hstack(parts).astype(np.float32)

    def tokens(self, X):
        X = self._impute(X)
        num = ((X[:, self.i_scaled] - self.mu) / self.sd).astype(np.float32)
        return num, self._cat_index(X)


# ===========================================================================
# 3. Neural models (PyTorch)
# ===========================================================================

import torch                      # noqa: E402  (imported after numpy config)
from torch import nn              # noqa: E402


class MultiHeadAttention(nn.Module):
    """Scaled dot-product attention over T tokens of width d.

    Shapes: input H (B, T, d); Q, K, V (B, h, T, d_k) with d_k = key_dim;
    attention A = softmax(Q K^T / sqrt(d_k)) of shape (B, h, T, T), softmax
    taken over the last axis (keys); output (B, T, d) after W_o.
    With T = 1, A is identically 1 and the block reduces to W_o W_v H + b.
    """

    def __init__(self, d, heads, key_dim, dropout=0.0):
        super().__init__()
        self.h, self.dk = heads, key_dim
        self.q = nn.Linear(d, heads * key_dim)
        self.k = nn.Linear(d, heads * key_dim)
        self.v = nn.Linear(d, heads * key_dim)
        self.o = nn.Linear(heads * key_dim, d)
        self.drop = nn.Dropout(dropout)
        self.last_attn = None

    def forward(self, x):
        B, T, _ = x.shape
        def split(t):
            return t.view(B, T, self.h, self.dk).transpose(1, 2)
        q, k, v = split(self.q(x)), split(self.k(x)), split(self.v(x))
        a = torch.softmax(q @ k.transpose(-2, -1) / self.dk ** 0.5, dim=-1)
        self.last_attn = a.detach()
        out = (self.drop(a) @ v).transpose(1, 2).reshape(B, T, self.h * self.dk)
        return self.o(out)


class TokenAffine(nn.Module):
    """Control mixer: the same d -> d affine map applied to every token
    separately. No information moves between tokens."""

    def __init__(self, d):
        super().__init__()
        self.lin = nn.Linear(d, d)

    def forward(self, x):
        return self.lin(x)


class MLP(nn.Module):
    """Input (B, p) -> 128 -> 64 -> 32 -> 1 (the original ANN baseline)."""

    def __init__(self, p):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(p, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class SingleTokenNet(nn.Module):
    """The originally submitted design. The whole patient vector is projected
    to one 64-d token (T = 1), passed through a mixer with a residual
    connection and LayerNorm, then a small head.
    mixer='attention': 4-head attention, key_dim 32 (as submitted).
    mixer='affine'   : a single 64 -> 64 affine layer, the control the
                       reviewer asked for. Same residual, same LayerNorm."""

    def __init__(self, p, mixer="attention", heads=4, key_dim=32):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(p, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3))
        self.mixer = (MultiHeadAttention(64, heads, key_dim)
                      if mixer == "attention" else TokenAffine(64))
        self.norm = nn.LayerNorm(64)
        self.head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(),
                                  nn.Dropout(0.2), nn.Linear(32, 1))

    def forward(self, x):
        h = self.proj(x).unsqueeze(1)             # (B, 1, 64)
        h = self.norm(h + self.mixer(h))          # residual + LayerNorm
        return self.head(h.squeeze(1)).squeeze(-1)


class FeatureTokenNet(nn.Module):
    """Each clinical feature becomes its own d-dimensional token (F tokens).
    Numeric feature j:     t_j = x_j * w_j + b_j            (w_j, b_j in R^d)
    Categorical feature j: t_j = E_j[x_j] + b_j             (embedding table)
    Blocks (post-norm, as in the original design):
        H <- LN(H + Drop(Mixer(H)));  H <- LN(H + Drop(FFN(H)))
    Mixer = multi-head attention across the F tokens, or the token-wise affine
    control. Readout: mean over tokens -> LayerNorm -> ReLU -> linear."""

    def __init__(self, n_num, cardinalities, mixer="attention", d=32,
                 heads=4, blocks=2, attn_dropout=0.2, ffn_dropout=0.1):
        super().__init__()
        self.n_num, self.n_cat = n_num, len(cardinalities)
        self.w_num = nn.Parameter(torch.randn(n_num, d) * d ** -0.5)
        self.b_num = nn.Parameter(torch.zeros(n_num, d))
        if self.n_cat:
            self.emb = nn.Embedding(int(sum(cardinalities)), d)
            nn.init.normal_(self.emb.weight, std=d ** -0.5)
            self.b_cat = nn.Parameter(torch.zeros(self.n_cat, d))
        self.mixers = nn.ModuleList(
            [MultiHeadAttention(d, heads, d // heads, attn_dropout)
             if mixer == "attention" else TokenAffine(d) for _ in range(blocks)])
        self.norm1 = nn.ModuleList([nn.LayerNorm(d) for _ in range(blocks)])
        self.ffn = nn.ModuleList([nn.Sequential(
            nn.Linear(d, 2 * d), nn.ReLU(), nn.Dropout(ffn_dropout),
            nn.Linear(2 * d, d)) for _ in range(blocks)])
        self.norm2 = nn.ModuleList([nn.LayerNorm(d) for _ in range(blocks)])
        self.drop = nn.Dropout(ffn_dropout)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.ReLU(), nn.Linear(d, 1))

    def tokenize(self, num, cat):
        toks = [num.unsqueeze(-1) * self.w_num + self.b_num]   # (B, Fn, d)
        if self.n_cat:
            toks.append(self.emb(cat) + self.b_cat)            # (B, Fc, d)
        return torch.cat(toks, dim=1)                          # (B, F, d)

    def forward(self, num, cat):
        h = self.tokenize(num, cat)
        for mix, n1, ffn, n2 in zip(self.mixers, self.norm1, self.ffn, self.norm2):
            h = n1(h + self.drop(mix(h)))
            h = n2(h + self.drop(ffn(h)))
        return self.head(h.mean(dim=1)).squeeze(-1)

    def attention_maps(self):
        """(B, F, F) per block, averaged over heads; rows = queries."""
        return [m.last_attn.mean(1) for m in self.mixers
                if isinstance(m, MultiHeadAttention)]


def build_torch(name, enc, ft_kwargs=None):
    p = len(enc.flat_names)
    if name == "MLP":
        return MLP(p), "flat"
    if name == "ST-Attn":
        return SingleTokenNet(p, "attention"), "flat"
    if name == "ST-Affine":
        return SingleTokenNet(p, "affine"), "flat"
    kw = dict(FT_DEFAULT, **(ft_kwargs or {}))
    mixer = "attention" if name == "FT-Attn" else "affine"
    return FeatureTokenNet(len(enc.scaled), enc.cardinalities, mixer, **kw), "tokens"


def to_inputs(enc, X, kind):
    if kind == "flat":
        return (torch.from_numpy(enc.flat(X)),)
    num, cat = enc.tokens(X)
    return (torch.from_numpy(num), torch.from_numpy(cat))


def train_torch(model, kind, enc, X_fit, y_fit, X_val, y_val, seed):
    """Adam, BCE loss, batch 16, ReduceLROnPlateau(0.5, 5) and early stopping
    (patience 20) on validation loss; the best-validation-loss weights are
    restored before the model is used. `seed` drives initialisation (set in
    fit_model before the model is built), batch order and dropout."""
    torch.manual_seed(seed + 1)
    g = torch.Generator().manual_seed(seed)
    fit_in, val_in = to_inputs(enc, X_fit, kind), to_inputs(enc, X_val, kind)
    yf = torch.tensor(y_fit, dtype=torch.float32)
    yv = torch.tensor(y_val, dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, factor=0.5, patience=5, min_lr=1e-6)
    lossf = nn.BCEWithLogitsLoss()
    best, best_state, best_epoch, wait = np.inf, None, 0, 0
    hist = []
    max_epochs = 25 if SMOKE else MAX_EPOCHS
    n = len(yf)
    for epoch in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, BATCH):
            b = perm[i:i + BATCH]
            opt.zero_grad()
            loss = lossf(model(*[t[b] for t in fit_in]), yf[b])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            lf, lv = model(*fit_in), model(*val_in)
            tl, vl = lossf(lf, yf).item(), lossf(lv, yv).item()
            ta = ((lf > 0).float() == yf).float().mean().item()
            va = ((lv > 0).float() == yv).float().mean().item()
        hist.append((epoch, tl, vl, ta, va))
        sched.step(vl)
        if vl < best - 1e-6:
            best, best_epoch, wait = vl, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, hist, best_epoch


class Fitted:
    """A trained model plus its encoder; predicts from raw feature rows."""

    def __init__(self, name, enc, model, kind):
        self.name, self.enc, self.model, self.kind = name, enc, model, kind

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if self.kind == "sk":
            return self.model.predict_proba(self.enc.flat(X))[:, 1]
        with torch.no_grad():
            out = []
            for i in range(0, len(X), 8192):
                out.append(torch.sigmoid(
                    self.model(*to_inputs(self.enc, X[i:i + 8192], self.kind))))
            return torch.cat(out).numpy()


def build_sklearn(name, seed):
    import xgboost as xgb
    if name == "LR":
        return LogisticRegression(max_iter=1000)
    if name == "DT":
        return DecisionTreeClassifier(random_state=seed)
    if name == "RF":
        return RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1)
    if name == "XGB":
        return xgb.XGBClassifier(n_estimators=100, eval_metric="logloss",
                                 random_state=seed, n_jobs=1, verbosity=0)
    raise ValueError(name)


def fit_model(name, enc, X_tr, y_tr, seed, ft_kwargs=None, init_seed=None):
    """Returns (Fitted, history or None, best_epoch or None).
    `seed` fixes the inner fit/validation split (and sklearn randomness);
    `init_seed` (default = seed) fixes network initialisation, batch order and
    dropout. The torch seed is set BEFORE the network is built, so weights do
    not depend on whatever ran earlier in the same worker process."""
    if name not in NEURAL:
        m = build_sklearn(name, seed)
        m.fit(enc.flat(X_tr), y_tr)
        return Fitted(name, enc, m, "sk"), None, None
    init_seed = seed if init_seed is None else init_seed
    i_fit, i_val = inner_split(y_tr, seed)
    torch.manual_seed(init_seed)
    model, kind = build_torch(name, enc, ft_kwargs)
    model, hist, best_epoch = train_torch(model, kind, enc, X_tr[i_fit], y_tr[i_fit],
                                          X_tr[i_val], y_tr[i_val], init_seed)
    return Fitted(name, enc, model, kind), hist, best_epoch


def inner_split(y_tr, seed):
    idx = np.arange(len(y_tr))
    return train_test_split(idx, test_size=VAL_FRAC, stratify=y_tr, random_state=seed)


# ===========================================================================
# 4. Metrics and statistics
# ===========================================================================

def metrics(y, p):
    yhat = (p >= 0.5).astype(int)
    return {
        "roc_auc": roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan,
        "accuracy": accuracy_score(y, yhat),
        "sensitivity": recall_score(y, yhat, zero_division=0),
        "specificity": recall_score(1 - y, 1 - yhat, zero_division=0),
        "precision": precision_score(y, yhat, zero_division=0),
        "f1": f1_score(y, yhat, zero_division=0),
        "brier": brier_score_loss(y, p),
    }


METRICS = ["roc_auc", "accuracy", "sensitivity", "specificity", "precision",
           "f1", "brier"]


def corrected_ci(x, ratio, alpha=0.05):
    """Mean and 95% CI over J resampled folds with the Nadeau-Bengio
    correction: var_c = (1/J + n_test/n_train) * s^2."""
    x = np.asarray(x, dtype=float)
    J = len(x)
    se = np.sqrt((1 / J + ratio) * x.var(ddof=1))
    t = stats.t.ppf(1 - alpha / 2, J - 1)
    return x.mean(), x.mean() - t * se, x.mean() + t * se


def corrected_ttest(d, ratio):
    d = np.asarray(d, dtype=float)
    J = len(d)
    se = np.sqrt((1 / J + ratio) * d.var(ddof=1))
    t = d.mean() / se if se > 0 else 0.0
    return t, 2 * stats.t.sf(abs(t), J - 1)


def holm(pvals):
    p = np.asarray(pvals)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


def summarise_cv(folds, sizes):
    ratio = (sizes["n_test"] / sizes["n_train"]).mean()
    rows = []
    for m in MODELS:
        sub = folds[folds.model == m]
        if sub.empty:
            continue
        row = {"model": MODEL_NAMES[m], "key": m, "folds": len(sub)}
        for k in METRICS:
            mean, lo, hi = corrected_ci(sub[k], ratio)
            row[f"{k}_mean"] = round(mean, 4)
            row[f"{k}_sd"] = round(sub[k].std(ddof=1), 4)
            row[f"{k}_ci_low"] = round(lo, 4)
            row[f"{k}_ci_high"] = round(hi, 4)
        rows.append(row)
    return pd.DataFrame(rows)


PAIRS = ([(m, "LR") for m in MODELS if m != "LR"] +
         [("FT-Attn", "FT-Affine"), ("ST-Attn", "ST-Affine"),
          ("FT-Attn", "ST-Attn"), ("FT-Attn", "MLP"), ("FT-Attn", "RF")])


def pairwise(folds, sizes, metric="roc_auc"):
    ratio = (sizes["n_test"] / sizes["n_train"]).mean()
    wide = folds.pivot_table(index=["repeat", "fold"], columns="model", values=metric)
    rows = []
    for a, b in PAIRS:
        if a not in wide or b not in wide:
            continue
        d = (wide[a] - wide[b]).values
        mean, lo, hi = corrected_ci(d, ratio)
        t, p = corrected_ttest(d, ratio)
        rows.append({"model_a": MODEL_NAMES[a], "model_b": MODEL_NAMES[b],
                     "key_a": a, "key_b": b, "metric": metric,
                     "mean_diff": round(mean, 4), "ci_low": round(lo, 4),
                     "ci_high": round(hi, 4), "t": round(t, 3), "p": p,
                     "a_wins_folds": int((d > 0).sum()), "ties": int((d == 0).sum()),
                     "folds": len(d)})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p"].values)
    out["p"] = out["p"].round(4)
    out["p_holm"] = out["p_holm"].round(4)
    return out


# ===========================================================================
# 5. One CV fold (runs in a worker process)
# ===========================================================================

def kernel_shap(fitted, X_bg, X_explain, nsamples, seed):
    import shap
    np.random.seed(seed)
    ex = shap.KernelExplainer(fitted.predict, X_bg, link="identity")
    sv = ex.shap_values(X_explain, nsamples=nsamples, l1_reg=False, silent=True)
    return np.asarray(sv).reshape(len(X_explain), -1)


def tree_shap_rf(fitted, X_bg, X_explain):
    """Interventional TreeSHAP on the RF's one-hot inputs, summed back to the
    original features - an exact explainer to compare Kernel SHAP against."""
    import shap
    enc = fitted.enc
    ex = shap.TreeExplainer(fitted.model, data=enc.flat(X_bg),
                            feature_perturbation="interventional")
    sv = np.asarray(ex.shap_values(enc.flat(X_explain), check_additivity=False))
    if sv.ndim == 3:
        sv = sv[..., 1]
    out = np.zeros((len(X_explain), len(enc.features)))
    for c, f in enumerate(enc.flat_groups):
        out[:, enc.features.index(f)] += sv[:, c]
    return out


def run_fold(repeat, fold, tr, te, X, y, features, mode, models, do_shap,
             ft_variants=None, smoke=False):
    global SMOKE
    SMOKE = smoke
    torch.set_num_threads(1)
    t0 = time.time()
    seed = SEED + 1000 * repeat + fold
    X_tr, y_tr, X_te, y_te = X[tr], y[tr], X[te], y[te]
    enc = Encoder(features, mode).fit(X_tr)
    i_fit, i_val = inner_split(y_tr, seed)
    out = {"metrics": [], "preds": [], "hist": [], "shap": [], "attn": [],
           "attn_maps": [], "shap_check": [], "timing": []}
    out["sizes"] = {"repeat": repeat, "fold": fold, "seed": seed, "n_train": len(tr),
                    "n_fit": len(i_fit), "n_val": len(i_val), "n_test": len(te),
                    "pos_train": int(y_tr.sum()), "pos_fit": int(y_tr[i_fit].sum()),
                    "pos_val": int(y_tr[i_val].sum()), "pos_test": int(y_te.sum())}
    out["roles"] = (tr[i_fit].tolist(), tr[i_val].tolist(), te.tolist())

    variants = ft_variants or [(m, None) for m in models]
    rng = np.random.RandomState(seed)
    bg = X_tr[rng.choice(len(X_tr), SHAP_BG, replace=False)]
    for name, kw in variants:
        label = name if kw is None else json.dumps(kw, sort_keys=True)
        ts = time.time()
        fitted, hist, best_epoch = fit_model(name, enc, X_tr, y_tr, seed, kw)
        p = fitted.predict(X_te)
        row = {"repeat": repeat, "fold": fold, "model": label, **metrics(y_te, p),
               "best_epoch": best_epoch, "train_seconds": round(time.time() - ts, 2)}
        out["metrics"].append(row)
        if ft_variants:
            continue
        out["preds"] += [(repeat, fold, int(i), name, float(pi), int(yi))
                         for i, pi, yi in zip(te, p, y_te)]
        if hist:
            out["hist"] += [(repeat, fold, name, *h) for h in hist]
        if name == "FT-Attn":
            with torch.no_grad():
                fitted.model(*to_inputs(enc, X_te, "tokens"))
            maps = torch.stack(fitted.model.attention_maps()).numpy()  # (L, B, F, F)
            received = maps.mean(axis=(0, 2))        # mean over layers and queries
            for i, r in zip(te, received):
                out["attn"].append((repeat, fold, int(i), *[float(v) for v in r]))
            out["attn_maps"].append(maps.mean(axis=1))                 # (L, F, F)
            out["token_names"] = enc.token_names
        if do_shap:
            ts = time.time()
            n_exp = 4 if smoke else len(te)
            sv = kernel_shap(fitted, bg, X_te[:n_exp],
                             128 if smoke else SHAP_NSAMPLES, seed)
            for i, s in zip(te[:n_exp], sv):
                out["shap"].append((repeat, fold, int(i), name, "kernel", *s))
            if name == "RF":
                tv = tree_shap_rf(fitted, bg, X_te[:n_exp])
                for i, s in zip(te[:n_exp], tv):
                    out["shap"].append((repeat, fold, int(i), name, "tree", *s))
            if repeat == 0 and name in ("LR", "RF", "FT-Attn", "MLP"):
                r2 = np.random.RandomState(seed + SHAP_CHECK["seed_offset"])
                bg2 = X_tr[r2.choice(len(X_tr), SHAP_CHECK["bg"], replace=False)]
                sv2 = kernel_shap(fitted, bg2, X_te[:n_exp],
                                  256 if smoke else SHAP_CHECK["nsamples"],
                                  seed + SHAP_CHECK["seed_offset"])
                for i, s in zip(te[:n_exp], sv2):
                    out["shap_check"].append((repeat, fold, int(i), name, *s))
            out["timing"].append((repeat, fold, name, "shap", round(time.time() - ts, 2)))
    out["seconds"] = round(time.time() - t0, 1)
    return out


def run_cv(tag, features, mode, models, shap_repeats, n_jobs, smoke,
           ft_variants=None):
    df = load_cleveland()
    X = df[features].values.astype(float)
    y = df["target"].values
    n_rep = 1 if smoke else N_REPEATS
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=n_rep,
                                   random_state=SEED)
    jobs = []
    for k, (tr, te) in enumerate(rskf.split(X, y)):
        r, f = divmod(k, N_SPLITS)
        jobs.append((f"r{r:02d}_f{f}", run_fold,
                     (r, f, tr, te, X, y, features, mode, models,
                      r < shap_repeats, ft_variants, smoke)))
    # SHAP folds are the slowest; start them first so they do not finish last
    jobs.sort(key=lambda j: (not j[2][9], j[0]))
    log(f"[{tag}] {len(jobs)} folds x {len(ft_variants or models)} models")
    t0 = time.time()
    res = run_jobs(tag, jobs, n_jobs, describe_fold)
    res.sort(key=lambda r: (r["sizes"]["repeat"], r["sizes"]["fold"]))
    log(f"[{tag}] done in {time.time() - t0:.0f}s")

    folds = pd.DataFrame([m for r in res for m in r["metrics"]])
    sizes = pd.DataFrame([r["sizes"] for r in res])
    sfx = f"_{tag}"
    folds.round(6).to_csv(RESULTS / f"cv_folds{sfx}.csv", index=False)
    if ft_variants:
        return folds, sizes
    sizes.to_csv(RESULTS / f"fold_sizes{sfx}.csv", index=False)
    summary = summarise_cv(folds, sizes)
    summary.to_csv(RESULTS / f"cv_summary{sfx}.csv", index=False)
    pw = pd.concat([pairwise(folds, sizes, m) for m in ("roc_auc", "accuracy", "brier")])
    pw.to_csv(RESULTS / f"pairwise_tests{sfx}.csv", index=False)
    pd.DataFrame([p for r in res for p in r["preds"]],
                 columns=["repeat", "fold", "patient", "model", "prob", "y"]) \
        .round(6).to_csv(RESULTS / f"oof_predictions{sfx}.csv", index=False)
    pd.DataFrame([h for r in res for h in r["hist"]],
                 columns=["repeat", "fold", "model", "epoch", "train_loss",
                          "val_loss", "train_acc", "val_acc"]) \
        .round(5).to_csv(RESULTS / f"training_histories{sfx}.csv", index=False)
    roles = []
    for r in res:
        s = r["sizes"]
        for role, ids in zip(("fit", "val", "test"), r["roles"]):
            roles += [(s["repeat"], s["fold"], role, i) for i in ids]
    pd.DataFrame(roles, columns=["repeat", "fold", "role", "patient"]) \
        .sort_values(["repeat", "fold", "role", "patient"]) \
        .to_csv(RESULTS / f"split_indices{sfx}.csv", index=False)

    token_names = next((r["token_names"] for r in res if "token_names" in r), None)
    if token_names:
        pd.DataFrame([a for r in res for a in r["attn"]],
                     columns=["repeat", "fold", "patient"] + token_names) \
            .round(6).to_csv(RESULTS / f"attention_received{sfx}.csv", index=False)
        maps = np.stack([m for r in res for m in r["attn_maps"]])   # (folds, L, F, F)
        rows = []
        for layer in range(maps.shape[1]):
            mean = maps[:, layer].mean(0)
            for qi, q in enumerate(token_names):
                rows.append({"layer": layer + 1, "query": q,
                             **{k: round(float(mean[qi, ki]), 6)
                                for ki, k in enumerate(token_names)}})
        pd.DataFrame(rows).to_csv(RESULTS / f"attention_maps{sfx}.csv", index=False)
    shap_rows = [s for r in res for s in r["shap"]]
    if shap_rows:
        pd.DataFrame(shap_rows, columns=["repeat", "fold", "patient", "model",
                                         "explainer"] + features) \
            .round(6).to_csv(RESULTS / f"shap_values{sfx}.csv", index=False)
        pd.DataFrame([s for r in res for s in r["shap_check"]],
                     columns=["repeat", "fold", "patient", "model"] + features) \
            .round(6).to_csv(RESULTS / f"shap_values_check{sfx}.csv", index=False)
        pd.DataFrame([t for r in res for t in r["timing"]],
                     columns=["repeat", "fold", "model", "what", "seconds"]) \
            .to_csv(RESULTS / f"shap_timing{sfx}.csv", index=False)
    print(summary[["model", "roc_auc_mean", "roc_auc_sd", "accuracy_mean"]]
          .to_string(index=False), flush=True)
    return folds, sizes


# ===========================================================================
# 6. Attribution analysis (no training; reads the CV outputs)
# ===========================================================================

def importance(sv_df, features):
    return sv_df[features].abs().mean()


def spearman_boot(a_df, b_df, features, rng, n_boot=N_BOOT):
    """Spearman rho between two importance vectors, with a patient bootstrap.
    a_df / b_df: per-(repeat, patient) attributions for the same patients."""
    rho = stats.spearmanr(importance(a_df, features), importance(b_df, features))[0]
    pats = a_df["patient"].unique()
    ga, gb = a_df.groupby("patient"), b_df.groupby("patient")
    A = {p: g[features].abs().values for p, g in ga}
    B = {p: g[features].abs().values for p, g in gb}
    boots = []
    for _ in range(n_boot):
        s = rng.choice(pats, len(pats), replace=True)
        ia = np.vstack([A[p] for p in s]).mean(0)
        ib = np.vstack([B[p] for p in s]).mean(0)
        boots.append(stats.spearmanr(ia, ib)[0])
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return rho, lo, hi


def analyse_attributions(tag="onehot"):
    rng = np.random.RandomState(SEED)
    sv = pd.read_csv(RESULTS / f"shap_values_{tag}.csv")
    feats = FEATURES
    kern = sv[sv.explainer == "kernel"]
    imp = {m: importance(kern[kern.model == m], feats) for m in MODELS
           if m in kern.model.unique()}
    imp_df = pd.DataFrame(imp).T
    imp_df.index = [MODEL_NAMES[m] for m in imp_df.index]
    imp_df.round(6).to_csv(RESULTS / "shap_importance.csv", index_label="model")
    ranks = imp_df.rank(axis=1, ascending=False, method="min").astype(int)
    ranks.to_csv(RESULTS / "shap_ranks.csv", index_label="model")

    rows = []
    present = [m for m in MODELS if m in imp]
    for i, a in enumerate(present):
        for b in present[i + 1:]:
            r, lo, hi = spearman_boot(kern[kern.model == a], kern[kern.model == b],
                                      feats, rng)
            rows.append({"model_a": MODEL_NAMES[a], "model_b": MODEL_NAMES[b],
                         "key_a": a, "key_b": b, "rho": round(r, 4),
                         "ci_low": round(lo, 4), "ci_high": round(hi, 4)})
    pd.DataFrame(rows).to_csv(RESULTS / "shap_rank_agreement.csv", index=False)

    # Same model, different repeat: how stable is each model's ranking?
    stab = []
    for m in present:
        sub = kern[kern.model == m]
        reps = sorted(sub.repeat.unique())
        rs = [stats.spearmanr(importance(sub[sub.repeat == a], feats),
                              importance(sub[sub.repeat == b], feats))[0]
              for i, a in enumerate(reps) for b in reps[i + 1:]]
        if not rs:
            continue
        stab.append({"model": MODEL_NAMES[m], "key": m,
                     "rho_between_repeats_mean": round(float(np.mean(rs)), 4),
                     "rho_between_repeats_min": round(float(np.min(rs)), 4)})
    pd.DataFrame(stab).to_csv(RESULTS / "shap_repeat_stability.csv", index=False)

    # Explainer check on the same RF: Kernel vs exact interventional TreeSHAP
    tree = sv[(sv.explainer == "tree") & (sv.model == "RF")]
    kr = kern[kern.model == "RF"]
    chk = []
    if not tree.empty:
        r, lo, hi = spearman_boot(kr, tree, feats, rng)
        m = kr.merge(tree, on=["repeat", "fold", "patient"], suffixes=("_k", "_t"))
        per_val = np.corrcoef(m[[f + "_k" for f in feats]].values.ravel(),
                              m[[f + "_t" for f in feats]].values.ravel())[0, 1]
        chk.append({"comparison": "RF: Kernel SHAP vs interventional TreeSHAP",
                    "rho_importance": round(r, 4), "ci_low": round(lo, 4),
                    "ci_high": round(hi, 4),
                    "pearson_values": round(per_val, 4)})
    # Budget / background / seed check (repeat 0)
    chkdf = pd.read_csv(RESULTS / f"shap_values_check_{tag}.csv")
    base0 = kern[kern.repeat == 0]
    for m in chkdf.model.unique():
        a = base0[base0.model == m]
        b = chkdf[chkdf.model == m]
        if a.empty:
            continue
        r, lo, hi = spearman_boot(a, b, feats, rng)
        chk.append({"comparison": f"{MODEL_NAMES[m]}: 25 bg/1024 samples vs "
                                  f"50 bg/4096 samples, new seed",
                    "rho_importance": round(r, 4), "ci_low": round(lo, 4),
                    "ci_high": round(hi, 4), "pearson_values": np.nan})
    if {"RF", "FT-Attn"} <= set(chkdf.model.unique()):
        r, lo, hi = spearman_boot(chkdf[chkdf.model == "RF"],
                                  chkdf[chkdf.model == "FT-Attn"], feats, rng)
        chk.append({"comparison": "RF vs Feature-token attention under the larger "
                                  "budget (repeat 0)",
                    "rho_importance": round(r, 4), "ci_low": round(lo, 4),
                    "ci_high": round(hi, 4), "pearson_values": np.nan})
        r, lo, hi = spearman_boot(base0[base0.model == "RF"],
                                  base0[base0.model == "FT-Attn"], feats, rng)
        chk.append({"comparison": "RF vs Feature-token attention under the main "
                                  "budget (repeat 0)",
                    "rho_importance": round(r, 4), "ci_low": round(lo, 4),
                    "ci_high": round(hi, 4), "pearson_values": np.nan})
    pd.DataFrame(chk).to_csv(RESULTS / "shap_explainer_checks.csv", index=False)

    # Attention received vs SHAP for the feature-token model, same patients
    att = pd.read_csv(RESULTS / f"attention_received_{tag}.csv")
    att = att[att.repeat < SHAP_REPEATS]
    ft = kern[kern.model == "FT-Attn"]
    m = ft.merge(att, on=["repeat", "fold", "patient"], suffixes=("", "_attn"))
    shap_imp = m[feats].abs().mean()
    attn_imp = m[[f + "_attn" if f + "_attn" in m else f for f in feats]].mean()
    attn_imp.index = feats
    rho = stats.spearmanr(shap_imp, attn_imp)[0]
    pats = m.patient.unique()
    g = {p: grp for p, grp in m.groupby("patient")}
    boots = []
    for _ in range(N_BOOT):
        s = rng.choice(pats, len(pats), replace=True)
        bb = pd.concat([g[p] for p in s])
        boots.append(stats.spearmanr(
            bb[feats].abs().mean(),
            bb[[f + "_attn" for f in feats]].mean().values)[0])
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    # Per-patient agreement: rank correlation between a patient's attention
    # vector and |SHAP| vector
    per = [stats.spearmanr(r[feats].abs().values.astype(float),
                           r[[f + "_attn" for f in feats]].values.astype(float))[0]
           for _, r in m.iterrows()]
    tab = pd.DataFrame({"feature": feats,
                        "mean_abs_shap": shap_imp.values.round(6),
                        "mean_attention_received": attn_imp.values.round(6),
                        "shap_rank": shap_imp.rank(ascending=False, method="min")
                        .astype(int).values,
                        "attention_rank": attn_imp.rank(ascending=False, method="min")
                        .astype(int).values})
    tab.to_csv(RESULTS / "attention_vs_shap.csv", index=False)
    pd.DataFrame([{"rho_global": round(rho, 4), "ci_low": round(lo, 4),
                   "ci_high": round(hi, 4),
                   "rho_per_patient_median": round(float(np.nanmedian(per)), 4),
                   "rho_per_patient_q1": round(float(np.nanpercentile(per, 25)), 4),
                   "rho_per_patient_q3": round(float(np.nanpercentile(per, 75)), 4),
                   "patients": len(pats), "explanations": len(m),
                   "uniform_attention": round(1 / len(feats), 6)}]) \
        .to_csv(RESULTS / "attention_vs_shap_summary.csv", index=False)
    print("[attributions] written", flush=True)


# ===========================================================================
# 7. External validation
# ===========================================================================

def run_seed_fold(fold, tr, te, X, y, smoke):
    """Repeat-1 split `fold`: every neural model retrained with INIT_SEEDS
    different initialisation seeds, split and inner validation set fixed.
    Separates initialisation noise from split-to-split noise."""
    global SMOKE
    SMOKE = smoke
    torch.set_num_threads(1)
    t0 = time.time()
    seed = SEED + fold                      # same as repeat 0 in run_fold
    enc = Encoder(FEATURES, "onehot").fit(X[tr])
    rows = []
    for s in range(2 if smoke else INIT_SEEDS):
        for name in NEURAL:
            fitted, _, best_epoch = fit_model(name, enc, X[tr], y[tr], seed,
                                              init_seed=seed + 100003 * (s + 1))
            rows.append({"fold": fold, "init": s, "model": name,
                         **metrics(y[te], fitted.predict(X[te])),
                         "best_epoch": best_epoch})
    return {"metrics": rows, "seconds": round(time.time() - t0, 1)}


def run_seeds(n_jobs, smoke):
    df = load_cleveland()
    X, y = df[FEATURES].values.astype(float), df["target"].values
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=1, random_state=SEED)
    jobs = [(f"f{f}", run_seed_fold, (f, tr, te, X, y, smoke))
            for f, (tr, te) in enumerate(rskf.split(X, y))]
    res = run_jobs("seeds", jobs, n_jobs,
                   lambda r: f"{r['seconds']:.0f}s, {len(r['metrics'])} trainings")
    rows = pd.DataFrame([m for r in res for m in r["metrics"]])
    rows.round(6).to_csv(RESULTS / "nn_seed_variability.csv", index=False)
    out = []
    for m in NEURAL:
        g = rows[rows.model == m]
        within = g.groupby("fold")["roc_auc"].std(ddof=1)
        out.append({"model": MODEL_NAMES[m], "key": m, "inits": g.init.nunique(),
                    "folds": g.fold.nunique(),
                    "auc_sd_within_fold_mean": round(within.mean(), 4),
                    "auc_sd_within_fold_max": round(within.max(), 4),
                    "auc_range_within_fold_max": round(
                        g.groupby("fold")["roc_auc"].agg(np.ptp).max(), 4),
                    "accuracy_sd_within_fold_mean": round(
                        g.groupby("fold")["accuracy"].std(ddof=1).mean(), 4)})
    pd.DataFrame(out).to_csv(RESULTS / "nn_seed_variability_summary.csv", index=False)
    log("[seeds] written")


def external_features(max_missing=0.5):
    """Features whose missing share is <= max_missing at every external site
    (after impossible zeros are set to missing)."""
    rows, keep = [], []
    for s in ["hungarian", "switzerland", "va"]:
        d = load_site(s)
        rows.append({"site": SITES[s], "n": len(d), "positive": int(d.target.sum()),
                     **{f: round(float(d[f].isna().mean()), 3) for f in FEATURES}})
    miss = pd.DataFrame(rows)
    miss.to_csv(RESULTS / "external_missingness.csv", index=False)
    keep = [f for f in FEATURES if (miss[f] <= max_missing).all()]
    return keep


def run_external_seed(seed_i, features, tag, smoke):
    global SMOKE
    SMOKE = smoke
    torch.set_num_threads(1)
    seed = SEED + 7919 * (seed_i + 1)
    cle = load_cleveland()
    X, y = cle[features].values.astype(float), cle["target"].values
    enc = Encoder(features, "onehot").fit(X)
    sites = {s: load_site(s) for s in ["hungarian", "switzerland", "va"]}
    rows, preds = [], []
    for name in MODELS:
        fitted, _, best_epoch = fit_model(name, enc, X, y, seed)
        for s, d in sites.items():
            p = fitted.predict(d[features].values.astype(float))
            yt = d["target"].values
            rows.append({"feature_set": tag, "seed": seed_i, "model": name,
                         "site": SITES[s], **metrics(yt, p)})
            preds += [(tag, seed_i, name, SITES[s], i, float(pi), int(yi))
                      for i, (pi, yi) in enumerate(zip(p, yt))]
    return rows, preds


def run_external(n_jobs, smoke):
    keep = external_features()
    sets = {"ext": keep,
            "ext_plus": keep + [f for f in ["chol", "fbs"] if f not in keep]}
    json.dump(sets, open(RESULTS / "external_feature_sets.json", "w"), indent=2)
    n_seeds = 2 if smoke else EXT_SEEDS
    jobs = [(f"{tag}_seed{i:02d}", run_external_seed, (i, f, tag, smoke))
            for tag, f in sets.items() for i in range(n_seeds)]

    def describe(res):
        rr = [r for r in res[0] if r["model"] in ("LR", "FT-Attn")]
        return ", ".join(f"{r['model']}@{r['site'].split()[0]} {r['roc_auc']:.3f}"
                         for r in rr)
    res = run_jobs("external", jobs, n_jobs, describe)
    rows = pd.DataFrame([r for rr, _ in res for r in rr])
    preds = pd.DataFrame([p for _, pp in res for p in pp],
                         columns=["feature_set", "seed", "model", "site", "row",
                                  "prob", "y"])
    rows.round(6).to_csv(RESULTS / "external_by_seed.csv", index=False)
    preds.round(6).to_csv(RESULTS / "external_predictions.csv", index=False)

    # Seed-mean metrics, with a patient bootstrap CI on the seed-mean AUC
    rng = np.random.RandomState(SEED)
    summ = []
    for (tag, site, m), g in preds.groupby(["feature_set", "site", "model"]):
        wide = g.pivot_table(index="row", columns="seed", values="prob")
        yt = g.groupby("row")["y"].first().loc[wide.index].values
        P = wide.values
        aucs = [roc_auc_score(yt, P[:, k]) for k in range(P.shape[1])]
        boots = []
        n = len(yt)
        for _ in range(N_BOOT):
            s = rng.randint(0, n, n)
            if len(np.unique(yt[s])) < 2:
                continue
            boots.append(np.mean([roc_auc_score(yt[s], P[s, k])
                                  for k in range(P.shape[1])]))
        sub = rows[(rows.feature_set == tag) & (rows.site == site) & (rows.model == m)]
        rec = {"feature_set": tag, "site": site, "model": MODEL_NAMES[m], "key": m,
               "n": n, "positive": int(yt.sum()),
               "roc_auc_mean": round(float(np.mean(aucs)), 4),
               "roc_auc_sd_seeds": round(float(np.std(aucs, ddof=1)), 4),
               "roc_auc_ci_low": round(float(np.percentile(boots, 2.5)), 4),
               "roc_auc_ci_high": round(float(np.percentile(boots, 97.5)), 4)}
        for k in METRICS[1:]:
            rec[f"{k}_mean"] = round(float(sub[k].mean()), 4)
        summ.append(rec)
    pd.DataFrame(summ).to_csv(RESULTS / "external_summary.csv", index=False)
    print(pd.DataFrame(summ)[["feature_set", "site", "key", "roc_auc_mean",
                              "roc_auc_ci_low", "roc_auc_ci_high"]]
          .to_string(index=False), flush=True)
    return sets


# ===========================================================================
# 8. Bookkeeping
# ===========================================================================

def write_dataset_info():
    cle_raw = load_site("cleveland")
    cle = load_cleveland()
    info = {"sites": {}, "analysis_set": {
        "site": "Cleveland", "rows_raw": int(len(cle_raw)),
        "rows_dropped_missing": int(len(cle_raw) - len(cle)),
        "rows_used": int(len(cle)), "positive": int(cle.target.sum()),
        "negative": int((1 - cle.target).sum()),
        "target": "num > 0 (angiographic narrowing > 50% in any major vessel)"}}
    for s, label in SITES.items():
        d = load_site(s)
        info["sites"][label] = {
            "file": f"data/processed.{s}.data", "sha256": sha256(DATA / f"processed.{s}.data"),
            "rows": int(len(d)), "positive": int(d.target.sum()),
            "source": "https://archive.ics.uci.edu/ml/machine-learning-databases/"
                      f"heart-disease/processed.{s}.data"}
    info["variable_definitions"] = {
        "file": "data/heart-disease.names",
        "sha256": sha256(DATA / "heart-disease.names"),
        "thal": "thallium stress test: 3 = normal, 6 = fixed defect, 7 = reversible defect",
        "dataset_doi": "10.24432/C52P4X", "licence": "CC BY 4.0"}
    json.dump(info, open(RESULTS / "dataset_info.json", "w"), indent=2)


def write_param_counts():
    cle = load_cleveland()
    rows = []
    for mode in ("onehot", "integer"):
        enc = Encoder(FEATURES, mode).fit(cle[FEATURES].values.astype(float))
        for name in NEURAL:
            model, kind = build_torch(name, enc)
            n = sum(p.numel() for p in model.parameters())
            shape = (f"(B, {len(enc.flat_names)})" if kind == "flat" else
                     f"(B, {len(enc.token_names)}, {FT_DEFAULT['d']}) tokens")
            if name.startswith("ST"):
                shape += " -> (B, 1, 64), T = 1"
            rows.append({"encoding": mode, "model": MODEL_NAMES[name], "key": name,
                         "parameters": n, "input_shape": shape})
    pd.DataFrame(rows).to_csv(RESULTS / "param_counts.csv", index=False)


def write_run_info(stages, seconds):
    import shap, sklearn, xgboost
    path = RESULTS / "run_info.json"
    info = json.load(open(path)) if path.exists() else {}
    info.update({
        "finished_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "python": sys.version.split()[0], "platform": platform.platform(),
        "torch": torch.__version__, "sklearn": sklearn.__version__,
        "shap": shap.__version__, "xgboost": xgboost.__version__,
        "numpy": np.__version__, "pandas": pd.__version__,
        "config": {"seed": SEED, "n_splits": N_SPLITS, "n_repeats": N_REPEATS,
                   "val_frac": VAL_FRAC, "max_epochs": MAX_EPOCHS,
                   "patience": PATIENCE, "batch": BATCH, "lr": LR,
                   "shap_repeats": SHAP_REPEATS, "shap_background": SHAP_BG,
                   "shap_nsamples": SHAP_NSAMPLES, "shap_check": SHAP_CHECK,
                   "ft_default": FT_DEFAULT, "external_seeds": EXT_SEEDS,
                   "init_seeds": INIT_SEEDS, "bootstrap": N_BOOT},
        "code_sha256_12": CODE_HASH,
    })
    info.setdefault("stage_seconds", {}).update(seconds)
    info["smoke"] = SMOKE
    json.dump(info, open(path, "w"), indent=2)


def main():
    global SMOKE
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["all", "cv", "attrib", "sensitivity", "ablation",
                             "cv_ext", "external", "seeds"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--log", help="also append all output to this file")
    a = ap.parse_args()
    SMOKE = a.smoke
    if a.log:
        class Tee:
            def __init__(self, *streams):
                self.streams = streams

            def write(self, s):
                for st in self.streams:
                    st.write(s)
                    st.flush()

            def flush(self):
                for st in self.streams:
                    st.flush()
        Path(a.log).parent.mkdir(parents=True, exist_ok=True)
        fh = open(a.log, "a", encoding="utf-8")
        sys.stdout = Tee(sys.__stdout__, fh)
        sys.stderr = Tee(sys.__stderr__, fh)
        log(f"== run start: stages={a.stage} smoke={a.smoke} jobs={a.jobs} "
            f"code={CODE_HASH}")
    stages = (["cv", "attrib", "seeds", "sensitivity", "ablation", "cv_ext",
               "external"]
              if a.stage == "all" else [a.stage])
    write_dataset_info()
    write_param_counts()
    secs = {}
    for st in stages:
        t0 = time.time()
        if st == "cv":
            run_cv("onehot", FEATURES, "onehot", MODELS,
                   1 if a.smoke else SHAP_REPEATS, a.jobs, a.smoke)
        elif st == "attrib":
            analyse_attributions("onehot")
        elif st == "sensitivity":
            run_cv("integer", FEATURES, "integer", MODELS, 0, a.jobs, a.smoke)
        elif st == "ablation":
            variants = [("FT-Attn", {"heads": h, "attn_dropout": dr})
                        for h in (1, 2, 4, 8) for dr in (0.1, 0.2, 0.3)]
            folds, sizes = run_cv("ablation", FEATURES, "onehot", ["FT-Attn"], 0,
                                  a.jobs, a.smoke, ft_variants=variants)
            ratio = (sizes["n_test"] / sizes["n_train"]).mean()
            rows = []
            for lab, g in folds.groupby("model"):
                kw = json.loads(lab)
                mean, lo, hi = corrected_ci(g["roc_auc"], ratio)
                rows.append({"heads": kw["heads"], "attn_dropout": kw["attn_dropout"],
                             "roc_auc_mean": round(mean, 4),
                             "roc_auc_sd": round(g["roc_auc"].std(ddof=1), 4),
                             "roc_auc_ci_low": round(lo, 4),
                             "roc_auc_ci_high": round(hi, 4),
                             "accuracy_mean": round(g["accuracy"].mean(), 4),
                             "accuracy_sd": round(g["accuracy"].std(ddof=1), 4),
                             "folds": len(g)})
            pd.DataFrame(rows).sort_values(["heads", "attn_dropout"]) \
                .to_csv(RESULTS / "ablation_summary.csv", index=False)
        elif st == "cv_ext":
            run_cv("ext", external_features(), "onehot", MODELS, 0, a.jobs, a.smoke)
        elif st == "external":
            run_external(a.jobs, a.smoke)
        elif st == "seeds":
            run_seeds(a.jobs, a.smoke)
        secs[st] = round(time.time() - t0, 1)
        write_run_info(stages, secs)
        log(f"== stage {st} finished: {secs[st]}s")
    log("== ALL REQUESTED STAGES FINISHED")


if __name__ == "__main__":
    main()
