"""
AMPC – Attention-based Molecular Property Prediction
=====================================================
Reproduces the downstream experiments reported in the paper.

Tasks
-----
1. BACE binary classification  – active/inactive against BACE-1 protease.
2. LogP regression             – lipophilicity on the same BACE molecule set.

Both tasks apply a scaffold split (80/10/10) via DeepChem's ScaffoldSplitter,
so that training, validation, and test sets contain structurally distinct
scaffolds.

Representations compared per task
-----------------------------------
1. AMPC heads           – task-specific chemically motivated attention heads
                          identified by the AMPC analysis; ChemBERTa encoder
                          is kept fully frozen.
2. random_no_AMPC heads – same number of heads as the AMPC set, sampled
                          uniformly at random from heads NOT in the AMPC set;
                          encoder is kept frozen; results are reported as
                          mean ± std over N_RANDOM_SEEDS independent runs.
3. frozen mean pooling  – mean pooling over the final hidden state of the
                          frozen ChemBERTa encoder (no head selection).
4. fine-tuned CLS token – the [CLS] token embedding after end-to-end
                          fine-tuning of the full ChemBERTa encoder.

AMPC head sets (layer_index, head_index)
-----------------------------------------
BACE_HEADS  = [(0, 1), (0, 2), (0, 11), (1, 11), (2, 0),
               (2, 2), (2, 3), (2, 8),  (2, 9),  (2, 10)]

LogP_HEADS  = [(0, 2), (1, 11), (2, 2), (2, 3), (2, 8), (2, 9)]

Data
----
Place bace.csv (DeepChem BACE dataset) in data/raw/ and run:
    python ampc_experiments.py

The BACE CSV must contain the columns:
    mol   – SMILES strings
    Class – binary activity label (0 / 1)   [BACE classification]
    LogP  – measured logP value             [LogP regression]
"""

# ============================================================
# Task-specific AMPC head sets
# (layer_index, head_index), 0-indexed, for ChemBERTa-77M-MLM
# (3 transformer layers × 12 heads each)
# ============================================================

# AMPC-selected heads for BACE binary classification
BACE_HEADS = [
    (0, 1),   # ring endpoint, double bond, triple bond
    (0, 2),   # double bond, functional group
    (0, 11),  # ring endpoint
    (1, 11),  # triple bond, columnar matrix, functional group
    (2, 0),   # ring atom
    (2, 2),   # functional group
    (2, 3),   # functional group
    (2, 8),   # chirality, functional group
    (2, 9),   # functional group
    (2, 10),  # ring endpoint
]

# AMPC-selected heads for LogP regression
LogP_HEADS = [
    (0, 2),   # double bond, functional group
    (1, 11),  # triple bond, columnar matrix, functional group
    (2, 2),   # functional group
    (2, 3),   # functional group
    (2, 8),   # chirality, functional group
    (2, 9),   # functional group
]

# Maps each task name to its corresponding AMPC head set
TASK_AMPC_HEADS = {
    "bace":      BACE_HEADS,
    "logp": LogP_HEADS,
}

# ============================================================
# Imports
# ============================================================

import copy
import os
import random
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from deepchem.data import DiskDataset
from deepchem.splits import ScaffoldSplitter
from rdkit import Chem
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

warnings.filterwarnings("ignore")

# ============================================================
# Global configuration
# ============================================================

MODEL_NAME   = "DeepChem/ChemBERTa-77M-MLM"
MAX_LENGTH   = 128
BATCH_SIZE   = 32
EPOCHS       = 200
PATIENCE     = 5
LR           = 2e-5
WEIGHT_DECAY = 0.01
GRAD_CLIP    = 1.0
SEED         = 42

# Number of independent random seeds used to compute mean ± std for the
# random_no_AMPC baseline.
N_RANDOM_SEEDS = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int = 42) -> None:
    """Set all random seeds for fully reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


set_seed(SEED)

# ============================================================
# Dataset configuration
# ============================================================
# Both tasks read from the same bace.csv using different target columns.

DATASET_CONFIGS: Dict[str, dict] = {
    "bace": {
        "csv_filename":  "bace.csv",
        "smiles_col":    "mol",
        "target_col":    "Class",
        "is_regression": False,
        "oversample":    True,   # training split is class-imbalanced
    },
    "logp": {
        "csv_filename":  "bace.csv",  # same source file as the BACE task
        "smiles_col":    "mol",
        "target_col":    "LogP",
        "is_regression": True,
        "oversample":    False,
    },
}

# ============================================================
# Data loading
# ============================================================

def _canonicalize_joint(
    smiles_list: List[str],
    labels:      List,
) -> Tuple[List[str], List]:
    """
    Canonicalize SMILES with RDKit, dropping invalid entries.

    Labels are filtered in lockstep to prevent index misalignment.
    Dropping invalid SMILES without also dropping their corresponding labels
    would silently corrupt (smiles[i], label[i]) pairings downstream.
    """
    canonical_smiles: List[str] = []
    canonical_labels: List      = []
    n_invalid = 0

    for smi, lbl in zip(smiles_list, labels):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            n_invalid += 1
            continue
        canonical_smiles.append(
            Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
        )
        canonical_labels.append(lbl)

    if n_invalid:
        print(f"  Dropped {n_invalid} invalid SMILES.")

    return canonical_smiles, canonical_labels


def load_and_split_dataset(
    csv_path:        str,
    smiles_col:      str,
    target_col:      str,
    frac:            Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed:            int  = 42,
    oversample_train: bool = False,
) -> Tuple[List, List, List, List, List, List]:
    """
    Load a molecular property CSV and produce scaffold-split train/val/test sets.

    Steps
    -----
    1. Read the CSV with pandas.
    2. Canonicalize SMILES (labels filtered in lockstep).
    3. Scaffold-split via DeepChem's ScaffoldSplitter.
    4. Optionally oversample the training split only.
       Validation and test sets are never resampled to avoid optimistic metrics.

    Returns
    -------
    train_smiles, train_labels, val_smiles, val_labels, test_smiles, test_labels
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    for col, kind in [(smiles_col, "SMILES"), (target_col, "target")]:
        if col not in df.columns:
            raise ValueError(
                f"{kind} column '{col}' not found. "
                f"Available columns: {list(df.columns)}"
            )

    smiles = df[smiles_col].astype(str).tolist()
    labels = df[target_col].tolist()

    print(f"  Loaded {len(smiles)} rows from '{os.path.basename(csv_path)}'.")

    # Canonicalize SMILES and filter labels in lockstep
    smiles, labels = _canonicalize_joint(smiles, labels)
    print(f"  {len(smiles)} molecules after canonicalization.")

    # Scaffold split via DeepChem.
    # DiskDataset.from_numpy requires a numeric X; a zero placeholder suffices
    # because only the SMILES ids and target values matter to the splitter.
    labels_arr = np.array(labels, dtype=np.float32)
    dataset    = DiskDataset.from_numpy(
        X   = np.zeros(len(smiles), dtype=np.float32),
        y   = labels_arr,
        ids = np.array(smiles),
    )

    splitter = ScaffoldSplitter()
    train_ds, val_ds, test_ds = splitter.train_valid_test_split(
        dataset,
        frac_train = frac[0],
        frac_valid = frac[1],
        frac_test  = frac[2],
        seed       = seed,
    )

    tr_s = train_ds.ids.tolist();  tr_l = train_ds.y.flatten().tolist()
    va_s = val_ds.ids.tolist();    va_l = val_ds.y.flatten().tolist()
    te_s = test_ds.ids.tolist();   te_l = test_ds.y.flatten().tolist()

    print(
        f"  Scaffold split → "
        f"Train: {len(tr_s)} | Val: {len(va_s)} | Test: {len(te_s)}"
    )

    # Oversample the training split only (never val or test)
    if oversample_train:
        try:
            from imblearn.over_sampling import RandomOverSampler
        except ImportError:
            raise ImportError(
                "imbalanced-learn is required for oversampling. "
                "Install with:  pip install imbalanced-learn"
            )
        ros = RandomOverSampler(random_state=seed)
        X_res, y_res = ros.fit_resample(
            np.array(tr_s).reshape(-1, 1),
            np.array(tr_l),
        )
        tr_s = X_res.flatten().tolist()
        tr_l = y_res.tolist()
        print(f"  After oversampling → Train: {len(tr_s)}")

    return tr_s, tr_l, va_s, va_l, te_s, te_l


def _tokenize(smiles_list: List[str], tokenizer) -> dict:
    """Tokenize a list of SMILES strings using a Hugging Face tokenizer."""
    return tokenizer(
        smiles_list,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )


def _make_loader(
    smiles:     List[str],
    labels:     List,
    tokenizer,
    shuffle:    bool = False,
    batch_size: int  = BATCH_SIZE,
) -> torch.utils.data.DataLoader:
    """Build a DataLoader from a list of SMILES strings and their labels."""
    enc = _tokenize(smiles, tokenizer)
    ds  = torch.utils.data.TensorDataset(
        enc["input_ids"],
        enc["attention_mask"],
        torch.tensor(labels, dtype=torch.float32),
    )
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def create_dataloaders(
    task_name: str,
    tokenizer,
    data_dir:  str = "data/raw",
    seed:      int = SEED,
) -> Tuple[
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    bool,
]:
    """
    Build train / val / test DataLoaders for a named task.

    Parameters
    ----------
    task_name : "bace" or "logp"
    tokenizer : Hugging Face tokenizer instance
    data_dir  : directory containing the dataset CSV
    seed      : random seed for scaffold splitting and oversampling

    Returns
    -------
    train_loader, val_loader, test_loader, is_regression
    """
    key = task_name.lower()
    if key not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown task '{task_name}'. "
            f"Supported tasks: {list(DATASET_CONFIGS.keys())}"
        )

    cfg      = DATASET_CONFIGS[key]
    csv_path = os.path.join(data_dir, cfg["csv_filename"])

    print(f"\nLoading dataset: {task_name.upper()}")

    tr_s, tr_l, va_s, va_l, te_s, te_l = load_and_split_dataset(
        csv_path         = csv_path,
        smiles_col       = cfg["smiles_col"],
        target_col       = cfg["target_col"],
        seed             = seed,
        oversample_train = cfg["oversample"],
    )

    train_loader = _make_loader(tr_s, tr_l, tokenizer, shuffle=True)
    val_loader   = _make_loader(va_s, va_l, tokenizer, shuffle=False)
    test_loader  = _make_loader(te_s, te_l, tokenizer, shuffle=False)

    return train_loader, val_loader, test_loader, cfg["is_regression"]


# ============================================================
# Head-mask utilities
# ============================================================

def build_head_mask(
    ampc_heads: List[Tuple[int, int]],
    num_layers: int = 3,
    num_heads:  int = 12,
    mode:       str = "ampc",
    seed:       int = 42,
) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
    """
    Build a binary head-selection mask of shape [num_layers, num_heads].

    Parameters
    ----------
    ampc_heads : task-specific AMPC-selected (layer_idx, head_idx) pairs
    mode       : "ampc"           – activate only the AMPC-selected heads.
               : "random_no_ampc" – activate the same number of heads, sampled
                                    uniformly from heads NOT in the AMPC set.
                                    Results depend on the seed; callers should
                                    aggregate over multiple seeds (see
                                    run_comparison).
    seed       : RNG seed for "random_no_ampc" (ignored for "ampc")

    Returns
    -------
    mask   : float tensor of shape [num_layers, num_heads] with values 0.0 / 1.0
    chosen : list of active (layer_idx, head_idx) pairs
    """
    all_positions = [(l, h) for l in range(num_layers) for h in range(num_heads)]

    if mode == "ampc":
        chosen = list(ampc_heads)

    elif mode == "random_no_ampc":
        rng      = np.random.default_rng(seed)
        ampc_set = set(ampc_heads)

        # Candidate pool: all heads NOT in the AMPC set for this task
        non_ampc_positions = [p for p in all_positions if p not in ampc_set]

        k      = len(ampc_heads)
        idxs   = rng.choice(len(non_ampc_positions), size=k, replace=False)
        chosen = [non_ampc_positions[i] for i in idxs]

    else:
        raise ValueError(
            f"Unknown mode '{mode}'. Choose: 'ampc' or 'random_no_ampc'."
        )

    mask = torch.zeros(num_layers, num_heads)
    for l, h in chosen:
        mask[l, h] = 1.0

    return mask.float(), chosen


# ============================================================
# Pooling utility
# ============================================================

def mean_pooling(
    last_hidden_state: torch.Tensor,  # [B, L, D]
    attention_mask:    torch.Tensor,  # [B, L]
) -> torch.Tensor:                    # [B, D]
    """Mean-pool token embeddings over non-padding positions."""
    mask   = attention_mask.unsqueeze(-1).float()          # [B, L, 1]
    summed = (last_hidden_state * mask).sum(dim=1)          # [B, D]
    denom  = mask.sum(dim=1).clamp(min=1e-8)                # [B, 1]
    return summed / denom


# ============================================================
# Model
# ============================================================

class SpecializedHeadChemBERTa(nn.Module):
    """
    ChemBERTa-based probing model supporting four experimental representations.

    representation_type | freeze_encoder | Paper terminology
    --------------------|----------------|----------------------------------
    "heads"             | True           | AMPC heads / random_no_AMPC heads
    "mean_pool"         | True           | frozen mean pooling
    "cls"               | False          | fine-tuned CLS token

    For AMPC heads and random_no_AMPC heads:
        - The encoder is frozen; only the MLP prediction head is trained.
        - For each selected head, the per-position value vectors are weighted
          by the actual softmax attention probabilities from the frozen encoder
          [ softmax(QK^T / sqrt(d)) @ V ] and mean-pooled over token positions
          to form the molecular representation.

    For frozen mean pooling:
        - The encoder is frozen; the final hidden state is mean-pooled over
          non-padding token positions. No head selection is performed.

    For fine-tuned CLS token:
        - The full ChemBERTa encoder is fine-tuned end-to-end; the [CLS] token
          embedding at position 0 of the final hidden state is used as the
          molecular representation.
    """

    def __init__(
        self,
        model_name:          str,
        selected_heads:      List[Tuple[int, int]],
        task_type:           str   = "classification",  # "classification" or "regression"
        representation_type: str   = "heads",           # "heads", "mean_pool", or "cls"
        freeze_encoder:      bool  = True,
        dropout:             float = 0.1,
    ) -> None:
        super().__init__()

        self.encoder = AutoModel.from_pretrained(
            model_name,
            attn_implementation="eager",
        )

        # Freeze the encoder for AMPC heads, random_no_AMPC heads, and frozen
        # mean pooling.  Only the fine-tuned CLS token baseline trains the full
        # encoder end-to-end.
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.selected_heads      = selected_heads
        self.task_type           = task_type
        self.representation_type = representation_type

        self.num_heads   = self.encoder.config.num_attention_heads
        self.hidden_size = self.encoder.config.hidden_size
        self.head_dim    = self.hidden_size // self.num_heads

        # Feature dimensionality fed into the downstream MLP prediction head
        if representation_type in {"mean_pool", "cls"}:
            self.feature_dim = self.hidden_size
        else:
            # Each selected attention head contributes head_dim features
            self.feature_dim = len(selected_heads) * self.head_dim

        self.dropout = nn.Dropout(dropout)

        # Lightweight MLP prediction head (shared architecture for all representations)
        self.predictor = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def train(self, mode: bool = True):
        """
        Override train() to keep the frozen encoder permanently in eval() mode.
        This prevents encoder-internal Dropout and BatchNorm layers from
        switching to training behaviour when model.train() is called externally.
        The fine-tuned CLS baseline (freeze_encoder=False) is unaffected.
        """
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def forward(
        self,
        input_ids:      torch.Tensor,  # [B, L]
        attention_mask: torch.Tensor,  # [B, L]
    ) -> torch.Tensor:                 # [B]
        """
        Forward pass.

        Returns
        -------
        logits : Tensor of shape [B]
            Raw (unactivated) scalar predictions.
            For classification, apply torch.sigmoid() to obtain probabilities.
        """
        use_heads = (self.representation_type == "heads")

        # Run the ChemBERTa encoder.
        # For frozen representations, disable gradient tracking to save memory.
        # For the fine-tuned CLS token, gradients must flow through the encoder.
        if self.freeze_encoder:
            with torch.no_grad():
                outputs = self.encoder(
                    input_ids            = input_ids,
                    attention_mask       = attention_mask,
                    output_hidden_states = use_heads,  # per-layer inputs for head extraction
                    output_attentions    = use_heads,  # softmax attention probabilities
                    return_dict          = True,
                )
        else:
            outputs = self.encoder(
                input_ids            = input_ids,
                attention_mask       = attention_mask,
                output_hidden_states = False,
                output_attentions    = False,
                return_dict          = True,
            )

        # ── Frozen mean pooling ──────────────────────────────────────────────
        # Average the frozen encoder's final hidden state over all non-padding
        # token positions to obtain the molecular representation.
        if self.representation_type == "mean_pool":
            pooled = mean_pooling(outputs.last_hidden_state, attention_mask)
            return self.predictor(self.dropout(pooled)).squeeze(-1)

        # ── Fine-tuned CLS token ─────────────────────────────────────────────
        # Extract the [CLS] token (position 0) from the fine-tuned encoder's
        # final hidden state as the molecular representation.
        if self.representation_type == "cls":
            cls_embedding = outputs.last_hidden_state[:, 0, :]
            return self.predictor(self.dropout(cls_embedding)).squeeze(-1)

        # ── AMPC heads / random_no_AMPC heads ────────────────────────────────
        # For each selected head (layer_idx, head_idx):
        #   1. Project the layer's input hidden state through the frozen value
        #      weight matrix to obtain per-head value vectors.
        #   2. Weight the value vectors with the actual softmax attention
        #      probabilities from the frozen encoder: softmax(QK^T / sqrt(d)) @ V
        #   3. Concatenate the weighted outputs across all selected heads.
        # Finally, mean-pool over token positions (masking padding tokens).
        if not self.selected_heads:
            raise ValueError(
                "selected_heads is empty but representation_type='heads'."
            )

        # outputs.hidden_states is a tuple of length num_layers + 1:
        #   hidden_states[0]   = embedding layer output (input to layer 0)
        #   hidden_states[i]   = output of transformer layer i-1 (input to layer i)
        # outputs.attentions is a tuple of length num_layers:
        #   attentions[i]      = softmax attention probs of transformer layer i
        hidden_states = outputs.hidden_states
        attn_probs    = outputs.attentions

        # Group selected heads by layer to reuse the value projection per layer
        heads_by_layer: Dict[int, List[int]] = {}
        for layer_idx, head_idx in self.selected_heads:
            heads_by_layer.setdefault(layer_idx, []).append(head_idx)

        collected_heads = []

        for layer_idx, head_list in heads_by_layer.items():
            # hidden_states[layer_idx] is the input to transformer layer layer_idx.
            # The Q and K that produced attn_probs[layer_idx] were also computed
            # from this tensor, so the computation is internally consistent.
            layer_input = hidden_states[layer_idx]   # [B, L, D]
            value_proj  = (
                self.encoder.encoder.layer[layer_idx].attention.self.value
            )

            value   = value_proj(layer_input)         # [B, L, D]
            B, L, _ = value.shape
            # Reshape to [B, num_heads, L, head_dim]
            value = value.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

            for head_idx in head_list:
                # attn_probs[layer_idx] has shape [B, num_heads, L, L]
                head_attn   = attn_probs[layer_idx][:, head_idx]  # [B, L, L]
                head_value  = value[:, head_idx]                   # [B, L, head_dim]
                head_output = torch.matmul(head_attn, head_value)  # [B, L, head_dim]
                collected_heads.append(head_output)

        # Concatenate all selected heads along the feature dimension
        head_features = torch.cat(collected_heads, dim=-1)  # [B, L, n_heads * head_dim]

        # Mean-pool over token positions, excluding padding tokens
        mask   = attention_mask.unsqueeze(-1).float()        # [B, L, 1]
        pooled = (
            (head_features * mask).sum(dim=1)
            / mask.sum(dim=1).clamp(min=1e-8)
        )
        # pooled: [B, n_selected_heads * head_dim]

        return self.predictor(self.dropout(pooled)).squeeze(-1)


# ============================================================
# Metrics
# ============================================================

def compute_metrics(
    y_true:        np.ndarray,
    y_score:       np.ndarray,
    is_regression: bool,
    avg_loss:      float,
) -> dict:
    """
    Compute task-appropriate evaluation metrics.

    Classification : loss, ROC-AUC, PR-AUC
    Regression     : loss, RMSE, MAE
    """
    metrics: dict = {"loss": avg_loss}

    if is_regression:
        rmse = float(np.sqrt(mean_squared_error(y_true, y_score)))
        mae  = float(mean_absolute_error(y_true, y_score))
        metrics.update({"rmse": rmse, "mae": mae})
    else:
        try:
            roc = float(roc_auc_score(y_true, y_score))
        except ValueError:
            roc = float("nan")
        try:
            pr = float(average_precision_score(y_true, y_score))
        except ValueError:
            pr = float("nan")
        metrics.update({"roc_auc": roc, "pr_auc": pr})

    return metrics


# ============================================================
# Training utilities
# ============================================================

def run_epoch(
    model:         nn.Module,
    loader:        torch.utils.data.DataLoader,
    is_regression: bool,
    optimizer:     Optional[torch.optim.Optimizer] = None,
    scheduler                                       = None,
    train:         bool = True,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Run one full pass over *loader*.

    Parameters
    ----------
    train : bool
        If True, compute gradients and update model weights (training mode).
        If False, disable gradients for validation / test evaluation.

    Returns
    -------
    avg_loss : float
    y_true   : np.ndarray
    y_scores : np.ndarray – sigmoid probabilities for classification,
                            raw predictions for regression.
    """
    model.train(train)
    criterion = nn.MSELoss() if is_regression else nn.BCEWithLogitsLoss()

    total_loss  = 0.0
    all_targets: List[float] = []
    all_outputs: List[float] = []

    grad_ctx = torch.enable_grad() if train else torch.no_grad()

    with grad_ctx:
        for batch in loader:
            input_ids, attention_mask, labels = [x.to(DEVICE) for x in batch]
            labels = labels.float()

            if train:
                optimizer.zero_grad()

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss   = criterion(logits, labels)

            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                scheduler.step()

            total_loss += loss.item() * input_ids.size(0)
            all_targets.extend(labels.detach().cpu().numpy().tolist())

            scores = (
                logits.detach().cpu().numpy()
                if is_regression
                else torch.sigmoid(logits).detach().cpu().numpy()
            )
            all_outputs.extend(scores.tolist())

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, np.array(all_targets), np.array(all_outputs)


def train_model(
    model:          nn.Module,
    train_loader:   torch.utils.data.DataLoader,
    val_loader:     torch.utils.data.DataLoader,
    is_regression:  bool,
    task_name:      str,
    experiment_tag: str,
    seed:           int = SEED,
) -> nn.Module:
    """
    Train *model* with AdamW + linear warm-up schedule and early stopping.

    The best checkpoint (by val ROC-AUC for classification, val RMSE for
    regression) is saved to disk and reloaded before returning, so the
    caller always receives the best-seen model rather than the final one.

    Parameters
    ----------
    seed : included in the checkpoint filename so that repeated runs of the
           random_no_AMPC baseline (different seeds) do not overwrite each other.
    """
    model.to(DEVICE)

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    total_steps = len(train_loader) * EPOCHS
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = int(0.1 * total_steps),
        num_training_steps = total_steps,
    )

    save_path        = f"best_{task_name.lower()}_{experiment_tag}_seed{seed}.pt"
    best_metric      = float("inf") if is_regression else -float("inf")
    patience_counter = 0

    for epoch in tqdm(range(EPOCHS), desc=f"{task_name.upper()} | {experiment_tag}"):
        run_epoch(
            model, train_loader, is_regression,
            optimizer=optimizer, scheduler=scheduler, train=True,
        )

        val_loss, y_true, y_score = run_epoch(
            model, val_loader, is_regression, train=False,
        )
        val_metrics = compute_metrics(y_true, y_score, is_regression, val_loss)

        if is_regression:
            monitor  = val_metrics["rmse"]
            improved = monitor < best_metric
        else:
            monitor  = val_metrics["roc_auc"]
            improved = monitor > best_metric

        if improved:
            best_metric      = monitor
            patience_counter = 0
            torch.save(copy.deepcopy(model.state_dict()), save_path)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch + 1}.")
                break

    # Reload the best-seen checkpoint before returning
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=DEVICE))

    return model


# ============================================================
# Experiment runner
# ============================================================

def _build_experiment_tag(
    representation_type: str,
    mask_type:           Optional[str] = None,
) -> str:
    """
    Build a unique, filename-safe tag for each experiment.

    Examples
    --------
    ("mean_pool", None)             → "mean_pool"
    ("cls",       None)             → "cls"
    ("heads",     "ampc")           → "heads_ampc"
    ("heads",     "random_no_ampc") → "heads_random_no_ampc"
    """
    if representation_type in {"mean_pool", "cls"}:
        return representation_type
    if mask_type is None:
        raise ValueError(
            "mask_type must be specified when representation_type='heads'."
        )
    return f"heads_{mask_type}"


# Human-readable labels consistent with manuscript terminology
_PRETTY_LABELS: Dict[Tuple[str, Optional[str]], str] = {
    ("heads",     "ampc"):           "AMPC heads",
    ("heads",     "random_no_ampc"): "random_no_AMPC heads",
    ("mean_pool", None):             "frozen mean pooling",
    ("cls",       None):             "fine-tuned CLS token",
}


def run_experiment(
    task_name:           str,
    data_dir:            str         = "data/raw",
    representation_type: str         = "heads",
    mask_type:           Optional[str] = None,
    random_seed:         int         = 42,
) -> dict:
    """
    Train and evaluate one (task, representation) combination end-to-end.

    Parameters
    ----------
    task_name           : "bace" or "logp"
    data_dir            : directory containing bace.csv
    representation_type : "heads", "mean_pool", or "cls"
    mask_type           : "ampc" or "random_no_ampc" (required when "heads")
    random_seed         : controls random head sampling and data-loading seed

    Returns
    -------
    dict with keys: task, representation, mask_type, n_active_heads,
                    active_heads, test_metrics
    """
    set_seed(random_seed)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_loader, val_loader, test_loader, is_regression = create_dataloaders(
        task_name = task_name,
        tokenizer = tokenizer,
        data_dir  = data_dir,
        seed      = random_seed,
    )

    # Retrieve the AMPC head set for this task
    ampc_heads = TASK_AMPC_HEADS[task_name.lower()]

    # Resolve selected heads and determine whether to freeze the encoder
    if representation_type == "heads":
        if mask_type not in {"ampc", "random_no_ampc"}:
            raise ValueError(
                f"Unsupported mask_type '{mask_type}'. "
                "Choose: 'ampc' or 'random_no_ampc'."
            )
        _, chosen_heads = build_head_mask(
            ampc_heads, mode=mask_type, seed=random_seed,
        )
        freeze_encoder = True   # encoder is always frozen for head-based representations

    elif representation_type == "mean_pool":
        # Frozen mean pooling: no head selection; encoder is frozen
        chosen_heads   = []
        freeze_encoder = True

    elif representation_type == "cls":
        # Fine-tuned CLS token: encoder is fully fine-tuned end-to-end
        chosen_heads   = []
        freeze_encoder = False

    else:
        raise ValueError(
            f"Unsupported representation_type '{representation_type}'. "
            "Choose: 'heads', 'mean_pool', or 'cls'."
        )

    pretty_label = _PRETTY_LABELS.get(
        (representation_type, mask_type),
        f"{representation_type}/{mask_type}",
    )

    print(
        f"\n{'─' * 60}"
        f"\nTask           : {task_name.upper()}"
        f"\nRepresentation : {pretty_label}"
        f"\nAMPC heads     : {ampc_heads}"
        f"\nActive heads   : {chosen_heads} ({len(chosen_heads)} heads)"
        f"\nEncoder frozen : {freeze_encoder}  |  Seed: {random_seed}"
        f"\n{'─' * 60}"
    )

    model = SpecializedHeadChemBERTa(
        model_name          = MODEL_NAME,
        selected_heads      = chosen_heads,
        task_type           = "regression" if is_regression else "classification",
        representation_type = representation_type,
        freeze_encoder      = freeze_encoder,
    )

    experiment_tag = _build_experiment_tag(representation_type, mask_type)

    model = train_model(
        model          = model,
        train_loader   = train_loader,
        val_loader     = val_loader,
        is_regression  = is_regression,
        task_name      = task_name,
        experiment_tag = experiment_tag,
        seed           = random_seed,
    )

    # Evaluate on the held-out test set
    test_loss, y_true, y_score = run_epoch(
        model, test_loader, is_regression, train=False,
    )
    test_metrics = compute_metrics(y_true, y_score, is_regression, test_loss)

    print(f"\n  Test metrics: {test_metrics}")

    return {
        "task":           task_name.upper(),
        "representation": pretty_label,
        "mask_type":      mask_type,
        "n_active_heads": len(chosen_heads),
        "active_heads":   chosen_heads,
        "test_metrics":   test_metrics,
    }


# ============================================================
# Comparison experiment
# ============================================================

# The four representations compared in the paper, in presentation order.
PAPER_EXPERIMENTS: List[Tuple[str, Optional[str]]] = [
    ("heads",     "ampc"),           # AMPC heads
    ("heads",     "random_no_ampc"), # random_no_AMPC heads (averaged over N seeds)
    ("mean_pool", None),             # frozen mean pooling
    ("cls",       None),             # fine-tuned CLS token
]


def run_comparison(
    task_name:      str,
    data_dir:       str = "data/raw",
    base_seed:      int = SEED,
    n_random_seeds: int = N_RANDOM_SEEDS,
) -> pd.DataFrame:
    """
    Run all four paper experiments for one task and save a CSV result file.

    For the random_no_AMPC baseline, the experiment is repeated n_random_seeds
    times (using seeds base_seed, base_seed+1, …, base_seed+n_random_seeds-1)
    and results are reported as mean ± standard deviation across seeds.

    Parameters
    ----------
    task_name      : "bace" or "logp"
    data_dir       : directory containing bace.csv
    base_seed      : base random seed; random_no_AMPC seeds are base_seed + i
    n_random_seeds : number of independent runs for the random_no_AMPC baseline

    Returns
    -------
    pd.DataFrame with one row per experiment and columns for all metrics
    """
    rows: List[dict] = []

    for representation_type, mask_type in PAPER_EXPERIMENTS:

        # ── random_no_AMPC heads ─────────────────────────────────────────────
        # Repeat with multiple seeds and report mean ± std to obtain a stable
        # estimate of the expected performance of an uninformed head selection.
        if representation_type == "heads" and mask_type == "random_no_ampc":
            seed_results: List[dict] = []

            for i in range(n_random_seeds):
                seed_i = base_seed + i
                result = run_experiment(
                    task_name           = task_name,
                    data_dir            = data_dir,
                    representation_type = representation_type,
                    mask_type           = mask_type,
                    random_seed         = seed_i,
                )
                seed_results.append(result["test_metrics"])

            # Aggregate: mean and standard deviation across all seeds
            averaged = {
                k: float(np.mean([r[k] for r in seed_results]))
                for k in seed_results[0]
            }
            std_vals = {
                f"{k}_std": float(np.std([r[k] for r in seed_results]))
                for k in seed_results[0]
            }

            row = {
                "Task":           task_name.upper(),
                "Representation": "random_no_AMPC heads",
                "Mask":           f"random_no_ampc (n={n_random_seeds})",
                "N_Active_Heads": len(TASK_AMPC_HEADS[task_name.lower()]),
                "Seeds":          f"{base_seed}–{base_seed + n_random_seeds - 1}",
            }
            row.update(averaged)
            row.update(std_vals)
            rows.append(row)
            continue

        # ── All other representations (single run) ───────────────────────────
        result = run_experiment(
            task_name           = task_name,
            data_dir            = data_dir,
            representation_type = representation_type,
            mask_type           = mask_type,
            random_seed         = base_seed,
        )

        row = {
            "Task":           result["task"],
            "Representation": result["representation"],
            "Mask":           mask_type if mask_type is not None else "–",
            "N_Active_Heads": result["n_active_heads"],
            "Seeds":          str(base_seed),
        }
        row.update(result["test_metrics"])
        rows.append(row)

    df = pd.DataFrame(rows)

    out_path = f"{task_name.lower()}_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved results to '{out_path}'")
    print(df.to_string(index=False))

    return df


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    DATA_DIR = "data/raw"

    print("=" * 60)
    print("AMPC – Attention-based Molecular Property Prediction")
    print("=" * 60)
    print(f"Model          : {MODEL_NAME}")
    print(f"Device         : {DEVICE}")
    print(f"Base seed      : {SEED}")
    print(f"Random seeds   : {N_RANDOM_SEEDS}  (for random_no_AMPC baseline)")

    # ── Task 1: BACE binary classification ──────────────────────────────────
    print("\n\n" + "=" * 60)
    print("Task 1: BACE Binary Classification")
    print("=" * 60)
    bace_df = run_comparison(
        task_name      = "bace",
        data_dir       = DATA_DIR,
        base_seed      = SEED,
        n_random_seeds = N_RANDOM_SEEDS,
    )

    # ── Task 2: LogP regression on the BACE molecule set ────────────────────
    print("\n\n" + "=" * 60)
    print("Task 2: LogP Regression (BACE molecules)")
    print("=" * 60)
    logp_df = run_comparison(
        task_name      = "logp",
        data_dir       = DATA_DIR,
        base_seed      = SEED,
        n_random_seeds = N_RANDOM_SEEDS,
    )

    print("\n\nAll experiments complete.")
    print("Results saved to:  bace_results.csv  |  logp_results.csv")