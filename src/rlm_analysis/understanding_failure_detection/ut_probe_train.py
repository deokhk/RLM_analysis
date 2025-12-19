import argparse
import copy
import json
import os
import pickle
import random
import shutil
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from transformers import set_seed

from rlm_analysis.understanding_failure_detection.ut_probe_model import load_model, hs_dict

try:
    import wandb  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    wandb = None  # type: ignore


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


@dataclass
class SignalExample:
    language: str
    sample_id: str
    hidden_state: torch.Tensor
    label: int


def _to_int(value: object) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, str):
        val = value.strip().lower()
        if val in {"1", "true", "yes", "not_understood", "n"}:
            return 1
        if val in {"0", "false", "no", "understood", "y"}:
            return 0
        try:
            return int(float(val))
        except ValueError:
            return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _extract_label(sample_payload: Dict[str, object]) -> Optional[int]:
    label = _to_int(sample_payload.get("not_understood_label"))
    if label in (0, 1):
        return label
    understood = sample_payload.get("understood")
    if isinstance(understood, str):
        understood_clean = understood.strip().lower()
        if understood_clean in {"yes", "understood", "y", "true"}:
            return 0
        if understood_clean in {"no", "not understood", "n", "false"}:
            return 1
    elif isinstance(understood, bool):
        return 0 if understood else 1
    elif isinstance(understood, (int, np.integer)):
        return 0 if int(understood) else 1
    correct = _to_int(sample_payload.get("correct"))
    if correct in (0, 1):
        return 1 - correct
    return None


def _normalise_hidden_state(hidden_state: object) -> Optional[torch.Tensor]:
    if hidden_state is None:
        return None
    if isinstance(hidden_state, torch.Tensor):
        tensor = hidden_state.detach().cpu().to(torch.float32)
    else:
        tensor = torch.tensor(hidden_state, dtype=torch.float32)
    if tensor.ndim > 1:
        tensor = tensor.view(-1)
    return tensor


def _load_signal_payload(file_path: str) -> Dict[str, Dict[str, Dict[str, object]]]:
    if file_path.endswith(".json"):
        with open(file_path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    else:
        payload = torch.load(file_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected payload type in {file_path}: {type(payload).__name__}")
    return payload


def _iter_signal_files(path: str) -> Iterable[str]:
    if os.path.isfile(path):
        yield path
        return
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Signal path not found: {path}")
    for root, _dirs, files in os.walk(path):
        for file_name in files:
            if file_name.endswith((".pth", ".pt", ".bin", ".json")):
                yield os.path.join(root, file_name)


def upsample_to_balance(dataset: TensorDataset, *, ratio: float = 1.0, seed: int = 42) -> TensorDataset:
    """
    Upsample the minority class to reach (minority : majority) = ratio : 1.
    - ratio=1.0 → perfectly balanced (1:1)
    - Works with binary labels {0,1}.
    """
    features, labels = dataset.tensors
    labels = labels.view(-1).to(torch.long)

    idx_pos = (labels == 1).nonzero(as_tuple=True)[0]
    idx_neg = (labels == 0).nonzero(as_tuple=True)[0]
    n_pos, n_neg = idx_pos.numel(), idx_neg.numel()

    if n_pos == 0 or n_neg == 0:
        print("[upsample_to_balance] Only one class present; skipping upsampling.")
        return dataset

    # Identify minority / majority
    if n_pos < n_neg:
        minority_idx, majority_idx = idx_pos, idx_neg
    else:
        minority_idx, majority_idx = idx_neg, idx_pos

    n_min, n_maj = minority_idx.numel(), majority_idx.numel()
    target_minority = int(max(1, round(n_maj * ratio)))

    if n_min >= target_minority:
        # Already at or above desired ratio; just permute to avoid bias
        all_idx = torch.cat([majority_idx, minority_idx], dim=0)
        perm = torch.randperm(all_idx.numel(), generator=torch.Generator().manual_seed(seed))
        all_idx = all_idx[perm]
        return TensorDataset(features[all_idx], labels[all_idx])

    n_to_add = target_minority - n_min
    g = torch.Generator().manual_seed(seed)
    add_idx = minority_idx[torch.randint(low=0, high=n_min, size=(n_to_add,), generator=g)]
    balanced_idx = torch.cat([majority_idx, minority_idx, add_idx], dim=0)

    # Shuffle final index set
    perm = torch.randperm(balanced_idx.numel(), generator=g)
    balanced_idx = balanced_idx[perm]
    return TensorDataset(features[balanced_idx], labels[balanced_idx])


def load_signal_examples(
    signal_path: str,
    *,
    languages: Optional[Sequence[str]] = None,
) -> List[SignalExample]:
    allowed_langs = {lang for lang in languages if lang} if languages else None
    examples: List[SignalExample] = []

    for file_path in sorted(_iter_signal_files(signal_path)):
        payload = _load_signal_payload(file_path)
        for language in sorted(payload.keys()):
            if allowed_langs and language not in allowed_langs:
                continue
            samples = payload[language]
            if not isinstance(samples, dict):
                continue
            for sample_id in sorted(samples.keys(), key=lambda sid: str(sid)):
                sample_payload = samples[sample_id]
                if not isinstance(sample_payload, dict):
                    continue
                hidden_state = _normalise_hidden_state(sample_payload.get("last_hidden_state"))
                if hidden_state is None:
                    continue
                label = _extract_label(sample_payload)
                if label not in (0, 1):
                    continue
                examples.append(
                    SignalExample(
                        language=language,
                        sample_id=str(sample_id),
                        hidden_state=hidden_state,
                        label=int(label),
                    )
                )
    if not examples:
        raise ValueError("No usable signal examples were loaded. Check the path and filters.")
    feature_dim = examples[0].hidden_state.shape[0]
    for example in examples:
        if example.hidden_state.shape[0] != feature_dim:
            raise ValueError("Inconsistent hidden-state dimensionality across samples.")
    return examples


def build_tensor_dataset(examples: Sequence[SignalExample]) -> TensorDataset:
    features = torch.stack([ex.hidden_state for ex in examples], dim=0)
    labels = torch.tensor([ex.label for ex in examples], dtype=torch.long)
    return TensorDataset(features, labels)


def split_dataset(
    dataset: TensorDataset,
    *,
    val_ratio: float,
    seed: int,
) -> Tuple[TensorDataset, Optional[TensorDataset], np.ndarray, np.ndarray]:
    if val_ratio <= 0.0 or len(dataset) < 2:
        indices = np.arange(len(dataset))
        return dataset, None, indices, np.array([], dtype=int)
    features, labels = dataset.tensors
    indices = np.arange(labels.shape[0])
    labels_np = labels.cpu().numpy()
    unique_labels = np.unique(labels_np)
    stratify = labels_np if unique_labels.size > 1 else None
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_ratio,
        random_state=seed,
        stratify=stratify,
    )
    train_dataset = TensorDataset(features[train_idx], labels[train_idx])
    val_dataset = TensorDataset(features[val_idx], labels[val_idx])
    return train_dataset, val_dataset, train_idx, val_idx


def get_class_weights(dataset: TensorDataset) -> Tuple[torch.Tensor, torch.Tensor]:
    # Inverse frequency weighting: weight_c = N / (2 * count_c)
    labels = dataset.tensors[1]
    n_pos = (labels > 0.5).sum().item()
    total = labels.numel()
    n_neg = total - n_pos

    if n_pos == 0 or n_neg == 0:
        class_weights = None
        print("Warning: Only one class present in training data; no class weights applied.")
        return torch.tensor([1.0, 1.0], dtype=torch.float)

    w_pos = total / (2 * n_pos)
    w_neg = total / (2 * n_neg)
    print(f"Compute class weights: w_neg={w_neg:.4f}, w_pos={w_pos:.4f}")
    class_weights = torch.tensor([w_neg, w_pos], dtype=torch.float)
    
    return class_weights


def determine_input_dim(examples: Sequence[SignalExample], model_name: Optional[str]) -> int:
    inferred_dim = examples[0].hidden_state.shape[0]
    if model_name and model_name in hs_dict:
        expected_dim = hs_dict[model_name]
        if expected_dim != inferred_dim:
            print(
                f"[Warning] Inferred hidden-state dim {inferred_dim} differs from hs_dict expectation {expected_dim} for {model_name}."
            )
    return inferred_dim


def build_dataloader(dataset: TensorDataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def build_loss_fn(class_weights_tensor: Optional[float]) -> nn.Module:
    class_weights_tensor = class_weights_tensor.to(device)
    loss_fct = nn.CrossEntropyLoss(weight=class_weights_tensor)
    return loss_fct 


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion, optimizer) -> float:
    model.train()
    losses: List[float] = []
    for inputs, labels in loader:
        inputs = inputs.to(device).to(torch.float32)
        labels = labels.to(device).to(torch.long)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else 0.0


def run_eval(args, model, criterion, loader: DataLoader):
    model.eval()
    preds, labels, probs, losses = [], [], [], []
    with torch.no_grad():
        for inputs, batch_labels in loader:
            inputs = inputs.to(device).to(torch.float32)
            batch_labels = batch_labels.to(device).to(torch.long)
            outputs = model(inputs)
            loss = criterion(outputs, batch_labels)
            probabilities = torch.softmax(outputs, dim=-1)
            class1_probs = probabilities[:, 1]
            batch_preds = (class1_probs > args.threshold).long()

            preds.extend(batch_preds.cpu().numpy())
            labels.extend(batch_labels.cpu().numpy())
            probs.extend(class1_probs.cpu().numpy())
            losses.append(loss.item())
    avg_loss = float(np.mean(losses)) if losses else 0.0
    return np.array(labels), np.array(preds), np.array(probs), avg_loss


def get_metrics(labels: np.ndarray, preds: np.ndarray) -> Tuple[float, float, float, float, float]:
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds)
    return accuracy, precision, recall, f1



def evaluate_split(name: str, args, model, criterion, loader: Optional[DataLoader]) -> Optional[Dict[str, float]]:
    if loader is None:
        return None
    labels, preds, probs, loss = run_eval(args, model, criterion, loader)
    accuracy, precision, recall, f1 = get_metrics(labels, preds)
    metrics = {
        "loss": loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    print(
        f"[{name}] loss={loss:.4f} acc={accuracy:.4f} prec={precision:.4f} recall={recall:.4f} f1={f1:.4f}"
    )
    return metrics


def parse_language_filter(languages: str) -> Optional[List[str]]:
    langs = [lang.strip() for lang in languages.split(",") if lang.strip()]
    return langs if langs else None


def train_probe(
    run_args,
    *,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    class_weights: torch.Tensor,
    input_dim: int,
    model_stub: str,
    llm_model_stub: str,
    save_dir: Optional[str],
    log_to_wandb: bool,
):
    set_seed(run_args.seed)
    model = load_model(input_dim, run_args.hidden_size, 2)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=run_args.lr, weight_decay=run_args.weight_decay)
    criterion = build_loss_fn(class_weights)

    best_metric = float("-inf")
    best_state = None
    history: Dict[str, List[float]] = {"train_loss": [], "val_metric": [], "val_loss": []}
    epochs_without_improvement = 0

    run_name = (
        f"{model_stub}_{run_args.dataset_type}_{getattr(run_args, 'polymath_split', 'na')}_seed{run_args.seed}"
        f"_from_{llm_model_stub}_hidden_{run_args.hidden_size}_lr_{run_args.lr}"
    )

    wandb_run = None
    if log_to_wandb and getattr(run_args, "wandb_project", None):
        if wandb is None:
            raise RuntimeError("wandb is not installed but logging was requested.")
        if getattr(run_args, "wandb_mode", None):
            os.environ["WANDB_MODE"] = run_args.wandb_mode
        wandb_kwargs = {
            "project": run_args.wandb_project,
            "name": run_name,
            "config": {
                "batch_size": run_args.batch_size,
                "epochs": run_args.epochs,
                "lr": run_args.lr,
                "weight_decay": run_args.weight_decay,
                "hidden_size": run_args.hidden_size,
                "threshold": run_args.threshold,
                "patience": run_args.patience,
                "dataset_type": run_args.dataset_type,
                "polymath_split": getattr(run_args, "polymath_split", "na"),
                "model_name": run_args.model_name,
                "llm_model_name": run_args.llm_model_name,
            },
        }
        if getattr(run_args, "wandb_entity", None):
            wandb_kwargs["entity"] = run_args.wandb_entity
        wandb_run = wandb.init(**wandb_kwargs)
        wandb_run.config.update({"signal_with_label_path": run_args.signal_with_label_path})
        print(f"Initialized Weights & Biases run: {run_name}")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for epoch in range(1, run_args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        history["train_loss"].append(train_loss)
        print(f"Epoch {epoch}/{run_args.epochs} - train_loss={train_loss:.4f}")

        epoch_log: Dict[str, float] = {"epoch": float(epoch), "train_loss": float(train_loss)}
        val_metrics = evaluate_split("val", run_args, model, criterion, val_loader)
        early_stop_triggered = False

        if val_metrics:
            tracked_metric = val_metrics[run_args.metric_to_track]
            history["val_loss"].append(val_metrics["loss"])
            history["val_metric"].append(tracked_metric)
            epoch_log.update({f"val_{k}": float(v) for k, v in val_metrics.items()})

            if tracked_metric > best_metric:
                best_metric = tracked_metric
                best_state = {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                }
                epoch_log["is_best"] = 1.0
                print(f"New best validation {run_args.metric_to_track}: {tracked_metric:.4f} (epoch {epoch})")
                epochs_without_improvement = 0
                if wandb_run:
                    wandb_run.summary[f"best_val_{run_args.metric_to_track}"] = tracked_metric
                    wandb_run.summary["best_epoch"] = epoch
                    for key, value in val_metrics.items():
                        wandb_run.summary[f"best_val_{key}"] = value
            else:
                epochs_without_improvement += 1
                epoch_log["epochs_without_improvement"] = float(epochs_without_improvement)
                if run_args.patience > 0 and epochs_without_improvement >= run_args.patience:
                    print(f"Early stopping triggered after {epochs_without_improvement} epochs without improvement.")
                    epoch_log["early_stop"] = 1.0
                    early_stop_triggered = True
        else:
            best_state = {
                "model": model.state_dict(),
                "epoch": epoch,
            }

        if wandb_run:
            wandb.log(epoch_log, step=epoch)

        if early_stop_triggered:
            break

    if best_state is None:
        best_state = {
            "model": model.state_dict(),
            "epoch": run_args.epochs,
        }

    if not history["val_metric"]:
        best_metric = None

    model.load_state_dict(best_state["model"])

    if wandb_run and best_state.get("val_metrics"):
        for key, value in best_state["val_metrics"].items():
            wandb_run.summary.setdefault(f"best_val_{key}", value)
        wandb_run.summary.setdefault("best_epoch", best_state.get("epoch", run_args.epochs))

    if save_dir:
        ckpt_path = os.path.join(save_dir, "best_probe.pt")
        torch.save(best_state, ckpt_path)
        history_path = os.path.join(save_dir, "training_history.json")
        with open(history_path, "w", encoding="utf-8") as fh:
            json.dump(history, fh, indent=2)
        print(f"Saved checkpoint to {ckpt_path}")
        print(f"Saved training history to {history_path}")

    if wandb_run:
        wandb_run.finish()

    return {
        "best_metric": best_metric,
        "best_state_epoch": best_state.get("epoch", run_args.epochs),
        "best_val_metrics": best_state.get("val_metrics"),
        "history": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a probe on last_hidden_state signals.")
    parser.add_argument("--signal_with_label_path", type=str, required=True,
                        help="Path to the train split signals (file or directory).")
    parser.add_argument("--languages", type=str, default="en,de,es,ar,ja,ko,th,bn,sw,te",
                        help="Comma-separated list of languages to keep.")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="Fraction of training data to use for validation.")
    parser.add_argument("--dataset_type", type=str, default="polymath",
                        choices=["polymath", "mmlu_prox_lite_dev", "mmlu_prox_lite", "mgsm_filtered"],
                        help="Source dataset type for building rows and matching ids.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed used for generating llm outputs and for data splits.")

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--hidden_size", type=int, default=0,
                        help="Hidden layer size for MLP probe. 0 = linear.")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Decision threshold applied to class-1 softmax probability.")
    parser.add_argument("--patience", type=int, default=5,
                        help="Epochs to wait for validation improvement before early stopping. 0 disables early stopping.")
    parser.add_argument("--model_name", type=str, default="probe",
                        help="Identifier for the probe model (used for logging).")
    parser.add_argument("--polymath_split", type=str, default="low",
                        help="PolyMath split name (used for logging and directory structure).")
    parser.add_argument("--llm_model_name", type=str, default=None,
                        help="Model name for saving and dimensionality checks. e.g. Qwen/Qwen3-4B, openai/gpt-oss-20b")
    parser.add_argument("--output_dir", type=str, default="./probe_ft_understandability",
                        help="Base directory to save best checkpoint and metrics.")
    parser.add_argument("--custom_postfix", type=str, default=None, help="Custom string to append to save checkpoint.")
    parser.add_argument("--metric_to_track", type=str, default="f1",
                        choices=["f1", "accuracy", "recall", "precision", "macro_f1"],
                        help="Validation metric used for early selection.")

    parser.add_argument("--balance_training", action="store_true",
                        help="Upsample training set to achieve a 1:1 class ratio (minority:majority).")
    parser.add_argument("--balance_ratio", type=float, default=1.0,
                        help="Target minority:majority ratio (default 1.0 for 1:1).")
    parser.add_argument("--no_class_weights", action="store_true",
                        help="Use unweighted cross-entropy (equivalent to class weights [1,1]).")

    parser.add_argument("--use_pca", action="store_true",
                        help="Apply PCA to hidden states before training. PCA is fit on the train split.")
    parser.add_argument("--pca_dim", type=int, default=None,
                        help="Target dimensionality for PCA. Required when --use_pca is set.")
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="Weights & Biases project name. Leave empty to disable logging.")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="Optional Weights & Biases entity/team.")
    parser.add_argument("--wandb_mode", type=str, default=None,
                        help="Optional WANDB_MODE override (e.g. offline, disabled).")
    args = parser.parse_args()
    set_seed(args.seed)

    llm_model_stub = (args.llm_model_name or "unknown_llm").split('/')[-1]
    model_stub = (args.model_name or "probe").split('/')[-1]
    dataset_dir = (
        f"{args.dataset_type}_{args.polymath_split}"
        if args.dataset_type == "polymath"
        else args.dataset_type
    )
    base_dir = os.path.join(
        args.output_dir,
        llm_model_stub,
        dataset_dir,
        f"seed_{args.seed}_{args.custom_postfix}" if args.custom_postfix else f"seed_{args.seed}",
    )
    os.makedirs(base_dir, exist_ok=True)

    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    language_filter = parse_language_filter(args.languages)
    examples = load_signal_examples(
        args.signal_with_label_path,
        languages=language_filter,
    )
    print(f"Loaded {len(examples)} training examples from {args.signal_with_label_path}")

    input_dim = determine_input_dim(examples, args.llm_model_name)
    dataset = build_tensor_dataset(examples)
    train_dataset, val_dataset, train_indices, val_indices = split_dataset(
        dataset,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    def _group_ids(indices: np.ndarray) -> Dict[str, List[str]]:
        grouped: Dict[str, List[str]] = {}
        for idx in indices:
            example = examples[int(idx)]
            grouped.setdefault(example.language, []).append(example.sample_id)
        for lang in grouped:
            grouped[lang].sort()
        return grouped

    split_metadata = {
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "total_examples": len(examples),
        "train_example_count": int(len(train_indices)),
        "validation_example_count": int(len(val_indices)),
        "splits": {
            "train": _group_ids(train_indices),
            "validation": _group_ids(val_indices),
        },
    }
    split_metadata_path = os.path.join(base_dir, "sample_id_splits.json")
    with open(split_metadata_path, "w", encoding="utf-8") as fh:
        json.dump(split_metadata, fh, indent=2)
    print(f"Saved sample id splits to {split_metadata_path}")
    pca_model = None
    pca_save_path: Optional[str] = None
    if args.use_pca:
        if args.pca_dim is None:
            raise ValueError("--pca_dim must be provided when --use_pca is enabled.")
        if args.pca_dim <= 0:
            raise ValueError("--pca_dim must be a positive integer.")
        if args.pca_dim > input_dim:
            raise ValueError(f"--pca_dim ({args.pca_dim}) cannot exceed the hidden-state dimension ({input_dim}).")

        train_features = train_dataset.tensors[0].cpu().numpy()
        train_labels = train_dataset.tensors[1]
        pca_model = PCA(n_components=args.pca_dim, random_state=args.seed)
        pca_model.fit(train_features)
        transformed_train = torch.from_numpy(pca_model.transform(train_features)).to(torch.float32)
        train_dataset = TensorDataset(transformed_train, train_labels)

        if val_dataset is not None:
            val_features = val_dataset.tensors[0].cpu().numpy()
            val_labels = val_dataset.tensors[1]
            transformed_val = torch.from_numpy(pca_model.transform(val_features)).to(torch.float32)
            val_dataset = TensorDataset(transformed_val, val_labels)

        input_dim = args.pca_dim
        pca_save_path = os.path.join(base_dir, "pca_model.pkl")
        with open(pca_save_path, "wb") as fh:
            pickle.dump(
                {
                    "pca": pca_model,
                    "fit_seed": args.seed,
                    "original_dim": dataset.tensors[0].shape[1],
                    "pca_dim": args.pca_dim,
                    "languages": language_filter,
                    "signal_path": args.signal_with_label_path,
                },
                fh,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        print(f"Saved PCA model fitted on train split to {pca_save_path}")

    if args.balance_training:
        before_counts = torch.bincount(train_dataset.tensors[1].to(torch.long), minlength=2)
        print(f"[Balance] Before upsampling: n0={int(before_counts[0])}, n1={int(before_counts[1])}")
        train_dataset = upsample_to_balance(
            train_dataset,
            ratio=float(args.balance_ratio),
            seed=args.seed,
        )
        after_counts = torch.bincount(train_dataset.tensors[1].to(torch.long), minlength=2)
        print(f"[Balance] After  upsampling: n0={int(after_counts[0])}, n1={int(after_counts[1])} "
              f"(target ratio={args.balance_ratio}:1)")

    train_loader = build_dataloader(train_dataset, args.batch_size, shuffle=True)
    val_loader = build_dataloader(val_dataset, args.batch_size, shuffle=False) if val_dataset else None

    if args.patience > 0 and val_loader is None:
        print("Warning: Patience > 0 but no validation split available; early stopping is disabled.")

    if args.no_class_weights or args.balance_training:
        class_weights = torch.tensor([1.0, 1.0], dtype=torch.float)
        print("Using unweighted cross-entropy (class weights = [1.0, 1.0]).")
    else:
        class_weights = get_class_weights(train_dataset)

    learning_rates = [1e-3, 1e-4, 1e-5]
    hidden_sizes = [0, 32, 128, 512, input_dim // 2, input_dim]

    print(f"Learning rate grid: {learning_rates}")
    print(f"Hidden size grid: {hidden_sizes}")

    original_wandb_project = args.wandb_project
    search_results: List[Dict[str, object]] = []

    for lr in learning_rates:
        for hidden_size in hidden_sizes:
            run_args = copy.deepcopy(args)
            run_args.lr = lr
            run_args.hidden_size = hidden_size
            run_args.wandb_project = None
            run_args.wandb_entity = None
            combo_label = f"lr={lr} hidden_size={hidden_size}"
            print(f"=== Grid search: {combo_label} ===")
            save_dir = os.path.join(base_dir, f"lr_{lr}_hidden_{hidden_size}")
            result = train_probe(
                run_args,
                train_loader=train_loader,
                val_loader=val_loader,
                class_weights=class_weights,
                input_dim=input_dim,
                model_stub=model_stub,
                llm_model_stub=llm_model_stub,
                save_dir=save_dir,
                log_to_wandb=False,
            )
            best_metric = result["best_metric"]
            metric_str = (
                "N/A" if best_metric is None or (isinstance(best_metric, float) and np.isnan(best_metric))
                else f"{float(best_metric):.4f}"
            )
            print(f"Result for {combo_label}: {args.metric_to_track}={metric_str}")
            search_results.append(
                {
                    "lr": lr,
                    "hidden_size": hidden_size,
                    "best_metric": None if best_metric is None else float(best_metric),
                    "best_epoch": result["best_state_epoch"],
                    "best_val_metrics": result["best_val_metrics"],
                }
            )

    best_result: Optional[Dict[str, object]] = None
    for res in search_results:
        metric = res["best_metric"]
        if metric is None or (isinstance(metric, float) and np.isnan(metric)):
            continue
        if best_result is None or float(metric) > float(best_result["best_metric"]):
            best_result = res

    if best_result is None:
        if search_results:
            best_result = search_results[0]
            print("No validation metrics available across grid; selecting first configuration by default.")
        else:
            raise RuntimeError("Grid search produced no results.")

    print("=== Grid search summary ===")
    for res in search_results:
        metric = res["best_metric"]
        metric_display = (
            "N/A" if metric is None or (isinstance(metric, float) and np.isnan(metric))
            else f"{float(metric):.4f}"
        )
        print(
            f"lr={res['lr']} hidden_size={res['hidden_size']} -> "
            f"{args.metric_to_track}={metric_display}"
        )

    best_lr = float(best_result["lr"])
    best_hidden_size = int(best_result["hidden_size"])
    print(
        f"Best configuration: lr={best_lr}, hidden_size={best_hidden_size}"
        f" (epoch {best_result['best_epoch']}, metric={best_result['best_metric']})"
    )

    best_dir = os.path.join(base_dir, f"lr_{best_lr}_hidden_{best_hidden_size}")
    # Remove other dirs
    for entry in os.listdir(base_dir):
        entry_path = os.path.join(base_dir, entry)
        if entry_path != best_dir and os.path.isdir(entry_path):
            shutil.rmtree(entry_path)
            print(f"Removed non-best configuration directory: {entry_path}")
    # Rename best dir to 'best'
    final_best_dir = os.path.join(base_dir, "grid_search_best")
    if os.path.exists(final_best_dir):
        shutil.rmtree(final_best_dir)
    shutil.move(best_dir, final_best_dir)
    print(f"Renamed best configuration directory to: {final_best_dir}")
    # final best dir에 lr이랑 hidden size info 담긴 json 파일 생성
    best_config_path = os.path.join(final_best_dir, "best_configuration.json")
    with open(best_config_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "lr": best_lr,
                "hidden_size": best_hidden_size,
                "best_epoch": best_result["best_epoch"],
                "best_val_metrics": best_result["best_val_metrics"],
            },
            fh,
            indent=2,
        )
    if args.use_pca and pca_save_path and os.path.exists(pca_save_path):
        final_pca_path = os.path.join(final_best_dir, "pca_model.pkl")
        shutil.copy2(pca_save_path, final_pca_path)
        print(f"Copied PCA model to {final_pca_path}")

if __name__ == "__main__":
    main()
