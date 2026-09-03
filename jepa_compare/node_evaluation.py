from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def macro_micro_f1(target: Tensor, prediction: Tensor, num_classes: int) -> tuple[float, float]:
    micro = (target == prediction).float().mean().item()
    per_class = []
    for c in range(num_classes):
        tp = ((target == c) & (prediction == c)).sum().float()
        fp = ((target != c) & (prediction == c)).sum().float()
        fn = ((target == c) & (prediction != c)).sum().float()
        per_class.append((2 * tp / (2 * tp + fp + fn).clamp_min(1)).item())
    return float(sum(per_class) / len(per_class)), micro


@dataclass(frozen=True)
class ProbeSplit:
    train: Tensor
    validation: Tensor
    test: Tensor


def stratified_split(
    labels: Tensor,
    train_ratio: float,
    validation_ratio_within_train: float = 0.1,
    seed: int = 42,
) -> ProbeSplit:
    """Stratify by class; train+validation equals the stated label ratio."""
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be in (0, 1)")
    if not 0 <= validation_ratio_within_train < 1:
        raise ValueError("validation_ratio_within_train must be in [0, 1)")
    generator = torch.Generator(device=labels.device).manual_seed(seed)
    train_parts, validation_parts, test_parts = [], [], []
    for c in range(int(labels.max().item()) + 1):
        indices = (labels == c).nonzero(as_tuple=False).flatten()
        if indices.numel() < 3:
            raise ValueError("each class needs at least three nodes for train/validation/test")
        order = torch.randperm(indices.numel(), generator=generator, device=labels.device)
        indices = indices[order]
        labeled_count = max(2, int(indices.numel() * train_ratio))
        labeled_count = min(labeled_count, indices.numel() - 1)
        validation_count = int(labeled_count * validation_ratio_within_train)
        if validation_ratio_within_train > 0:
            validation_count = max(1, validation_count)
        training_count = labeled_count - validation_count
        train_parts.append(indices[:training_count])
        validation_parts.append(indices[training_count:labeled_count])
        test_parts.append(indices[labeled_count:])
    return ProbeSplit(torch.cat(train_parts), torch.cat(validation_parts), torch.cat(test_parts))


class MLPProbe(nn.Module):
    """Lightweight MLP downstream head stated by the SG-JEPA paper."""

    def __init__(self, input_dim: int, classes: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden_dim = input_dim if hidden_dim is None else hidden_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, classes)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)


def fit_probe(
    embeddings: Tensor,
    labels: Tensor,
    train_indices: Tensor,
    evaluation_indices: Tensor,
    epochs: int,
    seed: int,
    hidden_dim: int | None = None,
) -> dict[str, float]:
    rng_state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    classes = int(labels.max().item()) + 1
    head = MLPProbe(embeddings.shape[1], classes, hidden_dim).to(embeddings.device)
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-2, weight_decay=1e-4)
    frozen = embeddings.detach()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = F.cross_entropy(head(frozen[train_indices]), labels[train_indices])
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        prediction = head(frozen[evaluation_indices]).argmax(dim=-1)
    macro, micro = macro_micro_f1(labels[evaluation_indices], prediction, classes)
    torch.random.set_rng_state(rng_state)
    return {"macro_f1": macro, "micro_f1": micro}


def fit_probe_ensemble(
    embeddings: dict[str, Tensor],
    labels: Tensor,
    train_indices: Tensor,
    evaluation_indices: Tensor,
    epochs: int,
    seed: int,
    hidden_dim: int | None = None,
) -> dict[str, float]:
    """Fit one probe per frozen scale and average their logits.

    Keeping the heads separate prevents a high-variance or over-smoothed scale
    from dominating a concatenated feature vector.  The representation names
    and their order must be fixed before looking at the test split.
    """
    if len(embeddings) < 2:
        raise ValueError("a multi-scale ensemble requires at least two views")
    rng_state = torch.random.get_rng_state()
    classes = int(labels.max().item()) + 1
    logits = []
    for offset, frozen in enumerate(embeddings.values()):
        torch.manual_seed(seed + offset)
        head = MLPProbe(frozen.shape[1], classes, hidden_dim).to(frozen.device)
        optimizer = torch.optim.Adam(head.parameters(), lr=1e-2, weight_decay=1e-4)
        frozen = frozen.detach()
        for _ in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(head(frozen[train_indices]), labels[train_indices])
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            logits.append(head(frozen[evaluation_indices]))
    prediction = torch.stack(logits).mean(dim=0).argmax(dim=-1)
    macro, micro = macro_micro_f1(labels[evaluation_indices], prediction, classes)
    torch.random.set_rng_state(rng_state)
    return {"macro_f1": macro, "micro_f1": micro}


def validation_probe(embeddings: Tensor, labels: Tensor, split: ProbeSplit, epochs: int,
                     seed: int, hidden_dim: int | None = None) -> dict[str, float]:
    return fit_probe(embeddings, labels, split.train, split.validation, epochs, seed, hidden_dim)


def final_probe(embeddings: Tensor, labels: Tensor, split: ProbeSplit, epochs: int,
                seed: int, hidden_dim: int | None = None) -> dict[str, float]:
    full_train = torch.cat([split.train, split.validation])
    return fit_probe(embeddings, labels, full_train, split.test, epochs, seed, hidden_dim)


def validation_probe_ensemble(
    embeddings: dict[str, Tensor], labels: Tensor, split: ProbeSplit, epochs: int,
    seed: int, hidden_dim: int | None = None,
) -> dict[str, float]:
    return fit_probe_ensemble(
        embeddings, labels, split.train, split.validation, epochs, seed, hidden_dim
    )


def final_probe_ensemble(
    embeddings: dict[str, Tensor], labels: Tensor, split: ProbeSplit, epochs: int,
    seed: int, hidden_dim: int | None = None,
) -> dict[str, float]:
    full_train = torch.cat([split.train, split.validation])
    return fit_probe_ensemble(
        embeddings, labels, full_train, split.test, epochs, seed, hidden_dim
    )
