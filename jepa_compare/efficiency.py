"""Wall-clock and peak-memory accounting for one model run of the comparison.

One ``EfficiencyMeter`` is created per model *before* the model is built and
handed to the training loop, which wraps its phases in ``meter.measure(...)``.
``summary()`` then reports, in seconds / megabytes / counts only (so the
multi-seed aggregation can average them):

* ``parameters`` (trainable) and ``parameters_total`` (incl. frozen / EMA copies)
* ``setup_seconds``: model construction plus stream / history preparation
* ``train_epochs``, ``train_epoch_seconds_mean``, ``train_seconds_total``
* ``pretrain_epochs`` / ``pretrain_seconds_total`` (snapshot SSL baselines)
* ``validation_seconds_mean``: one validation pass during training
* ``time_to_best_seconds``: training + validation time until the selected
  checkpoint was produced (what a practitioner pays for the reported number)
* ``wall_seconds_total``: everything from construction to the final test pass
* ``test_seconds`` and ``test_examples_per_second``: the final test pass
* ``peak_memory_mb`` / ``baseline_memory_mb`` (CUDA only): peak allocation
  during the run versus what was already resident (the dataset) before it

Timings synchronise the accelerator, so a measured phase costs a few
microseconds more than an unmeasured one; the numbers of the main comparison
are unaffected.
"""
from __future__ import annotations

from contextlib import contextmanager
import time
from typing import Iterator

import torch
from torch import nn

_MB = 1024.0 * 1024.0


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return (trainable, total) parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return int(trainable), int(total)


class EfficiencyMeter:
    PHASES = (
        "setup",
        "pretrain_epoch",
        "train_epoch",
        "validation",
        "final_validation",
        "test",
    )

    def __init__(self, device: torch.device | str) -> None:
        self.device = torch.device(device)
        self.phases: dict[str, list[float]] = {phase: [] for phase in self.PHASES}
        # Epoch index of every validation pass (0 = before training), so the
        # time to the selected checkpoint can include exactly the validation
        # passes that preceded it whatever ``eval_every`` was.
        self.validation_epochs: list[int] = []
        self._started = time.perf_counter()
        self.baseline_memory_bytes: int | None = None
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
            self.baseline_memory_bytes = int(torch.cuda.memory_allocated(self.device))

    def _synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elif self.device.type == "mps" and hasattr(torch, "mps"):
            torch.mps.synchronize()

    @contextmanager
    def measure(self, phase: str, epoch: int | None = None) -> Iterator[None]:
        if phase not in self.phases:
            raise ValueError(f"unknown efficiency phase {phase!r}")
        if phase == "validation":
            if epoch is None:
                raise ValueError("validation measurements need their epoch")
            self.validation_epochs.append(int(epoch))
        self._synchronize()
        start = time.perf_counter()
        try:
            yield
        finally:
            self._synchronize()
            self.phases[phase].append(time.perf_counter() - start)

    def record(self, phase: str, seconds: float) -> None:
        """Add an externally timed phase (e.g. a preparation done elsewhere)."""
        if phase not in self.phases:
            raise ValueError(f"unknown efficiency phase {phase!r}")
        self.phases[phase].append(float(seconds))

    def summary(
        self,
        model: nn.Module | None,
        best_epoch: int,
        test_examples: float | None = None,
    ) -> dict[str, float]:
        """Collapse the recorded phases into the per-run efficiency record.

        ``best_epoch`` indexes ``train_epoch`` measurements (1-based; 0 means
        the untrained model was selected), so ``time_to_best_seconds`` covers
        setup, all pretraining, and the first ``best_epoch`` training epochs
        together with their validation passes.
        """
        train = self.phases["train_epoch"]
        pretrain = self.phases["pretrain_epoch"]
        validation = self.phases["validation"]
        test = self.phases["test"]
        best = max(0, min(int(best_epoch), len(train)))
        validation_until_best = sum(
            seconds
            for epoch, seconds in zip(self.validation_epochs, validation)
            if epoch <= best
        )
        time_to_best = (
            sum(self.phases["setup"])
            + sum(pretrain)
            + sum(train[:best])
            + validation_until_best
        )
        record: dict[str, float] = {
            "setup_seconds": float(sum(self.phases["setup"])),
            "pretrain_epochs": float(len(pretrain)),
            "pretrain_seconds_total": float(sum(pretrain)),
            "train_epochs": float(len(train)),
            "train_epoch_seconds_mean": float(sum(train) / len(train)) if train else 0.0,
            "train_seconds_total": float(sum(train)),
            "validation_passes": float(len(validation)),
            "validation_seconds_mean": (
                float(sum(validation) / len(validation)) if validation else 0.0
            ),
            "time_to_best_seconds": float(time_to_best),
            "test_seconds": float(sum(test)),
            "wall_seconds_total": float(time.perf_counter() - self._started),
        }
        if test_examples is not None and test:
            record["test_examples_per_second"] = float(test_examples) / float(sum(test))
        if model is not None:
            trainable, total = count_parameters(model)
            record["parameters"] = float(trainable)
            record["parameters_total"] = float(total)
        else:
            record["parameters"] = 0.0
            record["parameters_total"] = 0.0
        if self.device.type == "cuda":
            self._synchronize()
            record["peak_memory_mb"] = float(
                torch.cuda.max_memory_allocated(self.device) / _MB
            )
            record["baseline_memory_mb"] = float(
                (self.baseline_memory_bytes or 0) / _MB
            )
        return record
