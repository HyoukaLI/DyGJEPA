"""Optional Weights & Biases logging for the comparison entry points.

The comparison scripts run on Slurm nodes that may or may not reach
``api.wandb.ai``.  Everything here therefore degrades instead of failing:

* ``wandb`` not installed, or logging disabled -> every call is a no-op.
* ``mode: auto`` (the default) -> try ``online`` once; if that fails for any
  reason, fall back to ``offline`` for the remainder of the process and print
  the reason.  Offline runs land in ``<dir>/wandb`` and are uploaded later with
  ``wandb sync``.
* Any unexpected error raised by ``wandb`` while logging is swallowed after a
  single warning.  A telemetry backend must never take down a training job.

One run is created per ``(dataset, model, seed)`` triple::

    name      wikipedia-rcps_jepa-s0
    group     link-wikipedia          # every model and seed of one dataset
    job_type  rcps_jepa               # group by this to aggregate over seeds
    tags      link, wikipedia, rcps_jepa, seed-0
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterator, Mapping

__all__ = [
    "add_cli_arguments",
    "apply_cli_overrides",
    "configure",
    "is_enabled",
    "run_for_model",
]

_DEFAULTS: dict[str, Any] = {
    "enabled": False,
    "project": "dygjepa",
    "entity": None,
    "mode": "auto",
    "group": None,
    "tags": (),
    "dir": None,
    "init_timeout": 30,
    "notes": None,
}

_TRUE = {"1", "true", "yes", "on"}
_MODES = {"auto", "online", "offline", "disabled"}

# Process-wide state.  ``configure`` rewrites it once per dataset.
_context: dict[str, Any] = {}
_resolved_mode: str | None = None
_warned: set[str] = set()


def _reset_state() -> None:
    """Clear the process-wide state.  Used by the tests."""
    global _context, _resolved_mode

    _context = {}
    _resolved_mode = None
    _warned.clear()


def _warn(key: str, message: str) -> None:
    """Print a warning at most once per process for the given key."""
    if key in _warned:
        return
    _warned.add(key)
    print(f"[wandb] {message}", flush=True)


def _coerce_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in _TRUE


def _numeric(value: Any) -> float | int | None:
    """Return ``value`` when it is a plain number, otherwise ``None``."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    return None


def flatten(prefix: str, metrics: Mapping[str, Any] | None) -> dict[str, float]:
    """Flatten a metrics mapping into ``prefix/key`` entries, numbers only."""
    if not metrics:
        return {}
    flat: dict[str, float] = {}
    for key, value in metrics.items():
        number = _numeric(value)
        if number is not None:
            flat[f"{prefix}/{key}"] = float(number)
    return flat


def add_cli_arguments(parser) -> None:
    """Register the shared ``--wandb*`` flags on an argparse parser."""
    group = parser.add_argument_group("weights & biases")
    group.add_argument(
        "--wandb",
        dest="wandb",
        action="store_true",
        default=None,
        help="enable Weights & Biases logging (one run per dataset/model/seed)",
    )
    group.add_argument(
        "--no-wandb",
        dest="wandb",
        action="store_false",
        help="disable Weights & Biases logging even if the config enables it",
    )
    group.add_argument("--wandb-project", default=None, help="wandb project name")
    group.add_argument("--wandb-entity", default=None, help="wandb entity (team/user)")
    group.add_argument(
        "--wandb-mode",
        default=None,
        choices=sorted(_MODES),
        help="auto (try online, fall back to offline), online, offline, disabled",
    )
    group.add_argument("--wandb-group", default=None, help="override the run group")
    group.add_argument("--wandb-tags", nargs="*", default=None, help="extra run tags")
    group.add_argument(
        "--wandb-dir",
        default=None,
        help="directory that holds offline run data (default: ./wandb)",
    )


def apply_cli_overrides(config: dict, args) -> None:
    """Fold parsed ``--wandb*`` flags into ``config['wandb']`` in place."""
    section = dict(config.get("wandb") or {})
    if getattr(args, "wandb", None) is not None:
        section["enabled"] = bool(args.wandb)
    for flag, key in (
        ("wandb_project", "project"),
        ("wandb_entity", "entity"),
        ("wandb_mode", "mode"),
        ("wandb_group", "group"),
        ("wandb_dir", "dir"),
    ):
        value = getattr(args, flag, None)
        if value is not None:
            section[key] = value
    tags = getattr(args, "wandb_tags", None)
    if tags is not None:
        section["tags"] = list(tags)
    if section:
        config["wandb"] = section


def _resolve_settings(config: Mapping[str, Any]) -> dict[str, Any]:
    section = dict(config.get("wandb") or {})
    unknown = set(section) - set(_DEFAULTS)
    if unknown:
        raise ValueError(f"unknown wandb config keys: {sorted(unknown)}")
    settings = {**_DEFAULTS, **section}

    # Environment always wins, so a Slurm script can flip logging without
    # touching any YAML.
    env_enabled = os.environ.get("DYGJEPA_WANDB")
    if env_enabled is not None:
        settings["enabled"] = _coerce_bool(env_enabled, False)
    settings["enabled"] = _coerce_bool(settings["enabled"], False)

    for env_name, key in (
        ("WANDB_PROJECT", "project"),
        ("WANDB_ENTITY", "entity"),
        ("WANDB_DIR", "dir"),
    ):
        value = os.environ.get(env_name)
        if value:
            settings[key] = value

    env_mode = os.environ.get("WANDB_MODE")
    if env_mode:
        settings["mode"] = env_mode
    mode = str(settings["mode"]).strip().lower()
    if mode not in _MODES:
        raise ValueError(f"wandb mode must be one of {sorted(_MODES)}, got {mode!r}")
    settings["mode"] = mode
    if mode == "disabled":
        settings["enabled"] = False

    settings["tags"] = [str(tag) for tag in (settings["tags"] or ())]
    settings["init_timeout"] = int(settings["init_timeout"])
    return settings


def configure(config: Mapping[str, Any], task: str, dataset: str) -> bool:
    """Prepare per-dataset logging state.  Returns whether logging is on."""
    global _context

    settings = _resolve_settings(config)
    _context = {
        "settings": settings,
        "task": task,
        "dataset": dataset,
        "seed": int(config.get("seed", 0)) if _numeric(config.get("seed")) else 0,
        "config": _run_config(config),
    }
    if not settings["enabled"]:
        return False
    try:
        import wandb  # noqa: F401
    except ImportError:
        _warn(
            "import",
            "wandb is not installed; logging is skipped. "
            "Install it with: pip install wandb",
        )
        _context["settings"] = {**settings, "enabled": False}
        return False
    return True


def is_enabled() -> bool:
    return bool(_context) and bool(_context["settings"]["enabled"])


def _run_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Copy the reproducibility-relevant parts of a comparison config."""
    keys = (
        "seed",
        "device",
        "split",
        "common_model",
        "link_prediction",
        "training",
        "data",
        "models",
    )
    snapshot: dict[str, Any] = {}
    for key in keys:
        if key in config:
            snapshot[key] = config[key]
    return snapshot


def _init_kwargs(model: str, seed: int, extra: Mapping[str, Any] | None) -> dict:
    settings = _context["settings"]
    task = _context["task"]
    dataset = _context["dataset"]
    run_config = {
        **_context["config"],
        "task": task,
        "dataset": dataset,
        "model": model,
        "seed": seed,
        **dict(extra or {}),
    }
    kwargs: dict[str, Any] = {
        "project": settings["project"],
        "name": f"{dataset}-{model}-s{seed}",
        "group": settings["group"] or f"{task}-{dataset}",
        "job_type": model,
        "tags": [task, dataset, model, f"seed-{seed}", *settings["tags"]],
        "config": run_config,
        "reinit": True,
    }
    if settings["entity"]:
        kwargs["entity"] = settings["entity"]
    if settings["dir"]:
        kwargs["dir"] = settings["dir"]
    if settings["notes"]:
        kwargs["notes"] = settings["notes"]
    return kwargs


def _start(model: str, seed: int, extra: Mapping[str, Any] | None):
    """Create a wandb run, degrading online -> offline -> disabled."""
    global _resolved_mode

    import wandb

    settings = _context["settings"]
    kwargs = _init_kwargs(model, seed, extra)
    try:
        # Bounds how long a network-less compute node waits before falling back.
        init_settings = wandb.Settings(init_timeout=settings["init_timeout"])
    except TypeError:
        init_settings = None
    if init_settings is not None:
        kwargs["settings"] = init_settings

    configured = settings["mode"]
    order = ["offline"] if configured == "offline" else ["online"]
    if configured == "auto":
        order.append("offline")
    if _resolved_mode is not None:
        # A previous run already established what works in this process.
        order = [_resolved_mode]

    last_error: Exception | None = None
    for mode in order:
        try:
            run = wandb.init(mode=mode, **kwargs)
        except Exception as exc:  # noqa: BLE001 - never fail the training job
            last_error = exc
            _warn(
                f"init-{mode}",
                f"could not start an {mode} run ({type(exc).__name__}: {exc})",
            )
            continue
        if _resolved_mode != mode:
            _resolved_mode = mode
            _warn(f"mode-{mode}", f"logging in {mode} mode")
            if mode == "offline":
                _warn(
                    "sync-hint",
                    "upload later from a networked node with: wandb sync wandb/offline-run-*",
                )
        return run

    _warn(
        "init-failed",
        f"disabling logging for this process ({last_error})",
    )
    _context["settings"] = {**settings, "enabled": False}
    return None


class _Run:
    """Thin wrapper so callers never touch ``wandb`` or ``None`` directly."""

    __slots__ = ("_run",)

    def __init__(self, run: Any | None) -> None:
        self._run = run

    @property
    def enabled(self) -> bool:
        return self._run is not None

    def log(self, metrics: Mapping[str, Any], step: int | None = None) -> None:
        if self._run is None or not metrics:
            return
        try:
            self._run.log(dict(metrics), step=step)
        except Exception as exc:  # noqa: BLE001
            _warn("log", f"dropping metrics ({type(exc).__name__}: {exc})")

    def summary(self, values: Mapping[str, Any]) -> None:
        if self._run is None or not values:
            return
        try:
            for key, value in values.items():
                self._run.summary[key] = value
        except Exception as exc:  # noqa: BLE001
            _warn("summary", f"dropping summary ({type(exc).__name__}: {exc})")

    def finish(self, exit_code: int = 0) -> None:
        if self._run is None:
            return
        try:
            self._run.finish(exit_code=exit_code)
        except Exception as exc:  # noqa: BLE001
            _warn("finish", f"could not close the run ({type(exc).__name__}: {exc})")
        finally:
            self._run = None


@contextmanager
def run_for_model(
    model: str, seed: int | None = None, extra_config: Mapping[str, Any] | None = None
) -> Iterator[_Run]:
    """Yield a run handle for one ``(dataset, model, seed)`` training job.

    ``seed`` defaults to the seed of the configured comparison run, for the
    training helpers that do not receive one explicitly.  The handle is a no-op
    when logging is disabled, so call sites need no conditionals.  A failing
    training job closes its run with a non-zero exit code before the exception
    propagates.
    """
    if seed is None:
        seed = int(_context.get("seed", 0)) if _context else 0
    handle = _Run(_start(model, seed, extra_config) if is_enabled() else None)
    try:
        yield handle
    except BaseException:
        handle.finish(exit_code=1)
        raise
    else:
        handle.finish()
