"""The wandb integration must never be able to break a training run."""

from __future__ import annotations

import argparse
import sys
import types

import pytest

from jepa_compare import wandb_logging


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    """Every test starts from a blank module state and a clean environment."""
    for name in ("DYGJEPA_WANDB", "WANDB_MODE", "WANDB_PROJECT", "WANDB_ENTITY", "WANDB_DIR"):
        monkeypatch.delenv(name, raising=False)
    wandb_logging._reset_state()
    yield
    wandb_logging._reset_state()


class FakeRun:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.logs: list[tuple[int | None, dict]] = []
        self.summary: dict = {}
        self.exit_code: int | None = None

    def log(self, metrics, step=None):
        self.logs.append((step, dict(metrics)))

    def finish(self, exit_code=0):
        self.exit_code = exit_code


def install_fake_wandb(monkeypatch, *, fail_modes=()):
    """Register a stand-in ``wandb`` module and return it."""
    module = types.ModuleType("wandb")
    module.runs = []

    def Settings(**kwargs):  # noqa: N802 - mirrors the real wandb API
        return kwargs

    def init(mode=None, **kwargs):
        if mode in fail_modes:
            raise RuntimeError(f"{mode} unavailable")
        run = FakeRun(mode=mode, **kwargs)
        module.runs.append(run)
        return run

    module.Settings = Settings
    module.init = init
    monkeypatch.setitem(sys.modules, "wandb", module)
    return module


def enabled_config(**overrides):
    wandb_section = {"enabled": True, "project": "unit-test", **overrides}
    return {"seed": 3, "wandb": wandb_section, "common_model": {"window_size": 2}}


# --------------------------------------------------------------------- flatten


def test_flatten_keeps_numbers_and_drops_everything_else():
    flat = wandb_logging.flatten(
        "val", {"ap": 0.5, "auc": 1, "protocol": "dyglib", "converged": True, "x": None}
    )
    assert flat == {"val/ap": 0.5, "val/auc": 1.0}


def test_flatten_tolerates_missing_metrics():
    assert wandb_logging.flatten("val", None) == {}


# -------------------------------------------------------------------- disabled


def test_disabled_by_default_and_handle_is_inert():
    assert wandb_logging.configure({"seed": 0}, task="link", dataset="wikipedia") is False
    assert wandb_logging.is_enabled() is False
    with wandb_logging.run_for_model("rcps_jepa", 0) as run:
        assert run.enabled is False
        run.log({"train/loss": 1.0}, step=1)
        run.summary({"final/test/ap": 0.9})


def test_missing_wandb_package_disables_instead_of_raising(monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", None)  # import raises ImportError
    assert wandb_logging.configure(enabled_config(), task="link", dataset="mooc") is False
    assert wandb_logging.is_enabled() is False


def test_env_switch_overrides_the_config(monkeypatch):
    install_fake_wandb(monkeypatch)
    monkeypatch.setenv("DYGJEPA_WANDB", "0")
    assert wandb_logging.configure(enabled_config(), task="link", dataset="mooc") is False

    monkeypatch.setenv("DYGJEPA_WANDB", "1")
    config = {"seed": 1, "wandb": {"enabled": False}}
    assert wandb_logging.configure(config, task="link", dataset="mooc") is True


def test_disabled_mode_wins_over_enabled_flag(monkeypatch):
    install_fake_wandb(monkeypatch)
    config = enabled_config(mode="disabled")
    assert wandb_logging.configure(config, task="link", dataset="mooc") is False


# ------------------------------------------------------------------ validation


def test_unknown_config_key_is_rejected():
    with pytest.raises(ValueError, match="unknown wandb config keys"):
        wandb_logging.configure(
            {"seed": 0, "wandb": {"enabled": True, "porject": "typo"}},
            task="link",
            dataset="mooc",
        )


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="wandb mode must be one of"):
        wandb_logging.configure(
            {"seed": 0, "wandb": {"enabled": True, "mode": "sometimes"}},
            task="link",
            dataset="mooc",
        )


# ------------------------------------------------------------------- run shape


def test_run_identity_groups_seeds_of_one_dataset(monkeypatch):
    module = install_fake_wandb(monkeypatch)
    wandb_logging.configure(enabled_config(), task="link", dataset="wikipedia")
    with wandb_logging.run_for_model("rcps_jepa", 4) as run:
        assert run.enabled is True
    (created,) = module.runs
    assert created.kwargs["name"] == "wikipedia-rcps_jepa-s4"
    assert created.kwargs["group"] == "link-wikipedia"
    assert created.kwargs["job_type"] == "rcps_jepa"
    assert created.kwargs["tags"] == ["link", "wikipedia", "rcps_jepa", "seed-4"]
    assert created.kwargs["config"]["seed"] == 4
    assert created.kwargs["config"]["dataset"] == "wikipedia"
    assert created.exit_code == 0


def test_seed_defaults_to_the_configured_run_seed(monkeypatch):
    module = install_fake_wandb(monkeypatch)
    wandb_logging.configure(enabled_config(), task="node", dataset="dblp")
    with wandb_logging.run_for_model("tgn"):
        pass
    assert module.runs[0].kwargs["name"] == "dblp-tgn-s3"


def test_metrics_and_summary_reach_the_run(monkeypatch):
    module = install_fake_wandb(monkeypatch)
    wandb_logging.configure(enabled_config(), task="link", dataset="mooc")
    with wandb_logging.run_for_model("tgat", 0) as run:
        run.log({"train/loss": 0.25}, step=1)
        run.summary({"final/test/ap": 0.87})
    created = module.runs[0]
    assert created.logs == [(1, {"train/loss": 0.25})]
    assert created.summary == {"final/test/ap": 0.87}


def test_failing_training_closes_the_run_with_a_nonzero_exit(monkeypatch):
    module = install_fake_wandb(monkeypatch)
    wandb_logging.configure(enabled_config(), task="link", dataset="mooc")
    with pytest.raises(RuntimeError, match="non-finite"):
        with wandb_logging.run_for_model("tgat", 0):
            raise RuntimeError("non-finite loss")
    assert module.runs[0].exit_code == 1


# --------------------------------------------------------------------- fallback


def test_auto_mode_falls_back_to_offline_without_network(monkeypatch):
    module = install_fake_wandb(monkeypatch, fail_modes={"online"})
    wandb_logging.configure(enabled_config(mode="auto"), task="link", dataset="mooc")
    with wandb_logging.run_for_model("tgat", 0) as run:
        assert run.enabled is True
    assert module.runs[0].kwargs["mode"] == "offline"


def test_the_offline_decision_is_made_once_per_process(monkeypatch):
    module = install_fake_wandb(monkeypatch, fail_modes={"online"})
    calls: list[str] = []
    original = module.init

    def counting_init(mode=None, **kwargs):
        calls.append(mode)
        return original(mode=mode, **kwargs)

    module.init = counting_init
    wandb_logging.configure(enabled_config(mode="auto"), task="link", dataset="mooc")
    for seed in range(3):
        with wandb_logging.run_for_model("tgat", seed):
            pass
    # online is attempted once, then every later run goes straight to offline
    assert calls == ["online", "offline", "offline", "offline"]


def test_explicit_offline_never_touches_the_network(monkeypatch):
    module = install_fake_wandb(monkeypatch, fail_modes={"online"})
    wandb_logging.configure(enabled_config(mode="offline"), task="link", dataset="mooc")
    with wandb_logging.run_for_model("tgat", 0):
        pass
    assert [run.kwargs["mode"] for run in module.runs] == ["offline"]


def test_explicit_online_disables_logging_when_it_fails(monkeypatch):
    install_fake_wandb(monkeypatch, fail_modes={"online"})
    wandb_logging.configure(enabled_config(mode="online"), task="link", dataset="mooc")
    with wandb_logging.run_for_model("tgat", 0) as run:
        assert run.enabled is False
    assert wandb_logging.is_enabled() is False


def test_logging_errors_are_swallowed(monkeypatch):
    module = install_fake_wandb(monkeypatch)
    wandb_logging.configure(enabled_config(), task="link", dataset="mooc")
    with wandb_logging.run_for_model("tgat", 0) as run:
        module.runs[0].log = lambda *a, **k: (_ for _ in ()).throw(OSError("disk full"))
        run.log({"train/loss": 1.0}, step=1)  # must not raise


# -------------------------------------------------------------------------- CLI


def _parse(argv):
    parser = argparse.ArgumentParser()
    wandb_logging.add_cli_arguments(parser)
    return parser.parse_args(argv)


def test_cli_flags_fold_into_the_config():
    config = {"wandb": {"enabled": False, "project": "from-yaml"}}
    wandb_logging.apply_cli_overrides(
        config, _parse(["--wandb", "--wandb-project", "from-cli", "--wandb-mode", "offline"])
    )
    assert config["wandb"] == {
        "enabled": True,
        "project": "from-cli",
        "mode": "offline",
    }


def test_no_wandb_flag_overrides_an_enabled_config():
    config = {"wandb": {"enabled": True}}
    wandb_logging.apply_cli_overrides(config, _parse(["--no-wandb"]))
    assert config["wandb"]["enabled"] is False


def test_absent_flags_leave_the_config_untouched():
    config = {"wandb": {"enabled": True, "project": "from-yaml"}}
    wandb_logging.apply_cli_overrides(config, _parse([]))
    assert config["wandb"] == {"enabled": True, "project": "from-yaml"}
