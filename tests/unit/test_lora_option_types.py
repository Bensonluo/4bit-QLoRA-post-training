"""Workbench must not treat booleans or fractional ranks as valid LoRA options."""

from unittest.mock import patch

import pytest

from tests.unit.test_training_plans import _save
from tests.unit.test_workbench_training_runs import _prepare

pytest_plugins = ["tests.unit.test_workbench_training_runs", "tests.unit.test_training_plans"]


@pytest.mark.parametrize(
    "options",
    [
        {"r": True},
        {"r": 1.5},
        {"r": float("nan")},
        {"r": float("inf")},
        {"lora_alpha": True},
        {"lora_alpha": 1.5},
        {"lora_alpha": float("inf")},
        {"lora_dropout": True},
        {"lora_dropout": float("nan")},
        {"lora_dropout": float("inf")},
        {"use_rslora": "true"},
        {"use_dora": 1},
    ],
)
def test_direct_preparation_blocks_invalid_lora_types(environment, options):
    record = _prepare(environment, lora_options=options)
    assert record["status"] == "blocked"
    assert record["config"] is None
    assert any(issue["severity"] == "blocking" for issue in record["issues"])


@pytest.mark.parametrize(
    "option", [{"lora_dropout": True}, {"lora_dropout": float("nan")}, {"use_dora": 1}]
)
def test_plan_does_not_save_invalid_lora_options(plans, option):
    with pytest.raises(ValueError):
        _save(plans, lora_options={"r": 2, "lora_alpha": 4, **option})


def test_valid_lora_types_remain_usable(environment, plans):
    record = _prepare(
        environment,
        lora_options={"r": 2, "lora_alpha": 4, "lora_dropout": 0.25, "use_dora": False},
    )
    assert record["status"] == "prepared"
    plan = _save(plans, lora_options={"r": 2, "lora_alpha": 4, "lora_dropout": 0.25})
    assert plan["status"] == "ready"


@pytest.mark.parametrize("dropout", [0, 1])
def test_dropout_boundaries_remain_valid(environment, plans, dropout):
    record = _prepare(environment, lora_options={"lora_dropout": dropout})
    assert record["status"] == "prepared"
    plan = _save(plans, lora_options={"r": 2, "lora_alpha": 4, "lora_dropout": dropout})
    assert plan["status"] == "ready"


def test_prepared_run_with_valid_existing_configuration_remains_startable(environment):
    _, session, service, _ = environment
    prepared = _prepare(environment, lora_options={"r": 2, "lora_alpha": 4, "lora_dropout": 0.1})
    assert service.get_status(prepared["run_id"])["config"] == prepared["config"]
    with patch("src.tracking.runner.TrainingRunner.launch_training", return_value="stub") as launch:
        started = service.start(prepared["run_id"], session)
    assert started["status"] == "running"
    launch.assert_called_once()
