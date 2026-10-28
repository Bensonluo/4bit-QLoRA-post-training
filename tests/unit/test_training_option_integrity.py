"""Direct preparation and reviewed plans enforce the same training option contract."""

import pytest

from tests.unit.test_training_plans import _save

pytest_plugins = ["tests.unit.test_workbench_training_runs", "tests.unit.test_training_plans"]


@pytest.mark.parametrize(
    "option",
    [
        {"gradient_checkpointing": "false"},
        {"fp16": 0},
        {"bf16": 0},
        {"weight_decay": True},
        {"weight_decay": -0.1},
        {"weight_decay": float("nan")},
        {"weight_decay": float("inf")},
        {"weight_decay": 10**400},
        {"max_grad_norm": True},
        {"max_grad_norm": -1},
        {"max_grad_norm": float("nan")},
        {"max_grad_norm": float("inf")},
        {"logging_steps": 0},
        {"logging_steps": True},
        {"logging_steps": 0.5},
        {"save_steps": -1},
        {"save_steps": True},
        {"eval_steps": 0},
        {"eval_steps": float("inf")},
    ],
)
def test_invalid_training_options_cannot_be_prepared_or_saved(environment, plans, option):
    _, session, service, model = environment
    prepared = service.prepare(session, model, max_length=32, training_options=option)
    assert prepared["status"] == "blocked"
    assert prepared["config"] is None
    assert any(issue["severity"] == "blocking" for issue in prepared["issues"])
    with pytest.raises((ValueError, TypeError)):
        _save(plans, training_options={"num_epochs": 1, "batch_size": 1, **option})


def test_zero_regularization_and_disabled_clipping_are_valid(environment, plans):
    _, session, service, model = environment
    options = {"weight_decay": 0, "max_grad_norm": 0, "logging_steps": 1}
    prepared = service.prepare(session, model, max_length=32, training_options=options)
    assert prepared["status"] == "prepared"
    assert prepared["config"]["training"]["max_grad_norm"] == 0
    plan = _save(plans, training_options={"num_epochs": 1, "batch_size": 1, **options})
    assert plan["status"] == "ready"


def test_incompatible_save_and_eval_intervals_are_blocked(environment, plans):
    _, session, service, model = environment
    options = {"save_steps": 3, "eval_steps": 2}
    prepared = service.prepare(session, model, max_length=32, training_options=options)
    assert prepared["status"] == "blocked"
    assert any("整数倍" in issue["message"] for issue in prepared["issues"])
    with pytest.raises(ValueError, match="整数倍"):
        _save(plans, training_options={"num_epochs": 1, "batch_size": 1, **options})


def test_matching_save_and_eval_intervals_remain_valid(environment, plans):
    _, session, service, model = environment
    options = {"save_steps": 4, "eval_steps": 2}
    assert (
        service.prepare(session, model, max_length=32, training_options=options)["status"]
        == "prepared"
    )
    assert (
        _save(plans, training_options={"num_epochs": 1, "batch_size": 1, **options})["status"]
        == "ready"
    )
