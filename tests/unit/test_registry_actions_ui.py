"""Registry changes preserve model identity and report freshly observed state."""

from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("mlflow")
from mlflow.tracking import MlflowClient
from streamlit.testing.v1 import AppTest

from ui.registry_actions import change_alias

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "ui/pages/04_Registry.py"


@pytest.fixture
def registry(tmp_path):
    uri = f"sqlite:///{tmp_path / 'registry.db'}"
    client = MlflowClient(tracking_uri=uri)
    for name in ("first-model", "other-model"):
        client.create_registered_model(name)
        for version in (1, 2, 3):
            client.create_model_version(name, source=f"file:///not-loaded/{name}/{version}")
        client.set_registered_model_alias(name, "champion", "2")
    return uri, client


def test_real_alias_assignment_removal_and_other_model_isolation(registry):
    uri, client = registry
    result = change_alias(uri, "first-model", "champion", "2", "3")
    assert result.level == "success"
    assert "v2 → v3" in result.message
    assert str(client.get_model_version_by_alias("first-model", "champion").version) == "3"
    assert str(client.get_model_version_by_alias("other-model", "champion").version) == "2"
    result = change_alias(uri, "first-model", "champion", "3", None)
    assert result.level == "success"
    assert "v3 → 未绑定" in result.message
    assert "champion" not in client.get_registered_model("first-model").aliases
    assert {str(v.version) for v in client.search_model_versions("name='first-model'")} == {
        "1",
        "2",
        "3",
    }


def test_real_stale_binding_refuses_set_and_remove(registry):
    uri, client = registry
    client.set_registered_model_alias("first-model", "champion", "1")
    for target in ("3", None):
        result = change_alias(uri, "first-model", "champion", "2", target)
        assert result.level == "warning"
        assert "v2 变为 v1" in result.message and "未提交" in result.message
        assert str(client.get_model_version_by_alias("first-model", "champion").version) == "1"


def test_real_missing_target_and_model_do_not_change_binding(registry):
    uri, client = registry
    result = change_alias(uri, "first-model", "champion", "2", "99")
    assert result.level == "error" and "未提交" in result.message
    assert str(client.get_model_version_by_alias("first-model", "champion").version) == "2"
    result = change_alias(uri, "missing-model", "champion", None, "3")
    assert result.level == "error" and "未提交" in result.message


def test_query_uses_registered_model_bindings_and_updates_after_cache_clear(registry):
    from ui.queries import fetch_model_versions

    uri, client = registry
    client.set_registered_model_alias("first-model", "challenger", "1")
    client.set_registered_model_alias("first-model", "pre-champion", "3")
    fetch_model_versions.clear()
    frame = fetch_model_versions(uri, "first-model")
    assert set(frame["name"]) == {"first-model"}
    assert dict(zip(frame["version"], frame["aliases"])) == {
        1: "challenger",
        2: "champion",
        3: "pre-champion",
    }
    client.set_registered_model_alias("first-model", "champion", "3")
    fetch_model_versions.clear()
    frame = fetch_model_versions(uri)
    first = frame[frame["name"] == "first-model"].set_index("version")
    other = frame[frame["name"] == "other-model"].set_index("version")
    assert first.loc[3, "aliases"] == "champion, pre-champion"
    assert first.loc[2, "aliases"] == ""
    assert other.loc[2, "aliases"] == "champion"


def test_query_does_not_hide_alias_read_failure_as_unbound(monkeypatch, registry):
    from ui.queries import fetch_model_versions

    uri, client = registry

    def unreadable(*_):
        raise RuntimeError("binding read denied")

    monkeypatch.setattr(client, "get_registered_model", unreadable)
    monkeypatch.setattr("mlflow.tracking.MlflowClient", lambda **_: client)
    fetch_model_versions.clear()
    with pytest.raises(RuntimeError, match="binding read denied"):
        fetch_model_versions(uri)


@pytest.mark.parametrize("current", [None, "2"])
def test_noop_does_not_write(monkeypatch, current):
    client = SimpleNamespace(
        get_registered_model=lambda _: SimpleNamespace(aliases={"champion": current})
    )
    monkeypatch.setattr("mlflow.tracking.MlflowClient", lambda **_: client)
    result = change_alias("test-uri", "model", "champion", current, current)
    assert result.level == "warning" and "无需提交" in result.message


@pytest.mark.parametrize("failure", ["initial", "write", "readback", "different"])
def test_failed_or_unconfirmed_actions_never_report_success(monkeypatch, failure):
    reads = []
    writes = []

    def read(name):
        reads.append(name)
        if failure == "initial" or (failure == "readback" and len(reads) > 1):
            raise RuntimeError("store unavailable")
        return SimpleNamespace(aliases={"champion": "2" if len(reads) == 1 else "1"})

    def write(**kwargs):
        writes.append(kwargs)
        if failure == "write":
            raise RuntimeError("write unavailable")

    client = SimpleNamespace(
        get_registered_model=read,
        get_model_version=lambda *_: None,
        set_registered_model_alias=write,
    )
    monkeypatch.setattr("mlflow.tracking.MlflowClient", lambda **_: client)
    result = change_alias("test-uri", "model", "champion", "2", "3")
    assert result.level != "success"
    if failure == "initial":
        assert not writes and "未提交" in result.message
    elif failure == "write":
        assert "未确认" in result.message
    elif failure == "readback":
        assert "已提交" in result.message and "无法读取结果" in result.message
    else:
        assert "v1" in result.message and "目标 v3 不同" in result.message


def boot(monkeypatch, registry, *, display=False):
    from ui import config, queries, runtime

    uri, _ = registry
    queries.fetch_model_versions.clear()
    monkeypatch.setattr(config, "MLFLOW_TRACKING_URI", uri)
    monkeypatch.setattr(runtime, "DISPLAY_ONLY", display)
    app = AppTest.from_file(str(PAGE), default_timeout=30).run()
    if display:
        app.segmented_control(key="registry-source").set_value("本地注册版本").run()
    assert not app.exception
    return app


def action(app, label):
    return next(button for button in app.button if button.label == label)


def test_ui_remove_names_current_binding_independently_of_target_and_keeps_receipt(
    monkeypatch, registry
):
    _, client = registry
    app = boot(monkeypatch, registry)
    assert app.selectbox(key="alias_version").value == "3"
    assert any("champion → v2" in item.value for item in app.markdown)
    assert any("v2 → 未绑定" in item.value for item in app.caption)
    action(app, "🗑 移除").click().run()
    assert not app.exception
    assert any("v2 → 未绑定" in item.value for item in app.success)
    assert "champion" not in client.get_registered_model("first-model").aliases
    assert str(client.get_model_version_by_alias("other-model", "champion").version) == "2"
    assert action(app, "🗑 移除").disabled
    app.run()
    assert any("v2 → 未绑定" in item.value for item in app.success)


def test_ui_set_refresh_and_stale_change_guard(monkeypatch, registry):
    _, client = registry
    app = boot(monkeypatch, registry)
    action(app, "🏷 设置").click().run()
    assert not app.exception
    assert any("v2 → v3" in item.value for item in app.success)
    assert action(app, "🏷 设置").disabled
    # A different client edits the registry after this page was displayed.
    client.set_registered_model_alias("first-model", "champion", "1")
    action(app, "🗑 移除").click().run()
    assert not app.exception
    assert str(client.get_model_version_by_alias("first-model", "champion").version) == "1"
    assert any("未提交" in item.value for item in app.warning)
    assert any("champion → v1" in item.value for item in app.markdown)
    client.set_registered_model_alias("first-model", "champion", "2")
    app.button(key="registry-refresh").click().run()
    assert any("champion → v2" in item.value for item in app.markdown)


def test_display_forged_actions_cannot_write_but_refresh_remains_available(monkeypatch, registry):
    _, client = registry
    app = boot(monkeypatch, registry, display=True)
    assert action(app, "🏷 设置").disabled
    assert action(app, "🗑 移除").disabled
    assert not app.button(key="registry-refresh").disabled
    for label in ("🏷 设置", "🗑 移除"):
        action(app, label).click().run()
        assert not app.exception
        assert str(client.get_model_version_by_alias("first-model", "champion").version) == "2"
