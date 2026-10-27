"""A fresh named deep link must execute the common Streamlit entry point first."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
ROOT = Path(__file__).resolve().parents[2]

# Each invocation has a new interpreter: no prior st.navigation call can disable
# Streamlit's legacy pages-directory detection and conceal the startup regression.
COLD_PROCESS = r"""
import json
import sys
from dataclasses import replace
from pathlib import Path

app, route, local_root, assets = map(Path, [sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]])
route = str(route)
sys.path.insert(0, str(app.parent.parent))
sys.path.insert(1, str(assets))
from streamlit.proto.ClientState_pb2 import ClientState
from streamlit.testing.v1 import AppTest
from streamlit.testing.v1.local_script_runner import LocalScriptRunner
from ui import config, queries, runtime
from ui.components import workspace

config.PROJECT_ROOT = local_root
config.CONFIGS_DIR = local_root / "configs"
config.MLFLOW_TRACKING_URI = "file://" + str(local_root / "mlruns")
workspace.SHOWCASE_ROOT = assets
runtime.DISPLAY_ONLY = True
sys.modules["mlflow"] = None
queries_called = []
def forbidden_query(*args, **kwargs):
    queries_called.append(True)
    raise AssertionError("Cold public deep links must not query a local MLflow store")
for name in ("fetch_runs", "fetch_experiments", "fetch_model_versions"):
    setattr(queries, name, forbidden_query)

client_state = ClientState(page_name=route)
original_request = LocalScriptRunner.request_rerun
requests = []
def named_first_request(self, rerun_data):
    requests.append(client_state.page_name)
    return original_request(self, replace(
        rerun_data,
        page_name=client_state.page_name,
        page_script_hash=client_state.page_script_hash,
    ))
LocalScriptRunner.request_rerun = named_first_request
original_messages = LocalScriptRunner.forward_msgs
messages = []
def capture_messages(self):
    result = original_messages(self)
    messages.extend(result)
    return result
LocalScriptRunner.forward_msgs = capture_messages

page = AppTest.from_file(str(app), default_timeout=45).run()
navigation = [msg.navigation for msg in messages if msg.WhichOneof("type") == "navigation"]
logos = [msg.logo.image for msg in messages if msg.WhichOneof("type") == "logo"]
html = [
    msg.delta.new_element.html.body
    for msg in messages
    if msg.WhichOneof("type") == "delta"
    and msg.delta.WhichOneof("type") == "new_element"
    and msg.delta.new_element.WhichOneof("type") == "html"
]
result = {
    "route": route,
    "requested_pages": requests,
    "exceptions": [entry.message for entry in page.exception],
    "errors": [entry.value for entry in page.error],
    "titles": [entry.value for entry in page.title],
    "sidebar_captions": [entry.value for entry in page.sidebar.caption],
    "sidebar_expanders": [entry.label for entry in page.sidebar.expander],
    "logo": bool(logos and all(logos)),
    "shared_css": any((assets / "ui/styles.css").read_text() in body for body in html),
    "navigation_sections": list(navigation[-1].sections) if navigation else [],
    "navigation_titles": [entry.page_name for entry in navigation[-1].app_pages] if navigation else [],
    "navigation_routes": [entry.url_pathname for entry in navigation[-1].app_pages] if navigation else [],
    "selected_route": next((entry.url_pathname for entry in navigation[-1].app_pages
        if entry.page_script_hash == navigation[-1].page_script_hash), None) if navigation else None,
    "registry_sources": [entry.value for entry in page.segmented_control if entry.key == "registry-source"],
    "queries_called": len(queries_called),
    "local_files": [str(path.relative_to(local_root)) for path in local_root.rglob("*")],
}
print("COLD_RESULT=" + json.dumps(result, ensure_ascii=False))
"""


def cold_result(route, tmp_path, *, app=None):
    environment = dict(os.environ)
    environment["TUNESMITH_DISPLAY_ONLY"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    process = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            COLD_PROCESS,
            str(app or ROOT / "ui/app.py"),
            route,
            str(tmp_path),
            str(ROOT),
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert process.returncode == 0, process.stderr + process.stdout
    lines = [line for line in process.stdout.splitlines() if line.startswith("COLD_RESULT=")]
    assert len(lines) == 1, process.stderr + process.stdout
    return json.loads(lines[0].removeprefix("COLD_RESULT="))


@pytest.mark.parametrize(
    "route,title",
    [
        ("Registry", "模型注册表"),
        ("Showcase", "示例与历史"),
        ("Training_Lab", "训练实验室"),
    ],
)
def test_fresh_deep_link_renders_requested_page_and_shared_shell(tmp_path, route, title):
    result = cold_result(route, tmp_path)
    assert result["requested_pages"] and result["requested_pages"][0] == route
    assert result["exceptions"] == []
    assert result["errors"] == []
    assert result["selected_route"] == route
    assert any(title in actual for actual in result["titles"])
    assert result["logo"], "Cold deep links must not bypass the shared app logo."
    assert result["shared_css"], "Cold deep links must load the same shell stylesheet."
    assert {"业务任务", "训练与证据", "模型管理", "数据工具"}.issubset(
        result["navigation_sections"]
    )
    assert {"任务总览", "模型与版本", "示例与历史", "训练实验室"}.issubset(
        result["navigation_titles"]
    )
    assert "云端展示环境" in result["sidebar_captions"]
    assert "运行环境" in result["sidebar_expanders"]
    if route == "Registry":
        assert result["registry_sources"] == ["公开交付档案"]
    assert result["queries_called"] == 0
    assert result["local_files"] == []
