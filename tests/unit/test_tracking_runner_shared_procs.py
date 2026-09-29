"""跨实例进程注册表(R104):Streamlit 每次重跑都新建 TrainingRunner
(ui/pages/00_Training_Lab.py Activity 标签),实例级 _active 字典在页面
第一次重跑后即丢失 Popen 句柄,造成两个实测缺陷:

① 退出码永远无人记录——训练进程死后 get_status 走「pid 不存活 → unknown」,
   页面显示 ⚪ 而非 ✅/🔴,且 finished 门控的「🧭 下一步」面板永不出现;
   30s 自动刷新会把 running→unknown 这个坏迁移实时播给用户看。
② Stop 跨实例静默失效——stop_training 对 proc is None 直接 no-op,
   Activity 的 ⏹ Stop 按钮在页面重跑一次后点了没反应。

修法:Popen 句柄按 project_root 存进程级注册表,实例 _active 只是视图。
这些钉用真实子进程 + SCRIPTS 重定向到受控哑脚本,走完整的 launch →
poll → record 退出链路(不伪造 _active,不 mock Popen)。
"""

import time
from pathlib import Path

from src.tracking import runner as runner_module
from src.tracking.runner import TrainingRunner


def _write_dummy_script(root: Path, body: str) -> None:
    script = root / "scripts" / "_dummy_job.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(body, encoding="utf-8")


def test_new_instance_sees_finished_state(tmp_path, monkeypatch):
    """实例 A 启动的训练,退出码必须能被后续任何实例记录并读到 finished。"""
    _write_dummy_script(tmp_path, "import time; time.sleep(0.6)\n")
    monkeypatch.setattr(runner_module, "SCRIPTS", {"sft": "scripts/_dummy_job.py"})
    launcher = TrainingRunner(project_root=str(tmp_path))
    rid = launcher.launch_training(technique="sft", config_dict={}, run_name="job-ok")

    observer = TrainingRunner(project_root=str(tmp_path))
    assert observer.get_status(rid) == "running"
    for _ in range(100):
        if observer.get_status(rid) != "running":
            break
        time.sleep(0.1)
    assert observer.get_status(rid) == "finished", (
        "新实例必须观察到退出码 0 → finished(而非 unknown)"
    )
    assert observer.get_run_info(rid)["returncode"] == 0


def test_new_instance_sees_failed_state(tmp_path, monkeypatch):
    """非零退出码同样必须跨实例落盘:failed 而非 unknown。"""
    _write_dummy_script(tmp_path, "import sys; sys.exit(3)\n")
    monkeypatch.setattr(runner_module, "SCRIPTS", {"sft": "scripts/_dummy_job.py"})
    launcher = TrainingRunner(project_root=str(tmp_path))
    rid = launcher.launch_training(technique="sft", config_dict={}, run_name="job-bad")

    observer = TrainingRunner(project_root=str(tmp_path))
    for _ in range(100):
        if observer.get_status(rid) != "running":
            break
        time.sleep(0.1)
    assert observer.get_status(rid) == "failed", "新实例必须观察到退出码 3 → failed(而非 unknown)"
    assert observer.get_run_info(rid)["returncode"] == 3


def test_new_instance_can_stop_launched_run(tmp_path, monkeypatch):
    """Stop 跨实例必须真实生效:实例 B terminate 实例 A 启动的进程,
    而不是静默 no-op 让它跑完。"""
    _write_dummy_script(tmp_path, "import time; time.sleep(30)\n")
    monkeypatch.setattr(runner_module, "SCRIPTS", {"sft": "scripts/_dummy_job.py"})
    launcher = TrainingRunner(project_root=str(tmp_path))
    rid = launcher.launch_training(technique="sft", config_dict={}, run_name="job-long")

    observer = TrainingRunner(project_root=str(tmp_path))
    observer.stop_training(rid)
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and observer.get_status(rid) == "running":
        time.sleep(0.1)
    assert observer.get_status(rid) != "running", "stop_training 在新实例上必须真正终止子进程"
    assert observer.get_run_info(rid)["returncode"] != 0
