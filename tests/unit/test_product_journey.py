"""M2 旅程串联:零密钥新用户从建任务到训练产物的完整闭环,一段不断。

走真实服务 API(页面调用的同一路径):创建 → 基础分析(无 Agent)→ 对比核验 →
样例确认 → 全量验证/确认 → 盲标核验 → 物化 → 训练准备(门禁通过)→ 启动 → 真实产物。
旅程中任何一段断裂,本测试即失败——它就是 M2 的断点定位器。
"""

import json
import shutil
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("datasets")
pytest.importorskip("peft")

from src.workbench.baseline_analysis import propose_baseline_analysis
from src.workbench.intake_service import IntakeService
from src.workbench.training_runs import TrainingRunService
from tests.unit.test_data_intake import CSV

FULL_TICKETS = (
    "编号,客户描述,类别,处理结果\n"
    "001,杯子破损,质量,补发\n002,物流未更新,物流,补发\n"
    "003,屏幕碎裂,质量,换新\n004,快递丢失,物流,赔付\n"
    "005,开不了机,质量,检修\n006,地址填错,物流,改派\n"
    "007,异味,质量,退货\n008,延迟送达,物流,补偿\n"
    "009,无法充电,质量,检修\n010,包装破损,物流,补发\n"
).encode()


@pytest.fixture()
def journey(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parents[2]))
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    model = tmp_path / "tiny-model"
    backend = Tokenizer(models.WordLevel({"[UNK]": 0, "[PAD]": 1, "[BOS]": 2, "[EOS]": 3}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    backend.post_processor = processors.TemplateProcessing(
        single="[BOS] $A [EOS]", special_tokens=[("[BOS]", 2), ("[EOS]", 3)]
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]",
        bos_token="[BOS]", eos_token="[EOS]",
    )
    tokenizer.save_pretrained(model)
    GPT2LMHeadModel(
        GPT2Config(
            vocab_size=4, n_layer=1, n_head=1, n_embd=8, n_positions=64,
            bos_token_id=2, eos_token_id=3, pad_token_id=1,
        )
    ).save_pretrained(model)
    intake = IntakeService(tmp_path / "intake")
    project = tmp_path / "project"
    (project / "scripts").mkdir(parents=True)
    shutil.copy(
        Path(__file__).resolve().parents[2] / "scripts" / "workbench_train.py",
        project / "scripts/workbench_train.py",
    )
    training = TrainingRunService(
        tmp_path / "runs", project_root=project, python_executable=sys.executable
    )
    return intake, training, model


def test_zero_agent_journey_from_goal_to_trained_adapter(journey):
    intake, training, model = journey

    # ① 建任务:业务目标 + 样例(零密钥,无任何 Agent 调用)
    session = intake.create("根据客户首次咨询判断售后问题类型", "工单.csv", CSV)
    # ② 基础分析:产品内置判断,不需要 Agent
    session = intake.apply_analysis(
        session,
        propose_baseline_analysis(
            session, target_column="类别", group_columns=["编号"], excluded_columns=["处理结果"],
        ),
        model="baseline-deterministic",
    )
    assert session.preview is not None
    # ③ 对比核验:配对正确才继续
    pending = intake.start_contrast_check(session.session_id, session.revision)
    targets = {row.row_id: row.target for row in session.preview.rows}
    intake.submit_contrast_check(
        session.session_id,
        pending["check_id"],
        {item["row_id"]: targets[item["row_id"]] for item in pending["items"]},
    )
    assert intake.contrast_check_status(session.session_id)["verdict"] == "verified"
    # ④ 样例确认
    session = intake.confirm(session.session_id, session.revision)
    # ⑤ 全量验证 + 确认
    session = intake.validate_full_data(
        session.session_id, session.revision, "全量工单.csv", FULL_TICKETS
    )
    session = intake.confirm_full_data(session.session_id, session.revision)
    # ⑥ 盲标核验:隐藏答案独立作答,全一致(题目抽自全量预览)
    pending = intake.start_label_verification(session.session_id, session.revision)
    full_targets = {row.row_id: row.target for row in session.full_data.preview.rows}
    answers = {item["row_id"]: full_targets[item["row_id"]] for item in pending["items"]}
    verdict = intake.submit_label_verification(
        session.session_id, pending["verification_id"], answers
    )
    assert verdict["verdict"] == "verified"
    # ⑦ 物化分区
    session = intake.materialize_dataset(session.session_id, session.revision)
    assert session.dataset is not None and session.label_verification["verdict"] == "verified"
    # ⑧ 训练准备:盲标门禁放行,真实预检通过
    record = training.prepare(
        session,
        model,
        max_length=32,
        training_options={
            "num_epochs": 1, "batch_size": 2, "gradient_accumulation_steps": 1,
            "gradient_checkpointing": False, "logging_steps": 1,
        },
    )
    assert record["status"] == "prepared", record["issues"]
    # ⑨ 启动并等待真实产物
    training.start(record["run_id"], session)
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        result = training.get_status(record["run_id"])
        if result["status"] in {"succeeded", "failed", "stopped"}:
            break
        time.sleep(0.2)
    assert result["status"] == "succeeded", training.read_logs(record["run_id"], tail=50)
    manifest = json.loads(Path(result["artifacts"]["manifest"]).read_text())
    assert manifest["dataset_version"] == session.dataset.version
    # ⑩ 文档-产品漂移守卫：试用剧本引用的盲标核验文案必须与服务层真实文案逐字一致。
    trial_log = (
        Path(__file__).resolve().parents[2] / "docs" / "validation" / "user-trial-log.md"
    ).read_text(encoding="utf-8")
    marker = "盲标核验 verdict_note:"
    anchored = [line for line in trial_log.splitlines() if marker in line]
    assert anchored, "试用文档缺少「产品原文锚点」中的盲标核验 verdict_note 行"
    assert anchored[0].split(marker, 1)[1].strip() == verdict["verdict_note"], (
        "试用文档与产品文案漂移：请同步更新 docs/validation/user-trial-log.md 的锚点行"
    )
