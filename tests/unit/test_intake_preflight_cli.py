"""CLI preflight consumes a saved local tokenizer without model downloads."""

import json

from tests.unit import test_intake_training_preflight as preflight_fixtures
from tests.unit.test_full_data_cli import invoke

exported = preflight_fixtures.exported
tokenizer = preflight_fixtures.tokenizer


def test_cli_preflight_records_real_answer_loss_and_uses_local_tokenizer(
    exported, tokenizer, tmp_path
):
    service, session = exported
    directory = tmp_path / "tokenizer"
    tokenizer.save_pretrained(directory)
    result = invoke(
        service,
        "preflight",
        session.session_id,
        "--revision",
        session.revision,
        "--tokenizer",
        directory,
        "--max-length",
        6,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)["training_preflight"]
    assert report["status"] == "blocked"
    assert any(issue["code"] == "answer_lost" and issue["row_ids"] for issue in report["issues"])
    assert all(row["was_truncated"] for row in report["rows"])
    assert "未启动训练" in result.stderr
    assert service.load(session.session_id).training_preflight == report


def test_preflight_cli_prints_plain_language_advice(tmp_path, capsys):
    """CLI 预检输出包含大白话建议(有截断时给具体长度建议)。"""
    import sys

    from scripts import data_intake
    from src.workbench.intake_service import IntakeService
    from tests.unit.test_data_materialize import _full

    service = IntakeService(tmp_path / "intake")
    session = _full(service)
    session = service.materialize_dataset(session.session_id, session.revision)

    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(
        models.WordLevel({"[UNK]": 0, "[PAD]": 1, "[EOS]": 2, "yes": 3}, unk_token="[UNK]")
    )
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]"
    )
    model_dir = tmp_path / "tok"
    tokenizer.save_pretrained(model_dir)

    monkeypatched = sys.argv
    sys.argv = [
        "data_intake.py",
        "--store",
        str(service.root),
        "preflight",
        session.session_id,
        "--revision",
        str(session.revision),
        "--tokenizer",
        str(model_dir),
        "--max-length",
        "16",
    ]
    try:
        code = data_intake.main()
    finally:
        sys.argv = monkeypatched
    assert code == 0
    err = capsys.readouterr().err
    assert "训练前检查" in err
    # 截断场景必然出现具体建议(长度或答案说明)
    assert ("最长一条记录需要" in err) or ("没有内容因长度超限被截断" in err) or ("答案" in err)


def test_cli_preflight_passed_tail_points_forward_not_back_to_preflight(
    exported, tokenizer, tmp_path
):
    """预检通过后的 CLI 尾行不再回环指向 preflight(R94 出口链断点修复):
    下一步状态转为 preflight_passed,短语点名 plan-recommend 与 train-prepare
    两个真实出口;此前尾行让用户重跑刚完成的 preflight,旅程在此死循环。"""
    service, session = exported
    directory = tmp_path / "tokenizer"
    tokenizer.save_pretrained(directory)
    result = invoke(
        service,
        "preflight",
        session.session_id,
        "--revision",
        session.revision,
        "--tokenizer",
        directory,
        "--max-length",
        "32",
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["training_preflight"]["status"] == "passed"
    assert "下一步状态: preflight_passed（训练前检查已通过：可运行 plan-recommend" in result.stderr
    assert "train-prepare" in result.stderr
    # 回环断言:通过态尾行不再出现「可运行 preflight 做训练前检查」的旧出口句
    assert "可运行 preflight 做训练前检查" not in result.stderr
