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
