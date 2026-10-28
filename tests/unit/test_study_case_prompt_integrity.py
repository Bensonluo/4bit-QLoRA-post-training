"""Each displayed R148 business case must bind to every real recorded prompt."""

import json
import shutil
from pathlib import Path

import pytest

from ui.showcase import R149_SOURCE, available_history_catalog
from ui.study_archive import ARCHIVE, PROTOCOLS, SOURCE, load_study

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def assets(tmp_path):
    target = tmp_path / "docs/validation"
    target.mkdir(parents=True)
    for name in (SOURCE, ARCHIVE, R149_SOURCE):
        shutil.copyfile(ROOT / "docs/validation" / name, target / name)
    return tmp_path


def edit(root, mutate):
    path = root / "docs/validation" / SOURCE
    data = json.loads(path.read_text())
    mutate(data)
    path.write_text(json.dumps(data, ensure_ascii=False))


@pytest.mark.parametrize("study", ["r148-controlled-v2", "r148-product-smoke"])
@pytest.mark.parametrize("field", ["input", "instruction", "form"])
def test_changed_shared_business_case_is_rejected_without_fabricating_a_prompt(
    assets, study, field
):
    def mutate(data):
        case = data["controlled_v2"]["data"]["development"][0]
        case[field] = "json" if field == "form" else "未经模型实际读取的内容"

    edit(assets, mutate)
    with pytest.raises(ValueError, match="业务输入"):
        load_study(assets, study)
    entries, errors = available_history_catalog(assets)
    assert any(item["id"] == "r149" for item in entries)
    assert not errors


@pytest.mark.parametrize("protocol", [PROTOCOLS[0], PROTOCOLS[2]])
def test_protocol_and_baseline_prompts_must_match_actual_shared_case(assets, protocol):
    def mutate(data):
        results = data["controlled_v2"]["results"]
        baseline = results["baseline"][protocol]
        baseline["rows"][0]["prompt"] = baseline["rows"][0]["prompt"].replace(
            "ticket: TK-9000", "ticket: INVENTED"
        )
        variants = PROTOCOLS[:2] if protocol == PROTOCOLS[0] else [PROTOCOLS[2]]
        for name in variants:
            row = results["arms"][name]["evaluation"]["rows"][0]
            row["prompt"] = row["prompt"].replace("ticket: TK-9000", "ticket: INVENTED")

    edit(assets, mutate)
    # Matching baseline and trained prompts alone are insufficient proof.
    with pytest.raises(ValueError, match="业务输入或任务要求"):
        load_study(assets, "r148-controlled-v2")
    # The independently recorded product remains readable.
    assert load_study(assets, "r148-product-smoke")["id"] == "r148-product-smoke"


def test_native_prompt_cannot_append_an_unrecorded_assistant_prefix(assets):
    def mutate(data):
        results = data["controlled_v2"]["results"]
        rows = [
            results["baseline"][PROTOCOLS[2]]["rows"][0],
            results["arms"][PROTOCOLS[2]]["evaluation"]["rows"][0],
        ]
        for row in rows:
            row["prompt"] += "unrecorded assistant content"

    edit(assets, mutate)
    with pytest.raises(ValueError, match="业务输入或任务要求"):
        load_study(assets, "r148-controlled-v2")
    assert load_study(assets, "r148-product-smoke")["id"] == "r148-product-smoke"


@pytest.mark.parametrize("study", ["r148-controlled-v2", "r148-product-smoke"])
@pytest.mark.parametrize("group", [None, "", "  "])
def test_missing_group_does_not_hide_verified_input_or_invent_group(assets, study, group):
    def mutate(data):
        case = data["controlled_v2"]["data"]["development"][0]
        if group is None:
            del case["group_id"]
        else:
            case["group_id"] = group

    edit(assets, mutate)
    result = load_study(assets, study)
    assert result["cases"][0].get("group_id") == group
    assert result["cases"][0]["input"].startswith("ticket: TK-9000")
    catalog, _ = available_history_catalog(assets)
    assert (
        next(item for item in catalog if item["id"] == study)["counts"]["development_groups"]
        is None
    )
