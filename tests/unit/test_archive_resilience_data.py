"""One unavailable public study must not hide independent, healthy records."""

import json
import shutil
from pathlib import Path

import pytest

from ui import showcase

ROOT = Path(__file__).resolve().parents[2]
IDS = ("r149", "r148-controlled-v2", "r148-product-smoke")


@pytest.fixture
def assets(tmp_path):
    directory = tmp_path / "docs/validation"
    directory.mkdir(parents=True)
    for name in (showcase.R149_SOURCE, showcase.R148_SOURCE):
        shutil.copyfile(ROOT / "docs/validation" / name, directory / name)
    return tmp_path


def edit_r148(root, change):
    path = root / "docs/validation" / showcase.R148_SOURCE
    data = json.loads(path.read_bytes())
    change(data)
    path.write_text(json.dumps(data, ensure_ascii=False))


def assert_available(root, expected_ids, error_ids, *, group_counts=None):
    entries, errors = showcase.available_history_catalog(root)
    original = {entry["id"]: entry for entry in showcase.history_catalog(ROOT)}
    assert [entry["id"] for entry in entries] == expected_ids
    # Rewritten shared JSON has a new source hash; every other healthy field
    # must still be exactly the original catalog record.
    for entry in entries:
        expected = dict(original[entry["id"]])
        expected["source"] = entry["source"]
        if group_counts and entry["id"] in group_counts:
            expected["counts"] = {
                **expected["counts"],
                "development_groups": group_counts[entry["id"]],
            }
        assert entry == expected
        source = root / "docs/validation" / entry["sourcefile"]
        assert entry["source"]["sha256"] == showcase._read(root, source.name)[1]["sha256"]
    assert [error["id"] for error in errors] == error_ids
    assert all(set(error) == {"id", "title", "sourcefile", "message"} for error in errors)
    assert all(error["title"] and error["message"] for error in errors)
    assert all(
        error["sourcefile"]
        == (showcase.R149_SOURCE if error["id"] == "r149" else showcase.R148_SOURCE)
        for error in errors
    )
    assert str(root) not in json.dumps(errors, ensure_ascii=False)
    if error_ids:
        with pytest.raises(ValueError):
            showcase.history_catalog(root)
    return entries, errors


def test_complete_assets_preserve_the_strict_catalog_exactly(assets):
    entries, errors = assert_available(assets, list(IDS), [])
    assert entries == showcase.history_catalog(assets) == showcase.history_catalog(ROOT)
    assert errors == []


@pytest.mark.parametrize("broken", ["missing", "invalid_json", "invalid_report"])
def test_r149_failure_keeps_both_r148_studies(assets, broken):
    path = assets / "docs/validation" / showcase.R149_SOURCE
    if broken == "missing":
        path.unlink()
    elif broken == "invalid_json":
        path.write_text("{broken")
    else:
        data = json.loads(path.read_bytes())
        data["actual_report"]["models"][0]["rows"].pop()
        path.write_text(json.dumps(data))
    assert_available(assets, list(IDS[1:]), [IDS[0]])


@pytest.mark.parametrize("broken", ["missing", "invalid_json", "invalid_date"])
def test_unreadable_shared_r148_source_reports_both_studies(assets, broken):
    path = assets / "docs/validation" / showcase.R148_SOURCE
    if broken == "missing":
        path.unlink()
    elif broken == "invalid_json":
        path.write_text("not-json")
    else:
        data = json.loads(path.read_bytes())
        del data["date"]
        path.write_text(json.dumps(data))
    assert_available(assets, [IDS[0]], list(IDS[1:]))


@pytest.mark.parametrize("broken", ["missing_study", "missing_arm", "wrong_summary"])
def test_controlled_study_failure_keeps_independent_product_smoke(assets, broken):
    def change(data):
        if broken == "missing_study":
            del data["controlled_v2"]
        elif broken == "missing_arm":
            data["controlled_v2"]["results"]["arms"].popitem()
        else:
            arm = next(iter(data["controlled_v2"]["results"]["arms"].values()))
            arm["evaluation"]["summary"]["all"]["accepted"] += 1

    edit_r148(assets, change)
    assert_available(
        assets,
        [IDS[0], IDS[2]],
        [IDS[1]],
        group_counts={IDS[2]: None} if broken == "missing_study" else None,
    )


@pytest.mark.parametrize("broken", ["missing_study", "missing_model", "wrong_summary"])
def test_product_smoke_failure_keeps_controlled_study(assets, broken):
    def change(data):
        if broken == "missing_study":
            del data["product_smoke"]
        elif broken == "missing_model":
            del data["product_smoke"]["results"]["evaluations"]["trained"]
        else:
            data["product_smoke"]["results"]["evaluations"]["trained"]["summary"]["all"][
                "accepted"
            ] += 1

    edit_r148(assets, change)
    assert_available(assets, [IDS[0], IDS[1]], [IDS[2]])


def test_all_missing_sources_return_no_fake_entries_and_three_identified_errors(tmp_path):
    entries, errors = assert_available(tmp_path, [], list(IDS))
    assert entries == []
    assert len(errors) == len(IDS)
    assert list(tmp_path.iterdir()) == []


def test_all_invalid_records_report_each_failure_independently(assets):
    (assets / "docs/validation" / showcase.R149_SOURCE).write_text("{}")
    edit_r148(assets, lambda data: data.update(controlled_v2={}, product_smoke={}))
    assert_available(assets, [], list(IDS))


def test_underlying_private_exception_details_are_not_public_errors(assets, monkeypatch):
    def unreadable(*args):
        raise OSError("/Users/private-owner/secret-client-name/archive.json")

    monkeypatch.setattr(showcase, "load_showcase", unreadable)
    entries, errors = showcase.available_history_catalog(assets)
    assert [entry["id"] for entry in entries] == list(IDS[1:])
    assert [error["id"] for error in errors] == [IDS[0]]
    assert "private-owner" not in json.dumps(errors)
    assert "secret-client-name" not in json.dumps(errors)


def test_tolerant_reader_does_not_write_or_initialize_any_store(assets, monkeypatch):
    before = {path: path.read_bytes() for path in assets.rglob("*") if path.is_file()}

    def forbidden(*args, **kwargs):
        pytest.fail("Reading available history must not write files or initialize stores.")

    for method in ("write_text", "write_bytes", "mkdir", "touch"):
        monkeypatch.setattr(Path, method, forbidden)
    entries, errors = showcase.available_history_catalog(assets)
    assert [entry["id"] for entry in entries] == list(IDS)
    assert errors == []
    assert {path: path.read_bytes() for path in assets.rglob("*") if path.is_file()} == before


@pytest.mark.parametrize("study_id", [IDS[1], IDS[2]])
def test_false_raw_output_score_cannot_be_hidden_by_rebalanced_summary(assets, study_id):
    def change(data):
        evaluation = (
            data["controlled_v2"]["results"]["arms"]["alpaca_full_prompt_v1"]["evaluation"]
            if study_id == IDS[1]
            else data["product_smoke"]["results"]["evaluations"]["trained"]
        )
        row = next(
            row
            for row in evaluation["rows"]
            if row["form"] == "classification" and not row["accepted"]
        )
        labels = {row["expected"] for row in evaluation["rows"] if row["form"] == "classification"}
        row["output"] = next(
            label for label in labels if label not in {row["expected"], row["output"]}
        )
        row["accepted"] = row["raw_output_matches"] = True
        for form in ("classification", "all"):
            summary = evaluation["summary"][form]
            summary["accepted"] += 1
            if "acceptance_rate" in summary:
                summary["acceptance_rate"] = summary["accepted"] / summary["total"]

    edit_r148(assets, change)
    assert_available(assets, [identity for identity in IDS if identity != study_id], [study_id])


def test_three_renamed_arms_are_not_the_original_three_protocols(assets):
    def change(data):
        arms = data["controlled_v2"]["results"]["arms"]
        arms["invented-protocol"] = arms.pop("alpaca_full_prompt_v1")
        assert len(arms) == 3

    edit_r148(assets, change)
    assert_available(assets, [IDS[0], IDS[2]], [IDS[1]])


@pytest.mark.parametrize("study_id", [IDS[1], IDS[2]])
def test_catalog_uses_the_same_raw_eos_validation_as_the_detailed_study(assets, study_id):
    def change(data):
        if study_id == IDS[1]:
            row = data["controlled_v2"]["results"]["arms"]["alpaca_answer_only_v1"]["evaluation"][
                "rows"
            ][0]
            row["generated_ids"][-1] += 1
        else:
            row = data["product_smoke"]["results"]["evaluations"]["trained"]["rows"][0]
            row["token_proof"]["generated_token_ids"][-1] += 1

    edit_r148(assets, change)
    assert_available(assets, [identity for identity in IDS if identity != study_id], [study_id])


def test_bad_controlled_baseline_does_not_hide_independent_product_study(assets):
    def change(data):
        evaluation = data["controlled_v2"]["results"]["baseline"]["native_chat_answer_only_v1"]
        evaluation["rows"][0]["output"] = "counterfeit output"

    edit_r148(assets, change)
    assert_available(assets, [IDS[0], IDS[2]], [IDS[1]])


def test_arm_protocol_metadata_must_match_its_known_catalog_identity(assets):
    def change(data):
        arm = data["controlled_v2"]["results"]["arms"]["alpaca_full_prompt_v1"]
        arm["protocol_identity"]["protocol"] = "native_chat_answer_only_v1"

    edit_r148(assets, change)
    assert_available(assets, [IDS[0], IDS[2]], [IDS[1]])


def test_all_real_catalogs_report_the_two_verified_development_groups(assets):
    strict = showcase.history_catalog(assets)
    available, errors = showcase.available_history_catalog(assets)
    assert available == strict and errors == []
    assert [entry["counts"]["development_groups"] for entry in strict] == [2, 2, 2]
    assert [entry["counts"]["development_rows"] for entry in strict] == [8, 16, 16]


@pytest.mark.parametrize(
    "broken", ["missing_preparation", "wrong_preparation", "missing_source_group"]
)
def test_r149_unknown_groups_do_not_discard_verified_scores(assets, broken):
    path = assets / "docs/validation" / showcase.R149_SOURCE
    data = json.loads(path.read_bytes())
    if broken == "missing_preparation":
        del data["preparation"]["development_groups"]
    elif broken == "wrong_preparation":
        data["preparation"]["development_groups"] = ["different-group"]
    else:
        for model in data["actual_report"]["models"]:
            for row in model["rows"]:
                row["source"].pop("group")
    path.write_text(json.dumps(data))
    entries, _ = assert_available(assets, list(IDS), [], group_counts={IDS[0]: None})
    assert entries == showcase.history_catalog(assets)


@pytest.mark.parametrize(
    "broken",
    [
        "missing_cases",
        "missing_group",
        "blank_group",
        "wrong_identity",
        "wrong_input",
        "wrong_answer",
    ],
)
def test_r148_unbound_logical_cases_leave_both_cards_but_no_group_claim(assets, broken):
    def change(data):
        cases = data["controlled_v2"]["data"]["development"]
        if broken == "missing_cases":
            del data["controlled_v2"]["data"]["development"]
        elif broken == "missing_group":
            del cases[0]["group_id"]
        elif broken == "blank_group":
            cases[0]["group_id"] = " "
        elif broken == "wrong_identity":
            cases[0]["case_id"] = "unrecorded-case"
        elif broken == "wrong_input":
            cases[0]["input"] = "different input"
        else:
            cases[0]["output"] = "different answer"

    edit_r148(assets, change)
    entries, _ = assert_available(assets, list(IDS), [], group_counts={IDS[1]: None, IDS[2]: None})
    assert entries == showcase.history_catalog(assets)


def test_product_data_source_mismatch_only_makes_product_groups_unknown(assets):
    edit_r148(assets, lambda data: data["product_smoke"].update(data_source="different-source"))
    entries, _ = assert_available(assets, list(IDS), [], group_counts={IDS[2]: None})
    assert entries == showcase.history_catalog(assets)


def test_product_card_survives_without_shared_controlled_input_data(assets):
    edit_r148(assets, lambda data: data.pop("controlled_v2"))
    assert_available(assets, [IDS[0], IDS[2]], [IDS[1]], group_counts={IDS[2]: None})


def test_group_count_is_derived_instead_of_fixed_at_two(assets):
    def change(data):
        for case in data["controlled_v2"]["data"]["development"]:
            case["group_id"] = "one-group"

    edit_r148(assets, change)
    assert_available(assets, list(IDS), [], group_counts={IDS[1]: 1, IDS[2]: 1})


def test_input_binding_is_checked_against_each_studys_actual_prompts(assets):
    def change(data):
        arm = data["controlled_v2"]["results"]["arms"]["native_chat_answer_only_v1"]
        arm["evaluation"]["rows"][0]["prompt"] = "different native user input"

    edit_r148(assets, change)
    # The scores remain valid, but controlled grouping cannot be attributed to
    # the logical cases. Product prompts still match their own original data.
    assert_available(assets, list(IDS), [], group_counts={IDS[1]: None})
