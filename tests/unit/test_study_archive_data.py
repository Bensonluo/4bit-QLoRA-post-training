"""R148 studies retain original prompts, precision and archived training logs."""

import hashlib
import io
import json
import shutil
import tarfile
from pathlib import Path

import pytest

from ui.study_archive import ARCHIVE, LOGS, PREFIX, SOURCE, STATE, load_study

ROOT = Path(__file__).resolve().parents[2]


def original(root=ROOT):
    return json.loads((root / "docs/validation" / SOURCE).read_bytes())


@pytest.fixture
def assets(tmp_path):
    target = tmp_path / "docs/validation"
    target.mkdir(parents=True)
    for name in (SOURCE, ARCHIVE):
        shutil.copyfile(ROOT / "docs/validation" / name, target / name)
    return tmp_path


def save(root, data):
    (root / "docs/validation" / SOURCE).write_text(json.dumps(data, ensure_ascii=False))


def variant(study, identity):
    return next(item for item in study["variants"] if item["id"] == identity)


def test_controlled_variants_keep_source_rows_and_separate_native_prompts():
    data = original()["controlled_v2"]["results"]
    study = load_study(ROOT, "r148-controlled-v2")
    assert study["default_pair"] == ["alpaca_full_prompt_v1", "alpaca_answer_only_v1"]
    assert len(study["variants"]) == 5
    for protocol, arm in data["arms"].items():
        model = variant(study, protocol)
        assert model["rows"] == arm["evaluation"]["rows"]
        assert model["summary"] == arm["evaluation"]["summary"]
        assert len(model["training_history"]) == 64
        assert model["training_precision"] == model["evaluation_precision"] == "torch.float32"
    native = variant(study, "native_chat_answer_only_v1")
    alpaca = variant(study, "alpaca_answer_only_v1")
    assert native["baseline_id"] == "base-native"
    assert alpaca["baseline_id"] == "base-alpaca"
    assert native["rows"][0]["prompt"].startswith("<|im_start|>user")
    assert alpaca["rows"][0]["prompt"].startswith("### Instruction:")
    assert native["rows"][0]["prompt_ids"] != alpaca["rows"][0]["prompt_ids"]
    assert variant(study, "base-native")["summary"]["all"]["accepted"] == 2
    for identity in ("base-native", "base-alpaca"):
        model = variant(study, identity)
        assert model["training_history"] == []
        assert model["training_precision"] is None


def test_product_smoke_keeps_training_fp32_inference_bf16_and_real_64_points():
    source = original()["product_smoke"]
    study = load_study(ROOT, "r148-product-smoke")
    assert study["default_pair"] == ["trained", "base"]
    base, trained = variant(study, "base"), variant(study, "trained")
    assert base["training_history"] == []
    assert base["training_precision"] is None
    assert trained["training_precision"] == "torch.float32"
    assert trained["evaluation_precision"] == base["evaluation_precision"] == "torch.bfloat16"
    assert trained["baseline_id"] == "base"
    assert [point["step"] for point in trained["training_history"]] == list(range(1, 65))
    assert trained["training_history"][-1]["loss"] == 0.34073030948638916
    for name in ("base", "trained"):
        model = variant(study, name)
        assert model["rows"] == source["results"]["evaluations"][name]["rows"]
        assert model["summary"] == source["results"]["evaluations"][name]["summary"]
    assert trained["summary"]["all"]["accepted"] == 12
    assert all(row["truncated"] for row in base["rows"])


def test_controlled_curves_equal_whitelisted_original_archive_logs():
    study = load_study(ROOT, "r148-controlled-v2")
    with tarfile.open(ROOT / "docs/validation" / ARCHIVE) as bundle:
        for protocol, relative in LOGS.items():
            logs = json.load(bundle.extractfile(PREFIX + relative))
            history = variant(study, protocol)["training_history"]
            assert history == [
                {"step": point["step"], "loss": point["loss"], "learning_rate": point["lr"]}
                for point in logs
            ]


def test_archive_and_member_identities_are_exposed_without_server_paths():
    study = load_study(ROOT, "r148-product-smoke")
    raw = (ROOT / "docs/validation" / SOURCE).read_bytes()
    assert study["source"]["sha256"] == hashlib.sha256(raw).hexdigest()
    assert study["source"]["archive"]["sha256"] == original()["detailed_evidence_archive"]["sha256"]
    assert {member["name"] for member in study["source"]["members"]} == {"manifest.json", STATE}
    assert "/Users/" not in json.dumps(study)
    assert "data_source" not in study and "actual_training_arguments" not in study


@pytest.mark.parametrize("name", [SOURCE, ARCHIVE])
def test_missing_required_evidence_is_not_replaced_with_fake_results(assets, name):
    (assets / "docs/validation" / name).unlink()
    with pytest.raises(ValueError, match="公开历史研究"):
        load_study(assets, "r148-product-smoke")


def test_changed_archive_bytes_are_rejected(assets):
    path = assets / "docs/validation" / ARCHIVE
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="归档大小或 SHA256"):
        load_study(assets, "r148-product-smoke")


@pytest.mark.parametrize(
    "mutation", ["summary", "raw_output", "native_prompt", "log_reference", "training_precision"]
)
def test_contradictory_json_metadata_fails_clearly(assets, mutation):
    data = original(assets)
    study_id = "r148-controlled-v2"
    arm = data["controlled_v2"]["results"]["arms"]["native_chat_answer_only_v1"]
    if mutation == "summary":
        arm["evaluation"]["summary"]["all"]["accepted"] = 16
    elif mutation == "raw_output":
        next(row for row in arm["evaluation"]["rows"] if row["accepted"])["output"] = (
            "invented replacement"
        )
    elif mutation == "native_prompt":
        arm["evaluation"]["rows"][0]["prompt"] = "### Instruction:\nwrong common prompt"
    elif mutation == "log_reference":
        arm["optimizer_update_evidence"]["full_log"]["archive_relative_path"] = "../../other.json"
    else:
        study_id = "r148-product-smoke"
        data["product_smoke"]["settings"]["actual_training_dtype"] = "torch.bfloat16"
    save(assets, data)
    with pytest.raises(ValueError, match="公开历史研究"):
        load_study(assets, study_id)


def repack(assets, change, *, attest=False):
    """Fixtures read only the same exact safe member allowlist as the product."""
    data = original(assets)
    path = assets / "docs/validation" / ARCHIVE
    selected = ["manifest.json", *LOGS.values(), STATE]
    with tarfile.open(path) as source:
        contents = {name: source.extractfile(PREFIX + name).read() for name in selected}
    contents, special = change(contents)
    if attest:
        state = contents[STATE]
        manifest = json.loads(contents["manifest.json"])
        manifest[STATE] = {"size": len(state), "sha256": hashlib.sha256(state).hexdigest()}
        contents["manifest.json"] = json.dumps(manifest).encode()
        data["product_smoke"]["trainer_state"]["sanitized_sha256"] = manifest[STATE]["sha256"]
    with tarfile.open(path, "w:gz") as destination:
        for name, raw in contents.items():
            member = tarfile.TarInfo(PREFIX + name)
            member.size = len(raw)
            if special == "symlink" and name == STATE:
                member.type = tarfile.SYMTYPE
                member.linkname = "../../outside.json"
                member.size = 0
            destination.addfile(member, io.BytesIO(raw))
            if special == "duplicate" and name == STATE:
                destination.addfile(member, io.BytesIO(raw))
    raw = path.read_bytes()
    data["detailed_evidence_archive"].update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    save(assets, data)


@pytest.mark.parametrize("mode", ["hash", "missing", "duplicate", "symlink", "short_log"])
def test_member_integrity_and_exact_64_updates_are_checked(assets, mode):
    def change(contents):
        if mode == "missing":
            del contents[STATE]
        elif mode == "hash":
            contents[STATE] += b" "
        elif mode == "short_log":
            state = json.loads(contents[STATE])
            del state["log_history"][0]
            contents[STATE] = json.dumps(state).encode()
        return contents, mode

    repack(assets, change, attest=mode == "short_log")
    with pytest.raises(ValueError, match="公开历史研究"):
        load_study(assets, "r148-product-smoke")


def test_reader_only_opens_public_files_and_extracts_exact_members(assets, monkeypatch):
    read_bytes = Path.read_bytes
    extractfile = tarfile.TarFile.extractfile
    reads, members = [], []

    def public_only(path):
        assert path == assets / "docs/validation" / path.name
        assert path.name in {SOURCE, ARCHIVE}
        reads.append(path.name)
        return read_bytes(path)

    def selected_only(bundle, member):
        assert member.name in {PREFIX + name for name in ["manifest.json", *LOGS.values(), STATE]}
        members.append(member.name)
        return extractfile(bundle, member)

    def forbidden(*args, **kwargs):
        raise AssertionError("Display must not write stores or extract archive files")

    monkeypatch.setattr(Path, "read_bytes", public_only)
    for name in ("write_bytes", "write_text", "mkdir", "touch"):
        monkeypatch.setattr(Path, name, forbidden)
    monkeypatch.setattr(tarfile.TarFile, "extractfile", selected_only)
    monkeypatch.setattr(tarfile.TarFile, "extract", forbidden)
    monkeypatch.setattr(tarfile.TarFile, "extractall", forbidden)
    load_study(assets, "r148-controlled-v2")
    load_study(assets, "r148-product-smoke")
    assert reads == [SOURCE, ARCHIVE, SOURCE, ARCHIVE]
    assert len(members) == 6


def test_unknown_study_does_not_read_any_file(monkeypatch):
    monkeypatch.setattr(Path, "read_bytes", lambda *args: pytest.fail("Unexpected read"))
    with pytest.raises(ValueError, match="未收录"):
        load_study(ROOT, "invented-study")


@pytest.mark.parametrize("study_id", ["r148-controlled-v2", "r148-product-smoke"])
def test_shared_business_cases_keep_source_inputs_but_not_a_fabricated_shared_prompt(study_id):
    study = load_study(ROOT, study_id)
    assert study["cases"] == original()["controlled_v2"]["data"]["development"]
    assert len(study["cases"]) == 16
    assert all(
        "input" in case and "instruction" in case and "prompt" not in case
        for case in study["cases"]
    )
    for model in study["variants"]:
        assert {(row["case_id"], row["form"], row["expected"]) for row in model["rows"]} == {
            (row["case_id"], row["form"], row["output"]) for row in study["cases"]
        }


def test_shared_cases_must_bind_the_actual_reference_answer(assets):
    data = original(assets)
    data["controlled_v2"]["data"]["development"][0]["output"] = "wrong"
    save(assets, data)
    with pytest.raises(ValueError, match="业务输入与版本"):
        load_study(assets, "r148-product-smoke")


def test_json_duplicate_key_cannot_silently_overwrite_a_wrong_answer(assets):
    data = original(assets)
    rows = data["controlled_v2"]["results"]["arms"]["alpaca_answer_only_v1"]["evaluation"]["rows"]
    row = next(row for row in rows if row["form"] == "json" and row["accepted"])
    row["output"] = '{"route":"wrong",' + row["output"].lstrip()[1:]
    save(assets, data)
    with pytest.raises(ValueError, match="逐题得分"):
        load_study(assets, "r148-controlled-v2")


def test_classification_whitespace_follows_original_strip_judge_without_repairing_output(assets):
    data = original(assets)
    rows = data["controlled_v2"]["results"]["arms"]["alpaca_answer_only_v1"]["evaluation"]["rows"]
    row = next(row for row in rows if row["form"] == "classification" and row["accepted"])
    row["output"] = " " + row["expected"] + " \n"
    save(assets, data)
    study = load_study(assets, "r148-controlled-v2")
    actual = next(
        item
        for item in variant(study, "alpaca_answer_only_v1")["rows"]
        if item["case_id"] == row["case_id"]
    )
    assert actual["output"] == row["output"]
    assert actual["accepted"] is True


@pytest.mark.parametrize("study_id", ["r148-controlled-v2", "r148-product-smoke"])
@pytest.mark.parametrize("change", ["last_id", "length", "finish"])
def test_raw_token_eos_and_lengths_cannot_disagree_with_claimed_completion(
    assets, study_id, change
):
    data = original(assets)
    if study_id == "r148-controlled-v2":
        row = data["controlled_v2"]["results"]["arms"]["alpaca_answer_only_v1"]["evaluation"][
            "rows"
        ][0]
        proof, ids_key = row, "generated_ids"
    else:
        row = data["product_smoke"]["results"]["evaluations"]["trained"]["rows"][0]
        proof, ids_key = row["token_proof"], "generated_token_ids"
    if change == "last_id":
        proof[ids_key][-1] = 151643
    elif change == "length":
        row["generated_tokens"] += 1
    else:
        proof["finish_reason"] = "length"
    save(assets, data)
    with pytest.raises(ValueError, match="token"):
        load_study(assets, study_id)
