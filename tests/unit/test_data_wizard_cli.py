"""数据向导 CLI 冒烟测试（不落 outputs/，全部走 tmp_path）。"""

import pytest

from scripts.data_wizard import main


@pytest.fixture()
def sample_csv(tmp_path):
    path = tmp_path / "raw.csv"
    lines = ["标准名,别名,编码,类型"]
    lines += [f"药{i},别名{i},Z{i},drug" for i in range(12)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class TestCli:
    def test_suggest_mode_exits_without_files(self, sample_csv, tmp_path) -> None:
        out_dir = tmp_path / "out"
        code = main(["--input", str(sample_csv), "--suggest", "--out-dir", str(out_dir)])
        assert code == 0
        assert not out_dir.exists()

    def test_full_run_exports(self, sample_csv, tmp_path) -> None:
        out_dir = tmp_path / "out"
        code = main(["--input", str(sample_csv), "--out-dir", str(out_dir)])
        assert code == 0
        assert (out_dir / "train.json").exists()
        assert (out_dir / "wizard_report.json").exists()

    def test_missing_input_returns_2(self, tmp_path) -> None:
        assert main(["--input", str(tmp_path / "nope.csv")]) == 2

    def test_unknown_template_returns_2(self, sample_csv, tmp_path) -> None:
        code = main(
            ["--input", str(sample_csv), "--template", "nope", "--out-dir", str(tmp_path / "o")]
        )
        assert code == 2

    def test_explicit_columns_override_suggestion(self, sample_csv, tmp_path) -> None:
        out_dir = tmp_path / "out"
        code = main(
            [
                "--input",
                str(sample_csv),
                "--out-dir",
                str(out_dir),
                "--standard-col",
                "标准名",
                "--query-col",
                "别名",
                "--type-col",
                "类型",
                "--candidates",
                "5",
            ]
        )
        assert code == 0
        import json

        with open(out_dir / "train.json", encoding="utf-8") as f:
            records = json.load(f)
        assert records[0]["metadata"]["entity_type"] == "drug"
