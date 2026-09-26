"""训练后下一步引导（src/tracking/next_steps.py）的单元测试。

覆盖：adapter 定位（含 PEFT 标志文件）、向导评测集兄弟文件发现、
自动注册名解析、相对/绝对 output_dir 解析、配置缺失时的安全回退。
"""

from __future__ import annotations

import yaml

from src.tracking.next_steps import summarize_run_artifacts

CONFIG_TEMPLATE = """\
model:
  name: Qwen/Qwen2.5-1.5B-Instruct
training:
  num_epochs: 1
  output_dir: {output_dir}
data:
  dataset_name: {dataset}
logging:
  use_mlflow: true
"""


def _write_config(path, output_dir: str, dataset: str, extra_logging: str = "") -> None:
    text = CONFIG_TEMPLATE.format(output_dir=output_dir, dataset=dataset)
    if extra_logging:
        text += extra_logging
    path.write_text(text, encoding="utf-8")


class TestSummarizeRunArtifacts:
    def test_full_success_path(self, tmp_path) -> None:
        out = tmp_path / "run-x"
        out.mkdir()
        (out / "adapter_config.json").write_text("{}", encoding="utf-8")
        (out / "adapter_model.safetensors").write_bytes(b"x")
        wizard_dir = tmp_path / "wizard" / "demo"
        wizard_dir.mkdir(parents=True)
        for split in ("train", "val", "test"):
            (wizard_dir / f"{split}.json").write_text("[]", encoding="utf-8")
        cfg = tmp_path / "run-x.yaml"
        _write_config(
            cfg,
            output_dir="./run-x",
            dataset=str(wizard_dir / "train.json"),
            extra_logging="  register_model: true\n  registry_model_name: Demo-QLoRA\n",
        )

        arts = summarize_run_artifacts(cfg, tmp_path)

        assert arts.has_adapter is True
        assert arts.output_dir == out.resolve()
        assert arts.eval_sets == {
            "test": (wizard_dir / "test.json").resolve(),
            "val": (wizard_dir / "val.json").resolve(),
        }
        assert arts.registered_name == "Demo-QLoRA"
        assert arts.model_name == "Qwen/Qwen2.5-1.5B-Instruct"

    def test_hf_dataset_yields_no_eval_sets(self, tmp_path) -> None:
        out = tmp_path / "run-y"
        out.mkdir()
        (out / "adapter_config.json").write_text("{}", encoding="utf-8")
        cfg = tmp_path / "run-y.yaml"
        _write_config(cfg, output_dir=str(out), dataset="yahma/alpaca-cleaned")

        arts = summarize_run_artifacts(cfg, tmp_path)

        assert arts.has_adapter is True
        assert arts.eval_sets == {}
        assert arts.registered_name is None

    def test_output_dir_without_adapter(self, tmp_path) -> None:
        out = tmp_path / "run-z"
        out.mkdir()
        cfg = tmp_path / "run-z.yaml"
        _write_config(cfg, output_dir=str(out), dataset="yahma/alpaca-cleaned")

        arts = summarize_run_artifacts(cfg, tmp_path)

        assert arts.has_adapter is False
        assert arts.output_dir == out.resolve()

    def test_adapter_model_glob_alone_counts(self, tmp_path) -> None:
        # 有些导出只留权重文件不留 config —— glob 兜底识别
        out = tmp_path / "run-g"
        out.mkdir()
        (out / "adapter_model.bin").write_bytes(b"x")
        cfg = tmp_path / "run-g.yaml"
        _write_config(cfg, output_dir=str(out), dataset="yahma/alpaca-cleaned")

        assert summarize_run_artifacts(cfg, tmp_path).has_adapter is True

    def test_missing_config_returns_safe_empty(self, tmp_path) -> None:
        arts = summarize_run_artifacts(tmp_path / "nope.yaml", tmp_path)
        assert arts.has_adapter is False
        assert arts.output_dir is None
        assert arts.eval_sets == {}
        assert arts.registered_name is None
        assert arts.model_name == ""

    def test_absolute_output_dir_untouched(self, tmp_path) -> None:
        out = tmp_path / "abs-run"
        out.mkdir()
        (out / "adapter_config.json").write_text("{}", encoding="utf-8")
        cfg = tmp_path / "abs.yaml"
        _write_config(cfg, output_dir=str(out), dataset="yahma/alpaca-cleaned")

        arts = summarize_run_artifacts(cfg, tmp_path)

        assert arts.output_dir == out.resolve()

    def test_eval_set_only_counts_existing_siblings(self, tmp_path) -> None:
        # 只有 test.json、没有 val.json → 只报 test
        wizard_dir = tmp_path / "wizard" / "solo"
        wizard_dir.mkdir(parents=True)
        (wizard_dir / "train.json").write_text("[]", encoding="utf-8")
        (wizard_dir / "test.json").write_text("[]", encoding="utf-8")
        cfg = tmp_path / "solo.yaml"
        _write_config(cfg, output_dir="./whatever", dataset=str(wizard_dir / "train.json"))

        arts = summarize_run_artifacts(cfg, tmp_path)

        assert set(arts.eval_sets) == {"test"}

    def test_register_model_without_name_is_none(self, tmp_path) -> None:
        cfg = tmp_path / "r.yaml"
        _write_config(
            cfg,
            output_dir="./x",
            dataset="yahma/alpaca-cleaned",
            extra_logging="  register_model: true\n",
        )
        arts = summarize_run_artifacts(cfg, tmp_path)
        assert arts.registered_name is None

    def test_yaml_with_non_dict_top_level(self, tmp_path) -> None:
        cfg = tmp_path / "bad.yaml"
        cfg.write_text("- just\n- a\n- list\n", encoding="utf-8")
        arts = summarize_run_artifacts(cfg, tmp_path)
        assert arts.model_name == ""
        assert arts.has_adapter is False


def test_config_template_is_valid_yaml(tmp_path) -> None:
    """钉住测试所用的 config 形态与 Training Lab 实际写出的一致（含 logging 嵌套）。"""
    cfg = tmp_path / "shape.yaml"
    _write_config(
        cfg,
        output_dir="./run",
        dataset="outputs/wizard/d/train.json",
        extra_logging="  register_model: true\n  registry_model_name: D-QLoRA\n",
    )
    parsed = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    assert set(parsed) == {"model", "training", "data", "logging"}
    assert parsed["logging"]["register_model"] is True
