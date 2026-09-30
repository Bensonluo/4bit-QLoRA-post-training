"""数据向导配置校验测试。"""

import pytest

from src.data.wizard.spec import FieldMapping, WizardError, WizardSpec


class TestFieldMappingValidate:
    def test_pass_with_valid_columns(self) -> None:
        mapping = FieldMapping(
            standard_name="标准名", query="别名", code="编码", variants="变体", entity_type="类型"
        )
        assert mapping.validate(["标准名", "别名", "编码", "变体", "类型"]) == []

    def test_missing_standard_name(self) -> None:
        assert any("standard_name" in e for e in FieldMapping().validate(["任意"]))

    def test_standard_not_in_columns(self) -> None:
        errors = FieldMapping(standard_name="标准名").validate(["别名"])
        assert any("不在表格列中" in e for e in errors)

    def test_optional_role_not_in_columns(self) -> None:
        errors = FieldMapping(standard_name="标准名", query="别名").validate(["标准名"])
        assert any("query 列" in e for e in errors)


class TestWizardSpecValidation:
    def test_defaults_valid(self) -> None:
        spec = WizardSpec()
        assert spec.template == "entity_matching"
        assert spec.n_candidates == 8

    def test_empty_template_rejected(self) -> None:
        with pytest.raises(WizardError, match="template"):
            WizardSpec(template="")

    def test_wrong_ratio_length(self) -> None:
        with pytest.raises(WizardError, match="三元组"):
            WizardSpec(split_ratios=(0.8, 0.2))

    def test_non_positive_ratio(self) -> None:
        with pytest.raises(WizardError, match="大于 0"):
            WizardSpec(split_ratios=(1.0, 0.0, 0.0))

    def test_ratios_not_summing_to_one(self) -> None:
        with pytest.raises(WizardError, match="约为 1.0"):
            WizardSpec(split_ratios=(0.5, 0.2, 0.1))

    def test_too_few_candidates(self) -> None:
        with pytest.raises(WizardError, match="n_candidates"):
            WizardSpec(n_candidates=1)
