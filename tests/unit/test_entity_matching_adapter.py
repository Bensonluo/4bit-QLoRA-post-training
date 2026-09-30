"""通用实体匹配域适配器（R120 评测闭环）注册测试。

generic-first 门面：02/03 域选择器默认落「实体匹配（通用）」，medical
保留为已验证案例（标签不动）。通用适配器以子类复用案例图表——非复制
分叉（与 R119 wizard 模板的案例子类化同一范式）。
"""

from ui.components.domain_adapters import (
    EntityMatchingAdapter,
    MedicalEntityAdapter,
    get_adapter,
    get_domain_display_name,
    list_domains,
)


class TestGenericAdapterRegistration:
    def test_generic_domain_registered_first(self) -> None:
        domains = list_domains()
        assert domains, "注册表不得为空"
        assert domains[0] == "entity_matching", "通用域必须排第一（02/03 选择器默认项）"
        assert "medical_entity" in domains, "医疗案例域保留"

    def test_generic_adapter_reuses_case_charts(self) -> None:
        adapter = get_adapter("entity_matching")
        assert isinstance(adapter, EntityMatchingAdapter)
        # 子类复用：图表逻辑全部继承自案例适配器，零复制分叉
        assert isinstance(adapter, MedicalEntityAdapter)

    def test_display_names(self) -> None:
        assert get_domain_display_name("entity_matching") == "实体匹配（通用）"
        # 案例标签不动（generic-first = 通用默认 + 案例如实标注，不是抹掉案例）
        assert get_domain_display_name("medical_entity") == "医疗实体匹配"

    def test_unknown_domain_falls_back_to_name(self) -> None:
        assert get_domain_display_name("nope") == "nope"
