"""学习率分档推荐（QUICKSTART 第 4 步外部分档，非本产品实测）按数据量给出正确档位。"""

import pytest

pytest.importorskip("streamlit")

from tests.unit import test_data_intake_ui as intake_ui

# <2,000 条的行为见 test_workbench_training_ui.test_learning_rate_tier_suggestion_fills_lower_tier_on_request
BIG_FULL = (
    "编号,客户描述,类别,处理结果\n".encode()
    + "".join(
        f"{index},客户描述{index},{'质量' if index % 2 else '物流'},补发\n" for index in range(2400)
    ).encode()
)


@pytest.fixture()
def big_training_page(tmp_path, monkeypatch):
    """≥2,000 条全量的同款页面夹具（只渲染页面，不启动训练，用真实服务即可）。"""
    import ui.config
    from src.workbench.intake_service import IntakeService
    from tests.unit.test_full_data import approved

    for name in ("PROVIDER", "BASE_URL", "MODEL", "API_KEY"):
        monkeypatch.delenv(f"TUNESMITH_AGENT_{name}", raising=False)
    monkeypatch.setattr(ui.config, "PROJECT_ROOT", tmp_path)
    service = IntakeService(tmp_path / "outputs/workbench/intake")
    session = approved(service)
    session = service.validate_full_data(session.session_id, session.revision, "full.csv", BIG_FULL)
    session = service.confirm_full_data(session.session_id, session.revision)
    session = service.materialize_dataset(session.session_id, session.revision)
    return service, session, intake_ui.AppTest.from_file(str(intake_ui.PAGE), default_timeout=30)


def test_large_dataset_suggests_2e4_tier(big_training_page):
    """≥2,000 条：建议档位为 2e-4（通用可靠默认），采用后仍是 2e-4，并明示非本产品实测。"""
    _, session, page = big_training_page
    page.run()
    page.selectbox(key="intake_select").select(session.session_id).run()
    assert not page.exception
    captions = "\n".join(caption.value for caption in page.caption)
    assert "全量 2400 条（≥ 2,000）" in captions
    assert "2e-4" in captions
    assert "非本产品实测" in captions
    lr_input = next(field for field in page.number_input if field.label == "学习率")
    assert lr_input.value == pytest.approx(0.0002)
    intake_ui.button(page, "采用建议学习率").click().run()
    assert not page.exception
    lr_input = next(field for field in page.number_input if field.label == "学习率")
    assert lr_input.value == pytest.approx(0.0002)
