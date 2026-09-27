"""Financial-style temporal data remains inspectable in the existing intake page."""

from src.workbench.evaluation_suites import EvalSuiteService
from src.workbench.intake_service import IntakeService
from tests.unit.test_data_intake_ui import PAGE, AppTest, button
from tests.unit.test_forecast_workflow import run_workflow


def test_temporal_dataset_shows_exclusions_and_hides_random_controls(tmp_path, monkeypatch):
    import ui.config

    monkeypatch.setattr(ui.config, "PROJECT_ROOT", tmp_path)
    for name in ("PROVIDER", "BASE_URL", "MODEL", "API_KEY"):
        monkeypatch.delenv(f"TUNESMITH_AGENT_{name}", raising=False)
    evidence = run_workflow(tmp_path / "outputs/workbench")
    page = AppTest.from_file(str(PAGE), default_timeout=20).run()
    page.selectbox(key="intake_select").select(evidence["session_id"]).run()
    assert not page.exception
    assert any("训练 2 条、验证 2 条、测试 2 条" in item.value for item in page.info)
    assert any("排除并保留 3 条" in item.value for item in page.info)
    assert any("已提出的时间分区方案" in item.value for item in page.markdown)
    assert not any("比例" in item.label or "种子" in item.label for item in page.number_input)
    frames = [item.value for item in page.dataframe]
    assert any("观察截止时标签尚未成熟" in frame.to_string() for frame in frames)
    assert any("训练标签窗口跨越验证起点" in frame.to_string() for frame in frames)
    service = IntakeService(tmp_path / "outputs/workbench/intake")
    session = service.load(evidence["session_id"])
    ref = EvalSuiteService(tmp_path / "outputs/workbench/evaluation-suites").freeze(
        session, new_suite=True
    )
    session.dataset = None
    service._save(session, session.revision)
    page.run()
    selector = next(item for item in page.selectbox if item.label == "后续轮次的固定开发/测试题集")
    selector.select(ref["suite_id"]).run()
    assert not page.exception
    assert any("新增资料按已确认时间归属" in item.value for item in page.info)
    # 页面与 CLI 同源的人话摘要：题数+锁定+不自动扩充+比较基线边界。
    assert any("这套固定题集含开发题" in item.value for item in page.markdown)
    assert any("不代表业务效果达标" in item.value for item in page.markdown)
    assert not any("比例" in item.label or "种子" in item.label for item in page.number_input)
    button(page, "生成数据集版本").click().run()
    assert not page.exception
    assert (
        service.load(session.session_id).dataset.statistics["split_method"]
        == "temporal_fixed_evaluation_suite"
    )
