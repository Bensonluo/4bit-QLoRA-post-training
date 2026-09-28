"""可学性探针:零样本 vs 多数类基线的最便宜证据,附样本量限制声明。"""

import json

import pytest

from src.workbench.business_evaluation import Generation
from src.workbench.intake_service import IntakeService
from src.workbench.learnability_probe import probe_learnability, save_probe
from tests.unit.test_data_materialize import _full


@pytest.fixture()
def store(tmp_path):
    service = IntakeService(tmp_path / "intake")
    session = _full(service)
    session = service.materialize_dataset(session.session_id, session.revision)
    return service, session


def _factory(answer):
    def make(model):
        class Runtime:
            def generate(self, prompt, protocol):
                assert protocol.max_new_tokens > 0
                return Generation(answer, truncated=False)

            def close(self):
                pass

        return Runtime()

    return make


def test_probe_reports_accuracy_against_majority_baseline(store, tmp_path):
    _, session = store
    labels = [row["output"] for row in _labels(session)]
    majority = max(set(labels), key=labels.count)

    result = probe_learnability(
        session, "/tmp/base", runtime_factory=_factory(majority), sample_size=4
    )
    assert result["zero_shot_accuracy"] == 1.0
    assert result["majority_baseline"] > 0
    assert result["difference"] >= 0
    assert result["sample_size"] == min(4, result["dev_total"])
    assert "不能预测微调效果" in result["note"]
    assert all(item["match"] for item in result["observations"])
    path = save_probe(tmp_path / "probes", result)
    assert path.exists()

    # 全错的零样本:与基线的差为负,如实呈现
    wrong = probe_learnability(
        session, "/tmp/base", runtime_factory=_factory("绝不正确的答案"), sample_size=4
    )
    assert wrong["zero_shot_accuracy"] == 0.0
    assert wrong["difference"] == -wrong["majority_baseline"]
    assert wrong["observations"][0]["expected"] in labels


def _labels(session):
    import json

    path = session.dataset.paths["validation"]
    return [json.loads(line) for line in open(path) if line.strip()]


def test_probe_validates_inputs(store):
    _, session = store
    with pytest.raises(ValueError, match="抽样数量"):
        probe_learnability(session, "/tmp/base", runtime_factory=_factory("x"), sample_size=0)
    bare = session.model_copy(deep=True)
    bare.dataset = None
    with pytest.raises(ValueError, match="独立分区"):
        probe_learnability(bare, "/tmp/base", runtime_factory=_factory("x"))


def test_label_error_candidates_cross_signal_ranking(store, tmp_path):
    """基座与用户盲标双信号都矛盾的行排前;截断不作为分歧证据。"""
    _, session = store
    labels = [row["output"] for row in _labels(session)]
    majority = max(set(labels), key=labels.count)

    # 先造一个用户盲标分歧信号:提交与数据标签不同的答案(同库服务)
    service = store[0]
    current = service.load(session.session_id)
    pending = service.start_label_verification(current.session_id, current.revision, sample_size=1)
    answers = {item["row_id"]: "用户也不认同的答案" for item in pending["items"]}
    service.submit_label_verification(current.session_id, pending["verification_id"], answers)
    current = service.load(session.session_id)
    assert current.label_verification["verdict"] == "insufficient_agreement"

    calls = {"n": 0}

    def make(model):
        class Runtime:
            def generate(self, prompt, protocol):
                calls["n"] += 1
                answer = "基座不认同的输出" if calls["n"] % 2 else majority
                return Generation(answer, truncated=False)

            def close(self):
                pass

        return Runtime()

    result = probe_learnability(
        current, "/tmp/base", runtime_factory=lambda model: make(model), sample_size=2
    )
    candidates = result["label_error_candidates"]
    assert isinstance(candidates, list)
    assert "候选不等于错误" in result["candidates_note"]
    strong = [c for c in candidates if c["user_blind_answer"]]
    if strong:
        assert candidates[0]["user_blind_answer"], "强证据排前"
        assert "强证据" in candidates[0]["evidence"]
        # 强证据行必须带溯源提示:建议对照原始来源行,不要只凭模型输出改数据
        assert "原始来源行" in candidates[0]["evidence"]
        weak = [c for c in candidates if not c["user_blind_answer"]]
        assert all("原始来源行" not in c["evidence"] for c in weak), "弱信号不冒充溯源建议"
    assert "原始来源行" in result["candidates_note"]


def test_describe_candidates_plain_lines_and_empty_message():
    """人话清单:强证据带 ⚠ 逐行可核对;没有候选时给明确说法而不是沉默。"""
    from src.workbench.learnability_probe import describe_candidates

    assert describe_candidates([]) == ["没有发现值得优先核对的行。"]

    lines = describe_candidates(
        [
            {
                "row_id": "r1",
                "data_label": "质量",
                "base_zero_shot": "物流",
                "user_blind_answer": "服务",
                "evidence": "强证据,优先人工核对",
            },
            {
                "row_id": "r2",
                "data_label": "物流",
                "base_zero_shot": "质量",
                "user_blind_answer": None,
                "evidence": "弱信号供参考",
            },
        ]
    )
    assert len(lines) == 3
    assert "候选 2 行" in lines[0] and "强证据 1 行" in lines[0]
    assert "候选不等于错误" in lines[0]
    assert lines[1].startswith("⚠") and "r1" in lines[1] and "服务" in lines[1]
    assert "原始来源行" in lines[1], "强证据行在 CLI 清单里也提示对照原始来源行"
    assert lines[2].startswith("·") and "r2" in lines[2] and "⚠" not in lines[2]
    assert "原始来源行" not in lines[2]


def test_describe_candidates_limits_listing_like_ui_pagination():
    """CLI 与页面分页对齐:默认只逐行列出前 20 条,截断如实说明并指向导出 CSV。"""
    from src.workbench.learnability_probe import describe_candidates

    candidates = [
        {
            "row_id": f"r{index:06d}",
            "data_label": "质量",
            "base_zero_shot": "物流",
            "user_blind_answer": "用户盲标" if index % 5 == 1 else None,
            "evidence": "弱信号供参考",
        }
        for index in range(1, 26)
    ]

    lines = describe_candidates(candidates)  # 默认 limit=20,与页面每页 20 条对齐
    assert len(lines) == 22  # 表头 + 前 20 行 + 截断说明,不是全量 25 行
    assert "候选 25 行" in lines[0], "总数说明始终如实"
    for index, line in zip(range(1, 21), lines[1:21]):
        assert f"r{index:06d}" in line
    tail = lines[-1]
    assert "前 20/25 条" in tail and "其余 5 条" in tail and "导出 CSV" in tail
    joined = "".join(lines)
    assert "r000021" not in joined and "r000025" not in joined, "没列出的行不冒充已显示"

    # limit=None 或 limit 不小于总数:逐行全列,行为与不分页时一致
    full_lines = describe_candidates(candidates, limit=None)
    assert len(full_lines) == 26
    assert "r000025" in full_lines[-1]
    assert "导出 CSV" not in full_lines[-1], "全量显示时不需要截断说明"
    assert describe_candidates(candidates, limit=25) == full_lines

    # 自定义 limit 同样如实
    two = describe_candidates(candidates, limit=2)
    assert len(two) == 4
    assert "前 2/25 条" in two[-1] and "其余 23 条" in two[-1]

    # 非法 limit 明确报错,不静默产出"前 0 条"之类的糊涂清单
    with pytest.raises(ValueError, match="limit"):
        describe_candidates(candidates, limit=0)


def test_probe_verdict_phrase_is_single_source_for_three_states():
    """三态判定词汇单一来源:页面与 CLI 同源同词汇;差异缺位时不编造方向。"""
    from src.workbench.learnability_probe import probe_verdict_phrase

    assert probe_verdict_phrase({"difference": 0.25}) == "零样本高于瞎猜基线"
    assert probe_verdict_phrase({"difference": 0.0}) == "零样本不低于瞎猜基线"
    assert (
        probe_verdict_phrase({"difference": -0.1}) == "零样本低于瞎猜基线——先核查提示格式与任务定义"
    )
    assert probe_verdict_phrase({"difference": None}) == "没有可比较的探针结果"


def test_describe_probe_verdict_lines_for_cli():
    """CLI 判定行:三组数字 + 三态短语,note 原文复述;裸记录给缺位句不编造。"""
    from src.workbench.learnability_probe import describe_probe_verdict

    result = {
        "zero_shot_accuracy": 0.5,
        "majority_baseline": 0.75,
        "difference": -0.25,
        "note": "样本量小,不能预测微调效果。",
    }
    lines = describe_probe_verdict(result)
    assert len(lines) == 2
    assert lines[0].startswith("可学性探针判定：基座零样本 50% vs 瞎猜多数类基线 75%")
    assert "差异 -25%" in lines[0]
    assert "零样本低于瞎猜基线——先核查提示格式与任务定义" in lines[0]
    assert lines[1] == "样本量小,不能预测微调效果。"

    # 裸记录(无可读字段)给缺位句,不编造数字
    assert describe_probe_verdict({}) == ["这份探针记录没有可读的判定内容。"]


def test_candidates_csv_export_is_excel_friendly():
    from src.workbench.learnability_probe import candidates_to_csv

    data = candidates_to_csv(
        [
            {
                "row_id": "r1",
                "data_label": "质量",
                "base_zero_shot": "物流",
                "user_blind_answer": None,
                "evidence": "弱信号",
            },
        ]
    )
    text = data.decode("utf-8-sig")
    assert "行ID" in text and "r1" in text and "弱信号" in text
    assert data.startswith(b"\xef\xbb\xbf")  # BOM: Excel 直接打开不乱码


def test_candidates_csv_export_covers_all_rows_beyond_one_page():
    """页面候选分页只影响展示;CSV 导出必须包含全部候选,不止当前页。"""
    from src.workbench.learnability_probe import candidates_to_csv

    candidates = [
        {
            "row_id": f"r{index:06d}",
            "data_label": "质量",
            "base_zero_shot": "物流",
            "user_blind_answer": None,
            "evidence": "弱信号",
        }
        for index in range(1, 26)
    ]
    text = candidates_to_csv(candidates).decode("utf-8-sig")
    lines = [line for line in text.splitlines() if line.strip()]
    assert len(lines) == 26  # 表头 + 全部 25 行,不是只有第一页的 20 行
    assert "r000025" in text


def test_saved_probe_can_be_loaded_back_per_dataset_version(store, tmp_path):
    """探针结果存盘后必须能按数据版本回读——重看结论不需要重新加载模型。"""
    from src.workbench.learnability_probe import load_latest_probe

    _, session = store
    root = tmp_path / "probes"
    assert load_latest_probe(root, session.dataset.version) is None  # 从未运行过

    result = probe_learnability(
        session, "/tmp/base", runtime_factory=_factory("质量"), sample_size=2
    )
    save_probe(root, result)
    loaded = load_latest_probe(root, session.dataset.version)
    assert loaded is not None
    assert loaded["zero_shot_accuracy"] == result["zero_shot_accuracy"]
    assert loaded["observations"] == result["observations"]

    # 数据版本不同(重物化后)不回读旧版本结果;损坏文件跳过不阻塞
    assert load_latest_probe(root, "other-version") is None
    (root / f"{session.dataset.version}-broken.json").write_text("{not json", encoding="utf-8")
    again = load_latest_probe(root, session.dataset.version)
    assert again is not None and again["kind"] == "learnability_probe"


def test_saved_probe_records_generation_time(store, tmp_path):
    """存盘记录自带生成时间:mtime 在复制/同步后会丢,证据文件自己说明何时产生。"""
    from datetime import datetime

    _, session = store
    result = probe_learnability(
        session, "/tmp/base", runtime_factory=_factory("yes"), sample_size=1
    )
    assert "saved_at" not in result  # 保存前不掺入
    path = save_probe(tmp_path / "probes", result)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["saved_at"] == result["saved_at"]
    datetime.fromisoformat(saved["saved_at"])  # 可解析的 ISO 时间戳
    assert saved["saved_at"].endswith("+00:00"), "带时区,不产生本地时间的歧义"


def test_load_latest_probe_record_returns_source_path(store, tmp_path):
    """回读连同记录文件路径一起返回:证据要能溯源到出处;兼容原 load_latest_probe。"""
    from src.workbench.learnability_probe import (
        load_latest_probe,
        load_latest_probe_record,
    )

    _, session = store
    root = tmp_path / "probes"
    assert load_latest_probe_record(root, session.dataset.version) is None

    path = save_probe(
        root,
        probe_learnability(session, "/tmp/base", runtime_factory=_factory("yes"), sample_size=1),
    )
    record = load_latest_probe_record(root, session.dataset.version)
    assert record is not None
    record_path, data = record
    assert record_path == path
    assert data["kind"] == "learnability_probe"
    assert load_latest_probe(root, session.dataset.version) == data


def test_cli_probe_show_reads_saved_result_without_rerunning(store, tmp_path, monkeypatch, capsys):
    """CLI learnability-probe-show 回读已存盘结果;没跑过探针时如实说明。"""
    import sys

    from scripts import data_intake

    service, session = store
    evaluation_root = tmp_path / "eval"  # 探针记录在 evaluation_root.parent / "probes"
    argv = [
        "data_intake.py",
        "--store",
        str(service.root),
        "--evaluation-root",
        str(evaluation_root),
        "learnability-probe-show",
        session.session_id,
    ]
    monkeypatch.setattr(sys, "argv", argv)
    assert data_intake.main() == 2  # 尚未运行过探针:明确告知,不伪造
    capsys.readouterr()

    result = probe_learnability(
        session, "/tmp/base", runtime_factory=_factory("质量"), sample_size=2
    )
    save_probe(evaluation_root.parent / "probes", result)
    monkeypatch.setattr(sys, "argv", argv)
    assert data_intake.main() == 0
    out, err = capsys.readouterr()
    assert json.loads(out)["zero_shot_accuracy"] == result["zero_shot_accuracy"]
    assert "最近一次已保存的探针结果" in err
    # 证据溯源:回读说明指出证据来自哪个记录文件、什么时候生成的
    assert result["saved_at"] in err and "记录文件" in err
    # 判定行与页面同源:三组数字 + 三态短语;结论先于候选清单或空清单说明
    assert "可学性探针判定：基座零样本" in err and "瞎猜多数类基线" in err
    assert "不能预测微调效果" in err, "note 原文复述,不改编"
    tail = err[err.index("可学性探针判定") :]
    assert ("标签问题候选" in tail) or ("没有发现值得优先核对的行" in tail), "判定行先于清单"


def _triage_result(answers, *, vocabulary=None, truncated=None, difference=-0.25):
    """构造低于基线的探针记录:answers 是逐条生成输出,truncated 同长布尔。"""
    flags = truncated if truncated is not None else [False] * len(answers)
    return {
        "difference": difference,
        "label_vocabulary": vocabulary,
        "observations": [
            {
                "row_id": f"r{i}",
                "expected": "yes",
                "generated": answer,
                "match": answer == "yes",
                "truncated": flag,
            }
            for i, (answer, flag) in enumerate(zip(answers, flags))
        ],
    }


def test_low_baseline_triage_direction_split_by_recorded_facts():
    """低于基线时的方向分辨:提示模板方向 vs 任务定义方向由记录内事实分流。"""
    from src.workbench.learnability_probe import low_baseline_triage_lines

    # 不低于基线(差异>=0)、差异缺位或没有观察:一行不发,不制造恐慌
    assert low_baseline_triage_lines({"difference": 0.1, "observations": [{}]}) == []
    assert low_baseline_triage_lines({"difference": None}) == []
    assert low_baseline_triage_lines(_triage_result(["no"], difference=0.0)) == []
    assert low_baseline_triage_lines({"difference": -0.1}) == []

    # 输出词汇不在标签全集 → 提示模板方向(模型没用任务的答案词汇作答)
    lines = low_baseline_triage_lines(
        _triage_result(["不知道", "拒绝回答"], vocabulary=["yes", "no"])
    )
    assert any("不在这份开发集的标签里出现过" in line for line in lines)
    assert any("提示模板" in line for line in lines)
    assert all("同一个输出" not in line for line in lines), "词汇行已解释同答,不重复"
    assert lines[-1].startswith("对号处理")

    # 全部未截断输出完全相同且在词汇内 → 模板没讲清与输入缺区分信息两方向都在
    lines = low_baseline_triage_lines(
        _triage_result(["yes", "yes", "yes"], vocabulary=["yes", "no"])
    )
    assert any("同一个输出「yes」" in line for line in lines)
    assert any("没有按输入区分作答" in line for line in lines)
    assert all("不在这份开发集的标签" not in line for line in lines)

    # 用任务词汇、按输入作答仍低于基线 → 更像任务定义/标注口径的问题
    lines = low_baseline_triage_lines(_triage_result(["yes", "no"], vocabulary=["yes", "no"]))
    assert any("任务定义或标注口径" in line for line in lines)
    assert len(lines) == 2  # 方向行 + 对号处理,不灌多余行


def test_low_baseline_triage_truncation_first_and_degrade_for_old_records():
    """截断优先提示先加长度重测;旧记录缺 label_vocabulary 时如实降级不报错。"""
    from src.workbench.learnability_probe import low_baseline_triage_lines

    # 有截断:被截断的输出已按不匹配计,先加大 max_new_tokens 重测再谈方向
    lines = low_baseline_triage_lines(
        _triage_result(["", "no"], vocabulary=["yes", "no"], truncated=[True, False])
    )
    assert any("条生成被截断" in line and "加大 max_new_tokens" in line for line in lines)
    # 全部截断:只有截断行 + 对号处理,不对没写完的输出编造方向
    lines = low_baseline_triage_lines(
        _triage_result(["", ""], vocabulary=["yes", "no"], truncated=[True, True])
    )
    assert len(lines) == 2
    assert all("任务定义或标注口径" not in line for line in lines)

    # 旧记录没有 label_vocabulary:跳过词汇检查,同答/其余分辨照常
    lines = low_baseline_triage_lines(_triage_result(["同一个答案", "同一个答案"]))
    assert any("同一个输出「同一个答案」" in line for line in lines)
    lines = low_baseline_triage_lines(_triage_result(["yes", "no"]))
    assert any("任务定义或标注口径" in line for line in lines)


def test_probe_stores_label_vocabulary_and_weak_signal_threshold(store):
    """记录携带标签全集供方向分辨;弱信号证据带改标签门槛,三处同源不冒充溯源。"""
    from src.workbench.learnability_probe import describe_candidates

    _, session = store
    result = probe_learnability(
        session, "/tmp/base", runtime_factory=_factory("绝不正确的答案"), sample_size=4
    )
    labels = {row["output"] for row in _labels(session)}
    assert result["label_vocabulary"] == sorted(labels), "标签全集按字典序存进记录"
    weak = [c for c in result["label_error_candidates"] if not c["user_blind_answer"]]
    assert weak and all("人工核对后仍不认同才修正数据" in c["evidence"] for c in weak)
    assert all("弱信号" in c["evidence"] for c in weak)
    assert all("原始来源行" not in c["evidence"] for c in weak), "门槛不冒充溯源建议"
    # 改标签门槛三处同源:候选 note、CLI 清单表头、弱信号证据列
    assert "人工核对后仍不认同才修正数据" in result["candidates_note"]
    header = describe_candidates(result["label_error_candidates"])[0]
    assert "人工核对后仍不认同才修正数据" in header
