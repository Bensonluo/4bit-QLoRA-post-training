"""CLI keeps named originals through actual multi-table full-data validation."""

import json

import pytest

from src.workbench.intake_models import IntakeAnalysis
from src.workbench.intake_service import IntakeService
from tests.unit.test_full_data_cli import invoke

MAIN = "ticket,text\n1,杯子破了\n2,物流慢\n3,配件损坏\n".encode()
LABELS = "id,category\n1,质量\n2,物流\n3,质量\n".encode()
FULL_MAIN = "ticket,text\n10,新杯子损坏\n11,新物流延迟\n12,新配件缺陷\n".encode()
FULL_LABELS = "id,category\n10,质量\n11,物流\n12,质量\n".encode()


def combined_plan():
    return IntakeAnalysis.model_validate(
        {
            "task": {
                "goal": "判断工单类别",
                "usage_input": "客户描述",
                "desired_output": "人工审核类别",
                "row_meaning": "一个工单",
                "supervision_source": "labels中人工审核类别",
                "success_criteria": ["类别正确"],
                "field_roles": [
                    {"column": "ticket", "role": "group", "reason": "工单编号"},
                    {
                        "column": "text",
                        "role": "input",
                        "reason": "首次咨询已知",
                        "available_at_prediction": True,
                    },
                    {"column": "label_id", "role": "unused", "reason": "关联核对"},
                    {"column": "label_category", "role": "target", "reason": "人工审核答案"},
                ],
            },
            "composition": {
                "base_source": "main",
                "steps": [
                    {
                        "operation": "join",
                        "right_source": "labels",
                        "left_on": ["ticket"],
                        "right_on": ["id"],
                        "how": "left",
                        "cardinality": "one_to_one",
                        "prefix": "label_",
                    }
                ],
            },
            "recipe": {
                "instruction": "判断工单类别",
                "inputs": [{"column": "text", "label": "描述"}],
                "targets": [
                    {"column": "label_category", "label": "类别", "value_kind": "categorical"}
                ],
                "group_columns": ["ticket"],
            },
            "findings": [],
            "training_approach": "SFT",
            "next_steps": ["核对类别映射"],
        }
    )


def composition_session(service, *, scope="sample"):
    session = service.create("判断工单类别", "tickets.csv", MAIN, scope=scope)
    session = service.add_source(
        session.session_id, session.revision, "labels", "labels.csv", LABELS, scope=scope
    )
    session = service.apply_analysis(session, combined_plan())
    return service.confirm(session.session_id, session.revision)


def test_cli_add_source_then_validate_actual_full_tables(tmp_path):
    service = IntakeService(tmp_path / "intake")
    session = service.create("判断工单类别", "tickets.csv", MAIN)
    labels = tmp_path / "labels.csv"
    labels.write_bytes(LABELS)
    result = invoke(
        service,
        "add-source",
        session.session_id,
        "--revision",
        session.revision,
        "--alias",
        "labels",
        "--input",
        labels,
        "--description",
        "人工审核类别，id 对应工单编号",
    )
    assert result.returncode == 0, result.stderr
    session = service.load(session.session_id)
    assert set(session.sources) == {"main", "labels"}
    assert "人工审核类别" in session.answers[-1]["answer"]
    session = service.apply_analysis(session, combined_plan())
    session = service.confirm(session.session_id, session.revision)
    main_path, labels_path = tmp_path / "full-main.csv", tmp_path / "full-labels.csv"
    main_path.write_bytes(FULL_MAIN)
    labels_path.write_bytes(FULL_LABELS)
    result = invoke(
        service,
        "full-sources",
        session.session_id,
        "--revision",
        session.revision,
        "--source",
        f"main={main_path}",
        "--source",
        f"labels={labels_path}",
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["full_data"]["preview"]["counts"]["ready"] == 3
    assert set(payload["full_data"]["sources"]) == {"main", "labels"}
    assert payload["sources"]["main"]["scope"] == "sample"
    assert "review_full_data" in result.stderr
    assert payload["full_data"]["preview"]["rows"][0]["target"] == "质量"


def test_cli_does_not_promote_sample_to_full(tmp_path):
    service = IntakeService(tmp_path / "intake")
    session = composition_session(service)
    result = invoke(service, "full-sources", session.session_id, "--revision", session.revision)
    assert result.returncode == 2
    assert "不能将样例自动当作全量" in result.stderr


def test_awaiting_full_validation_tail_notes_full_sources_and_delegated_validate_walks(tmp_path):
    """已声明全量态的尾行带上 full-sources 变体注(R90 收口 R85 nit-②),且该态出口
    实际走通(R89 同款「点名→走通」标准):组合任务要停在 awaiting_full_validation,
    每份来源必须都已声明全量(compose_sources 只在全部来源 scope=full 时才给组合源
    full,composition.py:488)——full-validate 省略 --input 委托组合校验,直接复用
    每份已声明全量来源,直达 review_full_data;同态带 --input 上传组合结果会被拒并
    指路按别名提供,这正是变体注的契约依据(此前无 pin)。样例混入的组合会话停在
    awaiting_full_data,那态的 full-sources 出口另有测试,不在此重复证明。
    """
    from src.workbench.intake_service import next_action, next_action_phrase

    service = IntakeService(tmp_path / "intake")
    session = composition_session(service, scope="full")
    assert next_action(session) == "awaiting_full_validation"
    # 尾行短语(单一来源)三要素齐:点名 full-validate、写明省略 --input 的复用条件、
    # 带上多资料变体注。
    phrase = next_action_phrase("awaiting_full_validation")
    assert "full-validate" in phrase and "省略 --input" in phrase and "full-sources" in phrase
    # 变体注的契约依据:组合任务 full-validate 带 --input 上传组合结果被拒,按别名
    # 提供才是正路(组合分支的拒收信息,此前全仓无 pin)。
    combined_path = tmp_path / "combined.csv"
    combined_path.write_bytes("ticket,text,label_category\n10,新杯子损坏,质量\n".encode())
    result = invoke(
        service,
        "full-validate",
        session.session_id,
        "--revision",
        session.revision,
        "--input",
        combined_path,
    )
    assert result.returncode == 2
    assert "不能把组合结果当原始来源上传" in result.stderr
    # 走通:省略 --input 委托组合校验,复用每份已声明全量来源,直达 review_full_data。
    result = invoke(service, "full-validate", session.session_id, "--revision", session.revision)
    assert result.returncode == 0, result.stderr
    assert "review_full_data" in result.stderr
    session = service.load(session.session_id)
    assert next_action(session) == "review_full_data"
    assert set(session.full_data.sources) == {"main", "labels"}


@pytest.mark.parametrize("source_args", [["broken"], ["main=FILE", "main=FILE"], ["main=FILE"]])
def test_cli_invalid_full_source_arguments_leave_session_unchanged(tmp_path, source_args):
    service = IntakeService(tmp_path / "intake")
    session = composition_session(service)
    path = tmp_path / "main.csv"
    path.write_bytes(FULL_MAIN)
    args = [
        value for item in source_args for value in ("--source", item.replace("FILE", str(path)))
    ]
    result = invoke(
        service, "full-sources", session.session_id, "--revision", session.revision, *args
    )
    assert result.returncode == 2
    assert service.load(session.session_id).revision == session.revision
    assert "Traceback" not in result.stderr


def test_replacing_named_source_stops_old_full_source_from_reuse(tmp_path):
    """替换具名原始资料后,旧全量来源不能再冒充当前资料被复用校验。

    此前 add_source 只把全量报告标记为 stale,full_data.sources 仍保留被替换前
    的旧来源;「验证已提供的全部全量资料」会继续校验旧字节并当成当前文件通过。
    """
    from src.workbench.intake_service import next_action

    service = IntakeService(tmp_path / "intake")
    session = composition_session(service)
    session = service.validate_full_sources(
        session.session_id,
        session.revision,
        {"main": ("full-main.csv", FULL_MAIN), "labels": ("full-labels.csv", FULL_LABELS)},
    )
    assert next_action(session) == "review_full_data"
    old_labels_digest = session.full_data.sources["labels"].digest
    labels_v2 = "id,category\n1,售后\n2,安装\n3,售后\n".encode()
    session = service.add_source(
        session.session_id, session.revision, "labels", "labels-v2.csv", labels_v2, scope="full"
    )
    assert "labels" not in (session.full_data.sources or {})
    assert "main" in session.full_data.sources
    session = service.apply_analysis(session, combined_plan())
    session = service.confirm(session.session_id, session.revision)
    with pytest.raises(ValueError, match="labels"):
        service.validate_full_sources(session.session_id, session.revision)
    # 明确重新提供每份全量文件后仍可正常验证,修复没有把合法路径堵死。
    labels_v2_full = "id,category\n10,售后\n11,安装\n12,售后\n".encode()
    session = service.validate_full_sources(
        session.session_id,
        session.revision,
        {"main": ("full-main.csv", FULL_MAIN), "labels": ("labels-v2.csv", labels_v2_full)},
    )
    assert next_action(session) == "review_full_data"
    import hashlib

    assert session.full_data.sources["labels"].digest == hashlib.sha256(labels_v2_full).hexdigest()
    assert session.full_data.sources["labels"].digest != old_labels_digest
    assert {row.target for row in session.full_data.preview.rows} == {"售后", "安装"}


def _workbook_bytes(*sheets):
    """sheets: (名称, 表头行, 数据行) 三元组；openpyxl 现场生成多 Sheet 工作簿。"""
    import io

    from openpyxl import Workbook

    workbook = Workbook()
    for index, (name, header, rows) in enumerate(sheets):
        sheet = workbook.active if index == 0 else workbook.create_sheet()
        sheet.title = name
        sheet.append(header)
        for row in rows:
            sheet.append(row)
    buffer = io.BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


def test_cli_full_sources_sheet_selects_each_excels_worksheet(tmp_path):
    """full-sources 的 --sheet 与 create/full-validate 对称：每份 Excel 各自指定 sheet。

    main 的真实数据在第二个 sheet（首 sheet 是说明），不指定 --sheet 会读错表；
    labels 用 1 起始序号指定，覆盖名称与序号两种写法。
    """
    service = IntakeService(tmp_path / "intake")
    session = composition_session(service)
    main_path, labels_path = tmp_path / "full-main.xlsx", tmp_path / "full-labels.xlsx"
    main_path.write_bytes(
        _workbook_bytes(
            ("说明", ("备注",), [("全量数据在下一个 sheet",)]),
            (
                "工单表",
                ("ticket", "text"),
                [("10", "新杯子损坏"), ("11", "新物流延迟"), ("12", "新配件缺陷")],
            ),
        )
    )
    labels_path.write_bytes(
        _workbook_bytes(
            ("类别表", ("id", "category"), [("10", "质量"), ("11", "物流"), ("12", "质量")]),
            ("说明", ("备注",), [("这份资料数据在第一个 sheet",)]),
        )
    )
    result = invoke(
        service,
        "full-sources",
        session.session_id,
        "--revision",
        session.revision,
        "--source",
        f"main={main_path}",
        "--source",
        f"labels={labels_path}",
        "--sheet",
        "main=工单表",
        "--sheet",
        "labels=1",
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["full_data"]["preview"]["counts"]["ready"] == 3
    assert payload["full_data"]["sources"]["main"]["sheet"] == "工单表"
    assert payload["full_data"]["sources"]["labels"]["sheet"] == "类别表"
    assert {row["target"] for row in payload["full_data"]["preview"]["rows"]} == {"质量", "物流"}
    assert "review_full_data" in result.stderr


def test_cli_add_source_sheet_selects_second_worksheet(tmp_path):
    """add-source 的 --sheet 与 create/full-validate/full-sources 对称。

    补充资料的真实数据在第二个 sheet（首 sheet 是说明），不指定 --sheet 会读错表；
    按名称指定后，会话里该资料读到的就是工单数据，画像如实标注按指定读取。
    """
    service = IntakeService(tmp_path / "intake")
    session = service.create("判断工单类别", "tickets.csv", MAIN)
    labels_path = tmp_path / "labels.xlsx"
    labels_path.write_bytes(
        _workbook_bytes(
            ("说明", ("备注",), [("类别数据在下一个 sheet",)]),
            ("类别表", ("id", "category"), [("1", "质量"), ("2", "物流"), ("3", "质量")]),
        )
    )
    result = invoke(
        service,
        "add-source",
        session.session_id,
        "--revision",
        session.revision,
        "--alias",
        "labels",
        "--input",
        labels_path,
        "--sheet",
        "类别表",
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    sources = payload["sources"]
    assert sources["labels"]["columns"] == ["id", "category"], sources["labels"]
    assert sources["labels"]["sheet"] == "类别表"
    assert "按指定读取「类别表」" in sources["labels"]["sheet_note"]
    # 不指定 --sheet 时同一份文件读到的是说明表——同摘要不同选择不串味
    result_default = invoke(
        service,
        "add-source",
        session.session_id,
        "--revision",
        session.revision + 1,
        "--alias",
        "labels",
        "--input",
        labels_path,
    )
    assert result_default.returncode == 0, result_default.stderr
    default_sources = json.loads(result_default.stdout)["sources"]
    assert default_sources["labels"]["columns"] == ["备注"], default_sources["labels"]
    assert "仅读取第一个「说明」" in default_sources["labels"]["sheet_note"]


@pytest.mark.parametrize(
    "sheet_args,with_files",
    [
        (["broken"], True),
        (["main=工单表", "main=类别表"], True),
        (["ghost=工单表"], True),
        (["main=工单表"], False),
    ],
)
def test_cli_invalid_full_sources_sheet_arguments_leave_session_unchanged(
    tmp_path, sheet_args, with_files
):
    service = IntakeService(tmp_path / "intake")
    session = composition_session(service)
    args = [value for item in sheet_args for value in ("--sheet", item)]
    if with_files:
        main_path, labels_path = tmp_path / "full-main.csv", tmp_path / "full-labels.csv"
        main_path.write_bytes(FULL_MAIN)
        labels_path.write_bytes(FULL_LABELS)
        args += [
            "--source",
            f"main={main_path}",
            "--source",
            f"labels={labels_path}",
        ]
    result = invoke(
        service, "full-sources", session.session_id, "--revision", session.revision, *args
    )
    assert result.returncode == 2
    assert service.load(session.session_id).revision == session.revision
    assert "Traceback" not in result.stderr
