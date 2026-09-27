"""Real business compositions preserve raw inputs, ordering and qualified origins."""

import json

import pytest

from src.workbench.composition import CompositionRecipe, compose_sources
from src.workbench.intake_models import DataRecipe
from src.workbench.recipes import preview_recipe
from src.workbench.sources import read_source


def source(records, *, scope="sample"):
    return read_source(
        "source.jsonl",
        "\n".join(json.dumps(row, ensure_ascii=False) for row in records).encode(),
        scope=scope,
    )


def compose(sources, *steps):
    return compose_sources(sources, CompositionRecipe(base_source="main", steps=list(steps)))


def join(**overrides):
    return {
        "operation": "join",
        "right_source": "labels",
        "left_on": ["ticket"],
        "right_on": ["id"],
        "how": "left",
        "cardinality": "many_to_one",
        "prefix": "label_",
        **overrides,
    }


def test_ticket_and_quality_labels_join_into_real_training_preview():
    main = source(
        [{"ticket": "001", "description": "杯子破损"}, {"ticket": "002", "description": "物流慢"}]
    )
    labels = source([{"id": "001", "category": "质量"}, {"id": "002", "category": "物流"}])
    before = main.model_dump()
    result = compose({"main": main, "labels": labels}, join())
    assert result.can_confirm
    assert main.model_dump() == before
    preview = preview_recipe(
        result.source,
        DataRecipe(
            instruction="判断类别",
            inputs=[{"column": "description", "label": "描述"}],
            targets=[{"column": "label_category", "label": "类别"}],
        ),
    )
    assert preview.rows[0].input == "描述: 杯子破损"
    assert preview.rows[0].target == "质量"
    assert {(ref.source_digest, ref.row_id) for ref in result.origins["r000001"]} == {
        (main.digest, "r000001"),
        (labels.digest, "r000001"),
    }
    assert len(result.source.digest) == 64
    assert compose({"labels": labels, "main": main}, join()).source.digest == result.source.digest


def test_order_items_explode_extract_and_join_catalog():
    main = source(
        [{"order": "O1", "items": [{"sku": "A", "quantity": 2}, {"sku": "B", "quantity": 1}]}]
    )
    catalog = source([{"sku": "A", "name": "杯子"}, {"sku": "B", "name": "勺子"}])
    result = compose(
        {"main": main, "catalog": catalog},
        {"operation": "explode", "column": "items", "target_column": "item", "empty": "error"},
        {"operation": "extract", "column": "item", "path": ["sku"], "target_column": "sku"},
        join(right_source="catalog", left_on=["sku"], right_on=["sku"], prefix="product_"),
    )
    assert result.can_confirm and result.requires_review
    assert [row.values["product_name"] for row in result.source.rows] == ["杯子", "勺子"]
    assert [row.values["item"]["quantity"] for row in result.source.rows] == [2, 1]
    assert len(main.rows) == 1
    assert all(
        result.origins[row.row_id][0].source_digest == main.digest for row in result.source.rows
    )


@pytest.mark.parametrize("how,expected_rows", [("left", 2), ("inner", 1)])
def test_unmatched_rows_are_reported_with_sources(how, expected_rows):
    main = source([{"ticket": "1"}, {"ticket": "2"}])
    labels = source([{"id": "1"}, {"id": "3"}])
    result = compose({"main": main, "labels": labels}, join(how=how))
    assert len(result.source.rows) == expected_rows
    assert result.requires_review
    unmatched = next(i for i in result.issues if i.code == "unmatched_left")
    assert unmatched.origins[0].row_id == "r000002"
    assert unmatched.origins[0].source_digest == main.digest
    assert any(i.code == "unused_right" for i in result.issues)


def test_join_never_silently_expands_undeclared_cardinality():
    sources = {
        "main": source([{"ticket": "1"}]),
        "labels": source([{"id": "1", "answer": "A"}, {"id": "1", "answer": "B"}]),
    }
    blocked = compose(sources, join())
    assert not blocked.can_confirm
    assert len(blocked.source.rows) == 1
    assert any(i.code == "cardinality_violation" for i in blocked.issues)
    expanded = compose(sources, join(cardinality="one_to_many"))
    assert expanded.can_confirm and expanded.requires_review
    assert len(expanded.source.rows) == 2


@pytest.mark.parametrize(
    "main,step,code",
    [
        ([{"ticket": ""}], join(), "empty_join_key"),
        ([{"ticket": "1", "label_id": "original"}], join(), "column_collision"),
        ([{"other": "1"}], join(), "missing_columns"),
    ],
)
def test_invalid_join_preserves_original_rows_and_blocks(main, step, code):
    sources = {"main": source(main), "labels": source([{"id": "1"}])}
    result = compose(sources, step)
    assert not result.can_confirm
    assert result.source.rows[0].values == main[0]
    assert any(i.code == code for i in result.issues)


@pytest.mark.parametrize(
    "value,empty,code,severity",
    [
        ("[1,2]", "keep", "not_list", "blocking"),
        ([], "error", "empty_list", "blocking"),
        ([], "keep", "empty_list", "review"),
    ],
)
def test_explode_does_not_parse_strings_or_drop_empty_rows(value, empty, code, severity):
    result = compose(
        {"main": source([{"items": value}])},
        {"operation": "explode", "column": "items", "target_column": "item", "empty": empty},
    )
    assert len(result.source.rows) == 1
    assert result.source.rows[0].values["items"] == value
    assert any(i.code == code and i.severity == severity for i in result.issues)


def test_extract_handles_explicit_list_path_and_reports_missing_path():
    result = compose(
        {"main": source([{"payload": {"answers": ["yes"]}}, {"payload": {"answers": []}}])},
        {
            "operation": "extract",
            "column": "payload",
            "path": ["answers", 0],
            "target_column": "answer",
        },
    )
    assert [row.values["answer"] for row in result.source.rows] == ["yes", None]
    assert not result.can_confirm
    assert result.issues[0].origins[0].row_id == "r000002"


CONVERSATION = {
    "operation": "conversation",
    "group_columns": ["session"],
    "order_column": "turn",
    "order_type": "numeric",
    "role_column": "role",
    "content_column": "text",
}


def test_conversation_pairs_never_cross_sessions_or_see_future_messages():
    main = source(
        [
            {"session": "A", "turn": "4", "role": "assistant", "text": "第二次回答"},
            {"session": "B", "turn": "1", "role": "user", "text": "B的问题"},
            {"session": "A", "turn": "1", "role": "user", "text": "A的问题"},
            {"session": "A", "turn": "2", "role": "assistant", "text": "第一次回答"},
            {"session": "B", "turn": "2", "role": "assistant", "text": "B的回答"},
            {"session": "A", "turn": "3", "role": "user", "text": "A的追问"},
        ]
    )
    result = compose({"main": main}, CONVERSATION)
    assert result.can_confirm and len(result.source.rows) == 3
    first, second = [row for row in result.source.rows if row.values["session"] == "A"]
    assert json.loads(first.values["history"]) == [{"role": "user", "content": "A的问题"}]
    assert first.values["answer"] == "第一次回答"
    assert "B的问题" not in second.values["history"]
    assert "第二次回答" not in second.values["history"]
    assert [item["content"] for item in json.loads(second.values["history"])] == [
        "A的问题",
        "第一次回答",
        "A的追问",
    ]
    assert {ref.row_id for ref in result.origins[first.row_id]} == {"r000003", "r000004"}
    assert len(main.rows) == 6


@pytest.mark.parametrize(
    "override,code",
    [
        ({"turn": None}, "invalid_order"),
        ({"turn": 1}, "duplicate_order"),
        ({"role": "agent"}, "invalid_message"),
        ({"role": ["assistant"]}, "invalid_message"),
    ],
)
def test_conversation_invalid_order_or_role_blocks_without_guessing(override, code):
    rows = [
        {"session": "A", "turn": 1, "role": "user", "text": "问题"},
        {"session": "A", "turn": 2, "role": "assistant", "text": "答案", **override},
    ]
    result = compose({"main": source(rows)}, CONVERSATION)
    assert not result.can_confirm
    assert result.source.rows == []
    assert any(i.code == code for i in result.issues)


def test_scope_uses_only_participating_sources():
    sources = {
        "main": source([{"nested": {"value": "A"}}], scope="full"),
        "unused": source([{"other": 1}]),
    }
    result = compose(
        sources,
        {"operation": "extract", "column": "nested", "path": ["value"], "target_column": "answer"},
    )
    assert result.source.scope == "full"
