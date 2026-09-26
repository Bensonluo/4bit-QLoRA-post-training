"""数据向导体检层测试：七项检查的通过与失败分支。"""

from src.data.wizard.checks import (
    POSITION_DOMINANCE,
    check_ambiguous_query,
    check_candidate_counts,
    check_difficulty_balance,
    check_dropped_rows,
    check_duplicates,
    check_position_bias,
    check_split_leakage,
    run_checks,
)
from src.data.wizard.templates import Candidate, MatchingSample, RowIssue


def sample(
    query: str = "q",
    standard: str = "S",
    code: str | None = "C1",
    difficulty: str = "easy",
    label_index: int = 0,
    n_candidates: int = 4,
    row: int = 1,
) -> MatchingSample:
    candidates = [Candidate(f"neg{i}", None, False) for i in range(n_candidates)]
    candidates[label_index] = Candidate(standard, code, True)
    return MatchingSample(
        query=query,
        standard_name=standard,
        code=code,
        entity_type="entity",
        difficulty=difficulty,
        candidates=candidates,
        source_row=row,
    )


class TestDroppedRows:
    def test_no_drops(self) -> None:
        r = check_dropped_rows([])
        assert r.passed

    def test_few_drops(self) -> None:
        r = check_dropped_rows([RowIssue(2, "标准名为空")])
        assert not r.passed
        assert r.severity == "warning"
        assert "第 2 行" in r.message

    def test_many_drops_summary(self) -> None:
        drops = [RowIssue(i, "标准名为空") for i in range(2, 8)]
        r = check_dropped_rows(drops)
        assert "等 6 行" in r.message
        assert r.details["count"] == 6


class TestDuplicates:
    def test_clean(self) -> None:
        r = check_duplicates({"train": [sample("a", "S1"), sample("b", "S2")]})
        assert r.passed

    def test_duplicates_detected_with_examples(self) -> None:
        splits = {
            "train": [sample(f"q{i}", f"S{i}") for i in range(4)]
            + [sample("q0", "S0"), sample("q1", "S1")],  # 2 条重复
            "val": [sample("v", "S9")],
            "test": [sample("t", "S8")],
        }
        r = check_duplicates(splits)
        assert not r.passed
        assert r.details["count"] == 2
        assert "[train] q0 → S0" in r.details["examples"]

    def test_more_than_three_examples_capped(self) -> None:
        splits = {
            "train": [sample("q", "S")] * 5,
            "val": [],
            "test": [],
        }
        r = check_duplicates(splits)
        assert r.details["count"] == 4
        assert len(r.details["examples"]) == 3


class TestSplitLeakage:
    def test_disjoint(self) -> None:
        splits = {
            "train": [sample("a", "S1", "C1")],
            "val": [sample("b", "S2", "C2")],
            "test": [sample("c", "S3", "C3")],
        }
        assert check_split_leakage(splits).passed

    def test_overlap_detected(self) -> None:
        splits = {
            "train": [sample("a", "S1", "C1")],
            "val": [sample("b", "S2", "C2")],
            "test": [sample("c", "S1", "C1")],  # 与 train 泄漏
        }
        r = check_split_leakage(splits)
        assert not r.passed
        assert r.severity == "error"
        assert "泄漏" in r.message

    def test_overlap_without_code_uses_names(self) -> None:
        splits = {
            "train": [sample("a", "共享名", None)],
            "val": [sample("b", "S2", "C2")],
            "test": [sample("c", "共享名", None)],
        }
        assert not check_split_leakage(splits).passed


class TestAmbiguousQuery:
    def test_unique_queries_pass(self) -> None:
        splits = {"train": [sample("甲", "S1", "C1"), sample("乙", "S2", "C2")]}
        r = check_ambiguous_query(splits)
        assert r.passed
        assert r.severity == "error"
        assert "自洽" in r.message

    def test_same_query_two_standards_fails(self) -> None:
        # 连锁门店共用简称 → 同一查询指向多个分店，标签自相矛盾
        splits = {
            "train": [
                sample("和平大药房", "和平大药房(一店)", "P1"),
                sample("和平大药房", "和平大药房(二店)", "P2"),
                sample("乙", "S2", "C2"),
            ]
        }
        r = check_ambiguous_query(splits)
        assert not r.passed
        assert r.severity == "error"
        assert "标签矛盾" in r.message
        assert r.details["count"] == 1

    def test_same_query_same_standard_across_splits_passes(self) -> None:
        # 同一查询+同一实体出现在不同 split 是泄漏问题（另一项检查管），不算歧义
        splits = {
            "train": [sample("甲", "S1", "C1")],
            "test": [sample("甲", "S1", "C1")],
        }
        assert check_ambiguous_query(splits).passed

    def test_included_in_run_checks(self) -> None:
        results = run_checks({"train": [sample()]}, [])
        assert "ambiguous_query" in [r.check_id for r in results]


class TestPositionBias:
    def make(self, label_index: int, n: int, split: str = "train"):
        return {split: [sample(f"q{i}", f"S{i}", label_index=label_index) for i in range(n)]}

    def test_uniform_passes(self) -> None:
        splits = {"train": []}
        for i in range(12):
            splits["train"].append(sample(f"q{i}", f"S{i}", label_index=i % 4))
        assert check_position_bias(splits).passed

    def test_dominant_position_warns(self) -> None:
        splits = self.make(label_index=0, n=12)
        r = check_position_bias(splits)
        assert not r.passed
        assert "位置偏差" in r.message
        assert r.details["position"] == 1

    def test_too_few_samples_skipped(self) -> None:
        splits = self.make(label_index=0, n=5)  # < POSITION_MIN_SAMPLES
        assert check_position_bias(splits).passed

    def test_unlabeled_candidates_ignored(self) -> None:
        s = sample(n_candidates=3)
        s.candidates = [Candidate("x", None, False)] * 3  # 无正例
        r = check_position_bias({"train": [s] * 12})
        assert r.passed  # 无可统计位置 → 不告警

    def test_later_split_not_worse_keeps_first(self) -> None:
        # train 100% 集中，val 均匀分布 → 后续 split 未刷新 worst 的循环分支
        splits = {
            "train": [sample(f"t{i}", f"T{i}", label_index=0) for i in range(10)],
            "val": [sample(f"v{i}", f"V{i}", label_index=i % 4) for i in range(12)],
        }
        r = check_position_bias(splits)
        assert not r.passed
        assert r.details["split"] == "train"
        assert r.details["position"] == 1

    def test_threshold_boundary(self) -> None:
        # 5/12 ≈ 41.7% > 40% 阈值 → 告警；确保阈值参与判定
        assert POSITION_DOMINANCE == 0.4
        splits = {"train": []}
        for i in range(12):
            idx = 0 if i < 5 else (i % 4) + 1
            splits["train"].append(sample(f"q{i}", f"S{i}", label_index=idx, n_candidates=6))
        assert not check_position_bias(splits).passed


class TestDifficultyBalance:
    def test_empty_splits(self) -> None:
        r = check_difficulty_balance({"train": [], "val": [], "test": []})
        assert r.passed
        assert "无样本" in r.message

    def test_missing_levels_reported(self) -> None:
        r = check_difficulty_balance({"train": [sample("a", "S1")]})
        assert r.passed
        assert "缺少" in r.message
        assert "medium" in r.message

    def test_all_levels_present(self) -> None:
        splits = {
            "train": [sample("a", "S1", difficulty="easy"), sample("b", "S2", difficulty="hard")],
            "val": [sample("c", "S3", difficulty="medium")],
            "test": [],
        }
        r = check_difficulty_balance(splits)
        assert r.passed
        assert "分层评测" in r.message


class TestCandidateCounts:
    def test_all_good(self) -> None:
        splits = {"train": [sample(n_candidates=5)], "val": [], "test": []}
        assert check_candidate_counts(splits).passed

    def test_single_candidate_is_error(self) -> None:
        s = sample(n_candidates=1)
        splits = {"train": [s], "val": [], "test": []}
        r = check_candidate_counts(splits)
        assert not r.passed
        assert r.severity == "error"
        assert "候选数不足" in r.message


class TestRunChecks:
    def test_order_and_count(self) -> None:
        splits = {
            "train": [sample("a", "S1", "C1"), sample("b", "S2", "C2", difficulty="hard")],
            "val": [sample("c", "S3", "C3", difficulty="medium")],
            "test": [sample("d", "S4", "C4")],
        }
        results = run_checks(splits, [])
        assert [r.check_id for r in results] == [
            "dropped_rows",
            "split_leakage",
            "ambiguous_query",
            "position_bias",
            "candidate_counts",
            "duplicates",
            "difficulty_balance",
        ]
        assert all(r.passed for r in results)
