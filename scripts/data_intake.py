#!/usr/bin/env python3
"""CLI for the same business-first data intake service used by the workbench."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent.intake import CompatibleChatClient
from src.agent.providers import PROVIDERS, AgentSettings, check_connection, load_settings
from src.workbench.intake_service import IntakeService, next_action

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _suite_reference(root: Path, suite_id: str) -> dict:
    from src.workbench.evaluation_suites import EvalSuiteService

    suites = EvalSuiteService(root)
    reference = suites.get(suite_id)
    suites.verify(reference)
    return reference


def _scoring_reference(root: Path, scoring_id: str, session_id: str) -> dict:
    from src.workbench.business_scoring import ScoringService

    record = ScoringService(root).get(scoring_id)
    if record["session_id"] != session_id or record["status"] != "confirmed":
        raise ValueError("请选择当前业务任务已明确确认的评分规则。")
    return {
        "root": str(Path(root).resolve()),
        "scoring_id": scoring_id,
        "spec_digest": record["spec_digest"],
    }


def _client(args: argparse.Namespace, *, probe: bool = False) -> CompatibleChatClient:
    settings = load_settings(args.agent_config_path)
    base_url = (
        (args.base_url if args.base_url is not None else settings.base_url).strip().rstrip("/")
    )
    key = os.environ.get("TUNESMITH_AGENT_API_KEY", "")
    if key and base_url != settings.base_url:
        raise ValueError(
            "临时 API 地址与当前密钥配置的地址不同。请配套设置 "
            "TUNESMITH_AGENT_BASE_URL、TUNESMITH_AGENT_MODEL 和 TUNESMITH_AGENT_API_KEY 后重试，"
            "避免把原服务密钥发送到另一个地址。"
        )
    return CompatibleChatClient(
        base_url,
        args.model if args.model is not None else settings.model,
        key,
        allow_remote=probe or args.allow_remote_data,
    )


def _save_config(args: argparse.Namespace) -> None:
    # Saving public settings must not persist environment overrides accidentally.
    saved = load_settings(args.agent_config_path, environ={})
    provider = args.provider if args.provider is not None else saved.provider
    defaults = PROVIDERS[provider] if provider != saved.provider else saved
    settings = AgentSettings(
        provider=provider,
        base_url=(args.base_url if args.base_url is not None else defaults.base_url)
        .strip()
        .rstrip("/"),
        model=(args.model if args.model is not None else defaults.model).strip(),
    )
    settings.save(args.agent_config_path)
    print(json.dumps(asdict(settings), ensure_ascii=False, indent=2))
    print("已保存公共配置；密钥仅从环境变量读取。环境配置优先于此文件。", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description="业务目标＋样例 → 数据诊断与真实预览")
    parser.add_argument("--store", default="outputs/workbench/intake")
    parser.add_argument(
        "--scoring-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/workbench/business-scoring",
        help="自定义业务评分规则目录",
    )
    parser.add_argument(
        "--acceptance-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/workbench/acceptance",
        help="独立业务验收记录目录",
    )
    parser.add_argument(
        "--plan-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/workbench/training-plans",
        help="Agent 推荐训练方案目录",
    )
    parser.add_argument(
        "--iteration-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/workbench/iterations",
        help="改进轮次与决策记录目录",
    )
    parser.add_argument(
        "--suite-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/workbench/evaluation-suites",
        help="固定开发集和测试题集保存目录",
    )
    parser.add_argument(
        "--training-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/workbench/training",
        help="工作台训练记录目录，与页面默认共享",
    )
    parser.add_argument(
        "--evaluation-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/workbench/evaluations",
        help="工作台业务对照报告目录",
    )
    parser.add_argument(
        "--agent-config-path",
        type=Path,
        default=PROJECT_ROOT / "outputs/workbench/agent-settings.json",
        help="与页面共享的供应商公共配置文件（不保存密钥）",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    config = sub.add_parser("agent-config", help="保存供应商、API 地址和模型名称，不调用网络")
    config.add_argument("--provider", choices=list(PROVIDERS))
    config.add_argument("--base-url")
    config.add_argument("--model")
    check = sub.add_parser("agent-check", help="仅发送合成工具调用探针，不读取业务数据")
    check.add_argument("--base-url")
    check.add_argument("--model")
    create = sub.add_parser("create", help="本地读取文件并保存任务，不调用模型")
    create.add_argument("--input", type=Path, required=True)
    create.add_argument("--goal", required=True)
    create.add_argument("--description", default="")
    create.add_argument("--scope", choices=["sample", "full"], default="sample")
    create.add_argument("--encoding")
    create.add_argument("--delimiter")
    add_source = sub.add_parser("add-source", help="增加或替换具名原始资料，重新分析业务方案")
    add_source.add_argument("session_id")
    add_source.add_argument("--revision", type=int, required=True)
    add_source.add_argument(
        "--alias", required=True, help="资料名称；同名替换，main 为首次上传的资料"
    )
    add_source.add_argument("--input", type=Path, required=True)
    add_source.add_argument(
        "--description", default="", help="这份资料的业务含义及与其他资料的关系"
    )
    add_source.add_argument("--scope", choices=["sample", "full"], default="sample")
    add_source.add_argument("--encoding")
    add_source.add_argument("--delimiter")
    analyze = sub.add_parser("analyze", help="使用配置的模型检查数据、澄清或生成方案")
    analyze.add_argument("session_id")
    analyze.add_argument("--answer", default="")
    analyze.add_argument("--base-url")
    analyze.add_argument("--model")
    analyze.add_argument(
        "--allow-remote-data",
        action="store_true",
        help="允许向所选远程模型服务发送本次业务说明、画像和选取的证据行",
    )
    show = sub.add_parser("show")
    show.add_argument("session_id")
    confirm = sub.add_parser("confirm", help="确认已查看的转换输入与答案符合业务含义")
    confirm.add_argument("session_id")
    confirm.add_argument("--revision", type=int, required=True)
    full_validate = sub.add_parser("full-validate", help="沿已确认方案本地验证全量文件")
    full_validate.add_argument("session_id")
    full_validate.add_argument("--revision", type=int, required=True)
    full_validate.add_argument("--input", type=Path, help="首次已声明全量时可省略并复用原文件")
    full_validate.add_argument("--encoding")
    full_validate.add_argument("--delimiter")
    full_sources = sub.add_parser(
        "full-sources", help="按已确认组合方案验证每份全量原始资料，无需自行拼表"
    )
    full_sources.add_argument("session_id")
    full_sources.add_argument("--revision", type=int, required=True)
    full_sources.add_argument(
        "--source",
        action="append",
        default=[],
        metavar="ALIAS=PATH",
        help="每份资料使用一个参数；不提供时复用已存全量来源，不会升级样例",
    )
    full_confirm = sub.add_parser("full-confirm", help="确认已核对全量报告，继续独立分区准备")
    full_confirm.add_argument("session_id")
    full_confirm.add_argument("--revision", type=int, required=True)
    materialize = sub.add_parser("materialize", help="生成固定版本的独立训练、验证与测试分区")
    materialize.add_argument("session_id")
    materialize.add_argument("--revision", type=int, required=True)
    materialize.add_argument("--name")
    materialize.add_argument("--registry-root", type=Path)
    materialize.add_argument(
        "--validation-fraction",
        type=float,
        default=0.1,
        help="普通分组分区的验证比例；时间方案与固定题集不使用",
    )
    materialize.add_argument(
        "--test-fraction",
        type=float,
        default=0.1,
        help="普通分组分区的测试比例；时间方案与固定题集不使用",
    )
    materialize.add_argument(
        "--seed", type=int, default=42, help="普通分组分区种子；时间方案不随机划分"
    )
    materialize.add_argument("--suite-id", help="继承已冻结开发/测试题集，新增独立资料进入训练集")
    materialize.add_argument("--iteration-id", help="自动使用已确认改进轮次的固定题集")
    materialize.add_argument(
        "--independent-rows-confirmed",
        action="store_true",
        help="仅在未声明分组字段且已确认每行属于独立业务对象时使用",
    )
    preflight = sub.add_parser(
        "preflight", help="用本地 tokenizer 检查实际 token 与答案保留，不启动训练"
    )
    preflight.add_argument("session_id")
    preflight.add_argument("--revision", type=int, required=True)
    preflight.add_argument("--tokenizer", required=True, help="本地 tokenizer 目录或已缓存标识")
    preflight.add_argument("--max-length", type=int, required=True)
    label_verify = sub.add_parser(
        "label-verify",
        help="盲标核验：抽样已标注行并隐藏答案，由业务用户独立作答（训练准备的前置门禁）",
    )
    label_verify.add_argument("session_id")
    label_verify.add_argument("--revision", type=int, required=True)
    label_verify.add_argument("--size", type=int, default=5, help="抽样条数（默认 5，上限 50）")
    verify_submit = sub.add_parser("label-verify-submit", help="提交盲标核验答案并得到一致性判定")
    verify_submit.add_argument("session_id")
    verify_submit.add_argument("--verification-id", required=True)
    verify_submit.add_argument(
        "--answer",
        action="append",
        required=True,
        help="行ID=你的答案，每条抽样行一个 --answer",
    )
    probe = sub.add_parser(
        "learnability-probe",
        help="可学性探针：基座模型对开发集抽样零样本探测，与多数类基线如实对比（证据，不是判决）",
    )
    probe.add_argument("session_id")
    probe.add_argument("--revision", type=int, required=True)
    probe.add_argument("--model-path", required=True, help="已准备好的本地基础模型目录")
    probe.add_argument("--size", type=int, default=8)
    probe.add_argument("--max-new-tokens", type=int, default=32)
    probe.add_argument(
        "--export-csv",
        type=Path,
        help="发现候选时把人工核对清单导出为 CSV 文件（Excel 直开）；没有候选则不写文件",
    )
    probe_show = sub.add_parser(
        "learnability-probe-show",
        help="回读当前数据版本最近一次已保存的探针结果，不重新加载模型",
    )
    probe_show.add_argument("session_id")
    probe_show.add_argument(
        "--export-csv",
        type=Path,
        help="存盘记录含候选时把人工核对清单导出为 CSV 文件（Excel 直开）；没有候选则不写文件",
    )
    train_prepare = sub.add_parser(
        "train-prepare", help="用本地基础模型准备真实训练配置并重做匹配 tokenizer 预检"
    )
    train_prepare.add_argument("session_id")
    train_prepare.add_argument("--revision", type=int, required=True)
    train_prepare.add_argument("--model-path", type=Path, required=True)
    train_prepare.add_argument("--max-length", type=int, default=1024)
    train_prepare.add_argument("--epochs", type=int, default=1)
    train_prepare.add_argument("--batch-size", type=int, default=1)
    train_prepare.add_argument("--learning-rate", type=float, default=2e-4)
    train_prepare.add_argument("--gradient-accumulation", type=int, default=4)
    train_prepare.add_argument("--lora-rank", type=int, default=8)
    train_prepare.add_argument(
        "--load-in-4bit", action="store_true", help="只在兼容的 NVIDIA CUDA 环境启用"
    )
    train_start = sub.add_parser("train-start", help="启动已经准备且未失效的训练方案")
    train_start.add_argument("session_id")
    train_start.add_argument("run_id")
    train_start.add_argument("--revision", type=int, required=True)
    train_start.add_argument("--acknowledge-warnings", action="store_true")
    train_start.add_argument(
        "--recover-technical-failures",
        action="store_true",
        help="允许显存不足后一次保持业务目标和有效 batch 的技术重试",
    )
    for name, help_text in (
        ("train-status", "查看训练状态与产物"),
        ("train-stop", "停止当前训练进程"),
        ("train-logs", "读取最近训练日志"),
    ):
        command = sub.add_parser(name, help=help_text)
        command.add_argument("run_id")
        if name == "train-logs":
            command.add_argument("--tail", type=int, default=100)
    train_list = sub.add_parser("train-list", help="列出数据任务的训练版本")
    train_list.add_argument("session_id")
    eval_compare = sub.add_parser(
        "eval-compare", help="在同一固定开发集上顺序比较基座与已完成微调版本"
    )
    eval_compare.add_argument("--scoring-id", help="使用当前任务已确认的自定义业务评分")
    eval_compare.add_argument("session_id")
    eval_compare.add_argument("run_id")
    eval_compare.add_argument(
        "--parent-run-id", help="使用同一固定题集，同时比较上一轮已成功训练的模型"
    )
    eval_compare.add_argument(
        "--iteration-id", help="自动使用已确认父轮，并将三模型结果绑定到改进轮次"
    )
    eval_compare.add_argument("--revision", type=int, required=True)
    eval_compare.add_argument("--max-new-tokens", type=int, default=256)
    eval_compare.add_argument(
        "--keep-whitespace", action="store_true", help="精确评分保留首尾空白差异"
    )
    eval_show = sub.add_parser("eval-show", help="读取已保存的开发集对照报告")
    eval_show.add_argument("evaluation_id")
    eval_analyze = sub.add_parser(
        "eval-analyze", help="让已配置 Agent 核查实际评测坏例，给出事实、待验证原因和下一步"
    )
    eval_analyze.add_argument("session_id")
    eval_analyze.add_argument("evaluation_id")
    eval_analyze.add_argument("--revision", type=int, required=True)
    eval_analyze.add_argument("--base-url")
    eval_analyze.add_argument("--model")
    eval_analyze.add_argument(
        "--allow-remote-data",
        action="store_true",
        help="允许向已选 Agent 服务发送业务目标、方案、训练配置与检查摘要及实际评测输出与坏例",
    )
    model_list = sub.add_parser("model-list", help="发现本机已有模型文件，不下载或加载模型")
    model_list.add_argument(
        "--root",
        action="append",
        type=Path,
        help="仅检查指定目录，可重复；省略时检查已知本机缓存和 models 目录",
    )
    plan_recommend = sub.add_parser(
        "plan-recommend", help="Agent 基于已确认数据、本机条件和本地候选模型推荐训练方案"
    )
    plan_recommend.add_argument("session_id")
    plan_recommend.add_argument("--revision", type=int, required=True)
    plan_recommend.add_argument(
        "--model-path",
        action="append",
        help="已准备好的本地候选模型目录，可重复；省略时自动发现完整候选",
    )
    plan_recommend.add_argument("--base-url")
    plan_recommend.add_argument("--model")
    plan_recommend.add_argument(
        "--allow-remote-data",
        action="store_true",
        help="允许发送任务、处理方案、数据统计、模型配置和本机硬件摘要，不含数据原文",
    )
    plan_list = sub.add_parser("plan-list", help="列出当前任务已保存的 Agent 训练方案")
    plan_list.add_argument("session_id")
    plan_show = sub.add_parser("plan-show", help="查看已保存的方案、理由与真实预检证据")
    plan_show.add_argument("plan_id")
    plan_prepare = sub.add_parser("plan-prepare", help="确认已审阅的推荐方案并准备训练，不启动训练")
    plan_prepare.add_argument("session_id")
    plan_prepare.add_argument("plan_id")
    plan_prepare.add_argument("--revision", type=int, required=True)
    suite_freeze = sub.add_parser(
        "suite-freeze", help="冻结当前版本的开发和测试题目，供后续轮次固定比较"
    )
    suite_freeze.add_argument("session_id")
    suite_freeze.add_argument("--revision", type=int, required=True)
    suite_show = sub.add_parser("suite-show", help="查看固定题集的来源、题数与摘要")
    suite_show.add_argument("suite_id")
    iteration_propose = sub.add_parser(
        "iteration-propose", help="根据父轮评测提出待确认的改进假设和变更"
    )
    iteration_propose.add_argument("session_id")
    iteration_propose.add_argument("--revision", type=int, required=True)
    iteration_propose.add_argument("--parent-run-id", required=True)
    iteration_propose.add_argument("--evaluation-id", required=True)
    iteration_propose.add_argument("--hypothesis", required=True)
    iteration_propose.add_argument("--expected-outcome", required=True)
    iteration_propose.add_argument("--change", action="append", required=True)
    iteration_propose.add_argument("--data-change", action="store_true")
    iteration_propose.add_argument("--epochs", type=int)
    iteration_propose.add_argument("--learning-rate", type=float)
    iteration_propose.add_argument("--max-length", type=int)
    for name in (
        "iteration-confirm",
        "iteration-prepare",
        "iteration-start",
        "iteration-bind",
        "iteration-revise",
    ):
        command = sub.add_parser(name)
        command.add_argument("session_id")
        command.add_argument("iteration_id")
        command.add_argument("--revision", type=int, required=True)
        if name == "iteration-start":
            command.add_argument("--acknowledge-warnings", action="store_true")
            command.add_argument(
                "--recover-technical-failures",
                action="store_true",
                help="允许显存不足后一次技术重试",
            )
        if name == "iteration-bind":
            command.add_argument("--evaluation-id", required=True)
        if name == "iteration-revise":
            command.add_argument("--base-url")
            command.add_argument("--model")
            command.add_argument(
                "--allow-remote-data",
                action="store_true",
                help="允许发送业务资料、已确认改进方向及父轮实际评测输出与坏例",
            )
    iteration_decide = sub.add_parser(
        "iteration-decide", help="记录采用、继续、停止或证据不足及理由"
    )
    iteration_decide.add_argument("iteration_id")
    iteration_decide.add_argument(
        "--decision", choices=["adopt", "continue", "stop", "insufficient_evidence"], required=True
    )
    iteration_decide.add_argument("--reason", required=True)
    iteration_list = sub.add_parser("iteration-list")
    iteration_list.add_argument("session_id")
    iteration_execute = sub.add_parser(
        "iteration-execute", help="按已确认范围自动物化、训练并完成固定开发集对照"
    )
    iteration_execute.add_argument("session_id")
    iteration_execute.add_argument("iteration_id")
    iteration_execute.add_argument("--revision", type=int, required=True)
    iteration_execute.add_argument("--independent-rows-confirmed", action="store_true")
    iteration_execute.add_argument(
        "--acknowledge-warnings",
        action="store_true",
        help="仅在已查看本次执行暂停的预检提示后继续",
    )
    for name, help_text in (
        ("iteration-execution-status", "查看自动执行状态，不推进执行"),
        ("iteration-execution-stop", "停止自动执行及其关联训练"),
    ):
        execution_command = sub.add_parser(name, help=help_text)
        execution_command.add_argument("iteration_id")
    acceptance_prepare = sub.add_parser(
        "acceptance-prepare", help="冻结单个成功模型的独立业务验收标准与测试题集"
    )
    acceptance_prepare.add_argument(
        "--scoring-id", help="以已确认业务规则的单题通过标准计算最终通过率"
    )
    acceptance_prepare.add_argument("session_id")
    acceptance_prepare.add_argument("run_id")
    acceptance_prepare.add_argument("--revision", type=int, required=True)
    acceptance_prepare.add_argument("--business-standard", required=True)
    acceptance_prepare.add_argument(
        "--minimum-score", type=float, required=True, help="用户确认的最低通过率，0 到 1"
    )
    acceptance_prepare.add_argument(
        "--minimum-cases", type=int, required=True, help="用户确认的最低测试题数"
    )
    acceptance_prepare.add_argument("--max-new-tokens", type=int, default=256)
    acceptance_prepare.add_argument("--keep-whitespace", action="store_true")
    for name in ("acceptance-run", "acceptance-review"):
        command = sub.add_parser(name)
        command.add_argument("session_id")
        command.add_argument("acceptance_id")
        command.add_argument("--revision", type=int, required=True)
        if name == "acceptance-review":
            command.add_argument(
                "--row-index", type=int, required=True, help="报告中的零起始 index"
            )
            command.add_argument("--decision", choices=["accepted", "rejected"], required=True)
            command.add_argument("--reason", required=True)
    acceptance_show = sub.add_parser("acceptance-show", help="查看冻结标准、实际输出与验收结果")
    acceptance_show.add_argument("acceptance_id")
    acceptance_list = sub.add_parser("acceptance-list", help="列出任务的独立业务验收记录")
    acceptance_list.add_argument("session_id")
    scoring_propose = sub.add_parser(
        "scoring-propose", help="Agent 根据业务要求拟定规则，并真实隔离验证开发样例正反例"
    )
    scoring_propose.add_argument("session_id")
    scoring_propose.add_argument("--revision", type=int, required=True)
    scoring_propose.add_argument("--business-standard", required=True)
    scoring_propose.add_argument("--base-url")
    scoring_propose.add_argument("--model")
    scoring_propose.add_argument(
        "--allow-remote-data",
        action="store_true",
        help="允许发送业务要求、处理方案及所需开发样例，不含最终测试题",
    )
    scoring_confirm = sub.add_parser(
        "scoring-confirm", help="确认已查看真实正反例评分及规则符合业务要求"
    )
    scoring_confirm.add_argument("session_id")
    scoring_confirm.add_argument("scoring_id")
    scoring_confirm.add_argument("--revision", type=int, required=True)
    scoring_show = sub.add_parser("scoring-show")
    scoring_show.add_argument("scoring_id")
    scoring_list = sub.add_parser("scoring-list")
    scoring_list.add_argument("session_id")
    args = parser.parse_args()
    try:
        if args.command == "agent-config":
            _save_config(args)
            return 0
        if args.command == "agent-check":
            print(check_connection(_client(args, probe=True)))
            return 0
        if args.command == "model-list":
            from src.workbench.local_models import discover_local_models

            print(json.dumps(discover_local_models(roots=args.root), ensure_ascii=False, indent=2))
            return 0
        service = IntakeService(args.store)
        if args.command.startswith("scoring-"):
            from src.workbench.business_scoring import ScoringService

            scoring = ScoringService(args.scoring_root)
            if args.command == "scoring-show":
                result = scoring.get(args.scoring_id)
            elif args.command == "scoring-list":
                result = scoring.list_specs(session_id=args.session_id)
            else:
                session = service.load(args.session_id)
                if session.revision != args.revision:
                    raise ValueError("任务已更新，请读取最新 revision 后重试。")
                if args.command == "scoring-confirm":
                    result = scoring.confirm(args.scoring_id, session)
                else:
                    from src.agent.scoring import recommend_scoring

                    result = recommend_scoring(
                        session,
                        args.business_standard,
                        _client(args),
                        output_root=args.scoring_root,
                    )
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        if args.command.startswith("acceptance-"):
            from src.workbench.acceptance import AcceptanceService
            from src.workbench.business_evaluation import EvaluationModel, EvaluationProtocol

            acceptance = AcceptanceService(args.acceptance_root, args.evaluation_root)
            if args.command == "acceptance-show":
                result = acceptance.get(args.acceptance_id)
            elif args.command == "acceptance-list":
                result = acceptance.list_acceptances(session_id=args.session_id)
            else:
                session = service.load(args.session_id)
                if session.revision != args.revision:
                    raise ValueError("任务已更新，请读取最新 revision 后重试。")
                if args.command == "acceptance-prepare":
                    from src.workbench.training_runs import TrainingRunService

                    run = TrainingRunService(
                        args.training_root, project_root=PROJECT_ROOT
                    ).get_status(args.run_id)
                    if (
                        run["status"] != "succeeded"
                        or run["session_id"] != session.session_id
                        or session.dataset is None
                        or run["dataset_version"] != session.dataset.version
                    ):
                        raise ValueError("请选择当前任务与数据版本对应的成功模型进行最终验收。")
                    recipe = session.analysis.recipe
                    scorer = (
                        "json_fields_exact"
                        if recipe.output_format == "json"
                        else "classification_exact"
                        if len(recipe.targets) == 1
                        and recipe.targets[0].value_kind == "categorical"
                        else "open_review"
                    )
                    custom_scoring = (
                        _scoring_reference(args.scoring_root, args.scoring_id, session.session_id)
                        if args.scoring_id
                        else None
                    )
                    if custom_scoring:
                        scorer = "custom_rules"
                    result = acceptance.prepare(
                        session,
                        EvaluationModel(
                            "待验收模型", run["model_path"], adapter_path=run["output_dir"]
                        ),
                        EvaluationProtocol(
                            scorer=scorer,
                            **({"custom_scoring": custom_scoring} if custom_scoring else {}),
                            fields=tuple(target.label for target in recipe.targets)
                            if scorer == "json_fields_exact"
                            else (),
                            max_new_tokens=args.max_new_tokens,
                            strip_whitespace=not args.keep_whitespace,
                        ),
                        {
                            "metric": "pass_rate"
                            if scorer == "custom_rules"
                            else "manual_acceptance_rate"
                            if scorer == "open_review"
                            else "exact_match",
                            "minimum_score": args.minimum_score,
                            "minimum_cases": args.minimum_cases,
                            "business_standard": args.business_standard,
                        },
                    )
                else:
                    record = acceptance.get(args.acceptance_id)
                    if record["session_id"] != session.session_id:
                        raise ValueError("验收记录与当前业务任务不匹配。")
                    if args.command == "acceptance-run":
                        result = acceptance.run(args.acceptance_id, session)
                    else:
                        result = acceptance.review(
                            args.acceptance_id,
                            [
                                {
                                    "index": args.row_index,
                                    "decision": args.decision,
                                    "reason": args.reason,
                                }
                            ],
                        )
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        if args.command.startswith("plan-"):
            from src.workbench.training_plans import TrainingPlanService

            plans = TrainingPlanService(args.plan_root, args.training_root)
            if args.command == "plan-list":
                result = plans.list_plans(session_id=args.session_id)
            elif args.command == "plan-show":
                result = plans.get(args.plan_id)
            else:
                session = service.load(args.session_id)
                if session.revision != args.revision:
                    raise ValueError("任务已更新，请读取最新 revision 后重试。")
                if args.command == "plan-prepare":
                    result = plans.prepare(args.plan_id, session)
                else:
                    from src.agent.training import recommend_training

                    candidates = args.model_path
                    if candidates is None:
                        from src.workbench.local_models import discover_local_models

                        candidates = [
                            item["model_path"]
                            for item in discover_local_models()
                            if item["status"] == "available"
                        ]
                    if not candidates:
                        raise ValueError(
                            "未发现完整本地模型。可运行 model-list 查看缺少的文件，或用 --model-path 指定已准备好的目录。"
                        )
                    result = recommend_training(
                        session,
                        candidates,
                        _client(args),
                        output_root=args.plan_root,
                        training_root=args.training_root,
                    )
            print(json.dumps(result, ensure_ascii=False, indent=2))
            if isinstance(result, dict) and result.get("preflight"):
                from src.workbench.report_summary import summarize_preflight

                for line in summarize_preflight(result["preflight"]):
                    print(line, file=sys.stderr)
            return 0
        if args.command in {
            "iteration-execute",
            "iteration-execution-status",
            "iteration-execution-stop",
        }:
            from src.workbench.iteration_execution import IterationExecutionService

            execution = IterationExecutionService(
                Path(args.iteration_root) / "executions",
                args.store,
                args.iteration_root,
                args.training_root,
                args.evaluation_root,
            )
            if args.command == "iteration-execution-status":
                result = execution.get(args.iteration_id)
                if result is None:
                    raise ValueError("这轮尚未提交自动执行。")
            elif args.command == "iteration-execution-stop":
                result = execution.stop(args.iteration_id)
            else:
                session = service.load(args.session_id)
                if session.revision != args.revision:
                    raise ValueError("任务已更新，请读取当前版本后再提交自动执行。")
                result = execution.start(
                    args.iteration_id,
                    session,
                    acknowledge_warnings=args.acknowledge_warnings,
                    independent_rows_confirmed=args.independent_rows_confirmed,
                )
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        if args.command.startswith("iteration-"):
            from src.workbench.iterations import IterationService

            iterations = IterationService(
                args.iteration_root, args.training_root, args.evaluation_root
            )
            if args.command == "iteration-decide":
                result = iterations.decide(args.iteration_id, args.decision, args.reason)
            elif args.command == "iteration-list":
                result = iterations.list_iterations(session_id=args.session_id)
            else:
                session = service.load(args.session_id)
                if session.revision != args.revision:
                    raise ValueError("任务已更新，请读取最新 revision 后重试。")
                if args.command == "iteration-propose":
                    options = {}
                    if args.epochs is not None:
                        options["num_epochs"] = args.epochs
                    if args.learning_rate is not None:
                        options["learning_rate"] = args.learning_rate
                    result = iterations.propose(
                        session,
                        parent_run_id=args.parent_run_id,
                        evaluation_id=args.evaluation_id,
                        hypothesis=args.hypothesis,
                        expected_outcome=args.expected_outcome,
                        changes="\n".join(args.change),
                        data_change=args.data_change,
                        training_options=options or None,
                        max_length=args.max_length,
                    )
                elif args.command == "iteration-revise":
                    from src.agent.revisions import revise_data_for_iteration

                    record = iterations.get(args.iteration_id)
                    if record["session_id"] != session.session_id:
                        raise ValueError("改进轮次与当前任务不匹配。")
                    result = revise_data_for_iteration(
                        service,
                        iterations,
                        args.iteration_id,
                        _client(args),
                        expected_revision=args.revision,
                    ).model_dump(mode="json")
                elif args.command == "iteration-confirm":
                    result = iterations.confirm(args.iteration_id, session)
                elif args.command == "iteration-prepare":
                    result = iterations.prepare(args.iteration_id, session)
                elif args.command == "iteration-start":
                    result = iterations.start(
                        args.iteration_id,
                        session,
                        acknowledge_warnings=args.acknowledge_warnings,
                        **(
                            {"recover_technical_failures": True}
                            if args.recover_technical_failures
                            else {}
                        ),
                    )
                else:
                    result = iterations.bind_evaluation(
                        args.iteration_id, session, args.evaluation_id
                    )
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        if args.command in {"suite-freeze", "suite-show"}:
            from src.workbench.evaluation_suites import EvalSuiteService

            suites = EvalSuiteService(args.suite_root)
            if args.command == "suite-freeze":
                session = service.load(args.session_id)
                if session.revision != args.revision:
                    raise ValueError("任务已更新，请读取最新 revision 后重试。")
                result = suites.freeze(session)
            else:
                result = suites.load(_suite_reference(args.suite_root, args.suite_id))
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        if args.command == "eval-analyze":
            from src.agent.evaluation import assess_evaluation
            from src.workbench.business_evaluation import BusinessEvaluationService

            session = service.load(args.session_id)
            if session.revision != args.revision:
                raise ValueError("任务已更新，请读取最新 revision 后重试。")
            report = BusinessEvaluationService(args.evaluation_root).get_report(args.evaluation_id)
            if (
                report.dataset.get("purpose") == "final_acceptance"
                or report.dataset.get("split") == "test"
            ):
                raise ValueError(
                    "最终独立测试题与坏例不能发送给 Agent 用于优化。请使用开发集报告分析。"
                )
            if session.dataset is None or report.dataset["version"] != session.dataset.version:
                raise ValueError("当前任务数据与评测版本不匹配，不能混用业务解释。")
            result = assess_evaluation(
                report, session, _client(args), output_root=args.evaluation_root
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        if args.command in {"eval-compare", "eval-show"}:
            from src.workbench.business_evaluation import (
                BusinessEvaluationService,
                EvaluationModel,
                EvaluationProtocol,
            )

            evaluation = BusinessEvaluationService(args.evaluation_root)
            if args.command == "eval-show":
                result = evaluation.get_report(args.evaluation_id)
            else:
                from src.workbench.training_runs import TrainingRunService

                session = service.load(args.session_id)
                if session.revision != args.revision:
                    raise ValueError("任务已更新，请读取最新 revision 后重试。")
                training = TrainingRunService(args.training_root, project_root=PROJECT_ROOT)
                run = training.get_status(args.run_id)
                if (
                    run["status"] != "succeeded"
                    or run["session_id"] != session.session_id
                    or session.dataset is None
                    or run["dataset_version"] != session.dataset.version
                ):
                    raise ValueError("仅可比较当前任务、当前数据版本对应的成功训练。")
                recipe = session.analysis.recipe
                scorer = (
                    "json_fields_exact"
                    if recipe.output_format == "json"
                    else "classification_exact"
                    if len(recipe.targets) == 1 and recipe.targets[0].value_kind == "categorical"
                    else "open_review"
                )
                iteration = None
                if args.iteration_id:
                    from src.workbench.iterations import IterationService

                    iterations = IterationService(
                        args.iteration_root, args.training_root, args.evaluation_root
                    )
                    iteration = iterations.get(args.iteration_id)
                    expected_run = iteration.get("new_run_id")
                    if iteration["session_id"] != session.session_id:
                        raise ValueError("改进轮次与当前任务或训练版本不匹配。")
                    if expected_run != args.run_id:
                        original_run = training.get_status(expected_run) if expected_run else {}
                        if (
                            run.get("recovery_parent_run_id") != expected_run
                            or (original_run.get("recovery") or {}).get("child_run_id")
                            != args.run_id
                            or not original_run.get("recover_technical_failures")
                        ):
                            raise ValueError("当前模型不是本轮获准的技术恢复产物。")
                    if args.parent_run_id and args.parent_run_id != iteration["parent_run_id"]:
                        raise ValueError("父轮与已确认改进方案不一致。")
                    args.parent_run_id = iteration["parent_run_id"]
                custom_scoring = (
                    _scoring_reference(args.scoring_root, args.scoring_id, session.session_id)
                    if args.scoring_id
                    else None
                )
                if custom_scoring:
                    scorer = "custom_rules"
                models = [EvaluationModel("基座", run["model_path"])]
                if args.parent_run_id:
                    if not getattr(session.dataset, "evaluation_suite", None):
                        raise ValueError("跨轮次比较需先将本轮数据绑定固定题集。")
                    parent_run = training.get_status(args.parent_run_id)
                    if parent_run["status"] != "succeeded" or parent_run["run_id"] == run["run_id"]:
                        raise ValueError("请选择另一轮已经成功完成的训练作为父轮。")
                    models.append(
                        EvaluationModel(
                            "父轮模型",
                            parent_run["model_path"],
                            adapter_path=parent_run["output_dir"],
                        )
                    )
                models.append(
                    EvaluationModel("本轮微调", run["model_path"], adapter_path=run["output_dir"])
                )
                result = evaluation.compare(
                    session,
                    models,
                    EvaluationProtocol(
                        scorer=scorer,
                        **({"custom_scoring": custom_scoring} if custom_scoring else {}),
                        fields=tuple(target.label for target in recipe.targets)
                        if scorer == "json_fields_exact"
                        else (),
                        max_new_tokens=args.max_new_tokens,
                        strip_whitespace=not args.keep_whitespace,
                    ),
                )
                if iteration:
                    iterations.bind_evaluation(
                        iteration["iteration_id"], session, result.evaluation_id
                    )
            print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
            from src.workbench.report_summary import summarize_comparison

            for line in summarize_comparison(result):
                print(line, file=sys.stderr)
            return 0
        if args.command.startswith("train-"):
            from src.workbench.training_runs import TrainingRunService

            training = TrainingRunService(args.training_root, project_root=PROJECT_ROOT)
            if args.command in {"train-prepare", "train-start"}:
                session = service.load(args.session_id)
                if session.revision != args.revision:
                    raise ValueError("任务已更新，请读取最新 revision 后重试。")
                if args.command == "train-prepare":
                    result = training.prepare(
                        session,
                        str(args.model_path),
                        max_length=args.max_length,
                        training_options={
                            "num_epochs": args.epochs,
                            "batch_size": args.batch_size,
                            "learning_rate": args.learning_rate,
                            "gradient_accumulation_steps": args.gradient_accumulation,
                        },
                        lora_options={"r": args.lora_rank, "lora_alpha": args.lora_rank * 2},
                        model_options={"quantization_bits": 4 if args.load_in_4bit else None},
                    )
                else:
                    result = training.start(
                        args.run_id,
                        session,
                        acknowledge_warnings=args.acknowledge_warnings,
                        **(
                            {"recover_technical_failures": True}
                            if args.recover_technical_failures
                            else {}
                        ),
                    )
            elif args.command == "train-status":
                result = training.get_status(args.run_id)
            elif args.command == "train-stop":
                result = training.stop(args.run_id)
            elif args.command == "train-list":
                result = training.list_runs(session_id=args.session_id)
            else:
                print(training.read_logs(args.run_id, tail=args.tail))
                return 0
            print(json.dumps(result, ensure_ascii=False, indent=2))
            if isinstance(result, dict):
                from src.workbench.report_summary import summarize_training_run

                for line in summarize_training_run(result):
                    print(line, file=sys.stderr)
                if result.get("preflight"):
                    from src.workbench.report_summary import summarize_preflight

                    for line in summarize_preflight(result["preflight"]):
                        print(line, file=sys.stderr)
            return 0
        if args.command == "create":
            session = service.create(
                args.goal,
                args.input.name,
                args.input.read_bytes(),
                data_description=args.description,
                scope=args.scope,
                encoding=args.encoding,
                delimiter=args.delimiter,
            )
        elif args.command == "add-source":
            session = service.add_source(
                args.session_id,
                args.revision,
                args.alias,
                args.input.name,
                args.input.read_bytes(),
                scope=args.scope,
                encoding=args.encoding,
                delimiter=args.delimiter,
            )
            if args.description.strip():
                session = service.answer(
                    session.session_id, f"资料 {args.alias}：{args.description}"
                )
        elif args.command == "analyze":
            client = _client(args)
            if args.answer:
                service.answer(args.session_id, args.answer)
            session = service.analyze(args.session_id, client)
        elif args.command == "confirm":
            session = service.confirm(args.session_id, args.revision)
        elif args.command == "full-validate":
            session = service.validate_full_data(
                args.session_id,
                args.revision,
                args.input.name if args.input else None,
                args.input.read_bytes() if args.input else None,
                encoding=args.encoding,
                delimiter=args.delimiter,
            )
        elif args.command == "full-sources":
            files = {}
            for item in args.source:
                alias, separator, filename = item.partition("=")
                if not separator or not alias.strip() or not filename.strip():
                    raise ValueError(
                        "--source 格式应为资料别名=文件路径，例如 main=./tickets.csv。"
                    )
                if alias in files:
                    raise ValueError(f"资料别名重复：{alias}。每份来源只提供一次。")
                path = Path(filename)
                files[alias] = (path.name, path.read_bytes())
            session = service.validate_full_sources(args.session_id, args.revision, files or None)
        elif args.command == "full-confirm":
            session = service.confirm_full_data(args.session_id, args.revision)
        elif args.command == "materialize":
            current_session = service.load(args.session_id)
            if (
                current_session.analysis
                and current_session.analysis.recipe
                and current_session.analysis.recipe.temporal_split
            ):
                print(
                    "按已确认时间边界生成分区；随机比例与种子不生效，跨窗口及未成熟行明确保留排除。",
                    file=sys.stderr,
                )
            suite_options = (
                {"evaluation_suite": _suite_reference(args.suite_root, args.suite_id)}
                if args.suite_id
                else {}
            )
            if args.iteration_id:
                from src.workbench.iterations import IterationService

                iteration = IterationService(
                    args.iteration_root, args.training_root, args.evaluation_root
                ).get(args.iteration_id)
                if iteration["session_id"] != args.session_id or iteration["status"] != "confirmed":
                    raise ValueError("请先确认当前任务的改进轮次，再准备其固定题集数据。")
                if args.suite_id and args.suite_id != iteration["evaluation_suite"]["suite_id"]:
                    raise ValueError("指定题集与已确认轮次不一致。")
                suite_options = {"evaluation_suite": iteration["evaluation_suite"]}
            session = service.materialize_dataset(
                args.session_id,
                args.revision,
                name=args.name,
                registry_root=args.registry_root,
                validation_fraction=args.validation_fraction,
                test_fraction=args.test_fraction,
                seed=args.seed,
                independent_rows_confirmed=args.independent_rows_confirmed,
                **suite_options,
            )
            if session.dataset.statistics.get("split_method", "").startswith("temporal"):
                statistics = session.dataset.statistics
                print(
                    f"时间分区：纳入 {statistics['included_rows']} 条；保留排除 {statistics['excluded_rows']} 条。原因：{json.dumps(statistics.get('exclusion_counts', {}), ensure_ascii=False)}；原行明细见 dataset.paths.manifest 的 metadata.excluded_rows。",
                    file=sys.stderr,
                )
        elif args.command == "preflight":
            from src.workbench.training_preflight import load_local_tokenizer

            if args.max_length <= 0:
                raise ValueError("max-length 必须为正整数。")
            tokenizer = load_local_tokenizer(args.tokenizer, local_files_only=True)
            session = service.preflight_training(
                args.session_id, args.revision, tokenizer, args.max_length
            )
            print(
                f"训练前检查：{session.training_preflight['status']}（未启动训练）", file=sys.stderr
            )
            from src.workbench.report_summary import summarize_preflight

            for line in summarize_preflight(session.training_preflight):
                print(line, file=sys.stderr)
        elif args.command == "learnability-probe-show":
            from src.workbench.learnability_probe import (
                candidates_to_csv,
                describe_candidates,
                load_latest_probe_record,
            )

            session = service.load(args.session_id)
            if session.dataset is None:
                raise ValueError("当前任务还没有数据集版本，先物化分区再运行探针。")
            record = load_latest_probe_record(
                Path(args.evaluation_root).parent / "probes", session.dataset.version
            )
            if record is None:
                print(
                    f"当前数据版本（{session.dataset.version}）没有已保存的探针记录；"
                    "先运行 learnability-probe。",
                    file=sys.stderr,
                )
                return 2
            record_path, saved = record
            print(json.dumps(saved, ensure_ascii=False, indent=2))
            # 证据溯源:回看结论要能对上是哪个记录文件、什么时候跑出来的。
            saved_at = saved.get("saved_at")
            provenance = f"（记录文件：{record_path.name}" + (
                f"，生成于 {saved_at}" if saved_at else ""
            )
            print(
                f"以上是数据版本 {session.dataset.version} 最近一次已保存的探针结果"
                f"{provenance}）；重新探测请运行 learnability-probe。",
                file=sys.stderr,
            )
            # 回读同样逐行列出候选:重看结论不应重新加载模型,也不该丢掉核对清单。
            for line in describe_candidates(saved.get("label_error_candidates") or []):
                print(line, file=sys.stderr)
            # 回读的存盘候选同样可导出 CSV,与 learnability-probe 同一份清单逻辑,
            # 重看证据时不必为了拿核对清单而重新加载模型。
            if args.export_csv is not None:
                if saved.get("label_error_candidates"):
                    args.export_csv.parent.mkdir(parents=True, exist_ok=True)
                    args.export_csv.write_bytes(candidates_to_csv(saved["label_error_candidates"]))
                    print(f"候选核对清单已导出：{args.export_csv}", file=sys.stderr)
                else:
                    print("没有候选，未生成核对清单 CSV。", file=sys.stderr)
            return 0
        elif args.command == "learnability-probe":
            from src.workbench.learnability_probe import (
                candidates_to_csv,
                describe_candidates,
                probe_learnability,
                save_probe,
            )

            session = service.load(args.session_id)
            if session.revision != args.revision:
                raise ValueError("任务已更新，请读取最新 revision 后重试。")
            result = probe_learnability(
                session,
                args.model_path,
                sample_size=args.size,
                max_new_tokens=args.max_new_tokens,
            )
            path = save_probe(Path(args.evaluation_root).parent / "probes", result)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            print(f"\n探针记录已保存：{path}", file=sys.stderr)
            for line in describe_candidates(result.get("label_error_candidates") or []):
                print(line, file=sys.stderr)
            if args.export_csv is not None:
                if result.get("label_error_candidates"):
                    args.export_csv.parent.mkdir(parents=True, exist_ok=True)
                    args.export_csv.write_bytes(candidates_to_csv(result["label_error_candidates"]))
                    print(f"候选核对清单已导出：{args.export_csv}", file=sys.stderr)
                else:
                    print("没有候选，未生成核对清单 CSV。", file=sys.stderr)
            return 0
        elif args.command == "label-verify":
            pending = service.start_label_verification(
                args.session_id, args.revision, sample_size=args.size
            )
            print("请仅根据输入作答，不要查看数据中的现有答案。", file=sys.stderr)
            # 抽题时如实预告本轮样本量最多能提供的证据强度：
            # 小样本下即使全部一致，真实一致率的置信下界也远低于 100%。
            print(pending["evidence_note"], file=sys.stderr)
            for item in pending["items"]:
                print(f"\n[{item['row_id']}] {item['input']}", file=sys.stderr)
            result = {
                "verification_id": pending["verification_id"],
                "sample_size": pending["sample_size"],
                "row_ids": [item["row_id"] for item in pending["items"]],
                "submit_hint": (
                    "data_intake.py label-verify-submit SESSION --revision R "
                    "--verification-id VERIFICATION_ID --answer 行ID=你的答案（每行一个 --answer）"
                ),
            }
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        elif args.command == "label-verify-submit":
            answers: dict[str, str] = {}
            for item in args.answer:
                row_id, separator, value = item.partition("=")
                if not separator or not row_id.strip() or not value.strip():
                    raise ValueError("--answer 格式应为 行ID=你的答案，例如 r000001=硬件。")
                if row_id in answers:
                    raise ValueError(f"行 {row_id} 重复作答。")
                answers[row_id] = value
            result = service.submit_label_verification(
                args.session_id, args.verification_id, answers
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
            verdict_note = (
                "盲标核验通过：监督信号的业务含义经独立复现。"
                if result["verdict"] == "verified"
                else "存在不一致，训练不会开始；请核对数据标签或业务定义后重新核验。"
            )
            print(
                f"\n判定：{result['verdict']}（{result['matched']}/{result['sample_size']} 一致）",
                file=sys.stderr,
            )
            print(verdict_note, file=sys.stderr)
            return 0
        else:
            session = service.load(args.session_id)
        print(session.model_dump_json(indent=2))
        print(f"\n下一步状态: {next_action(session)}", file=sys.stderr)
        return 0
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
