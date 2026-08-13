from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .presentation import build_natural_language_presentation, humanize_text
from .schemas import ResearchState


def _text(value: Any) -> str:
    if value is None or value == "":
        return "-"
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False, default=str)
    return humanize_text(str(value)).replace("\r", " ").replace("\n", " ").replace("|", "\\|").strip() or "-"


def _number(value: Any, digits: int = 2) -> str:
    try:
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def _percent(value: Any) -> str:
    try:
        return f"{float(value):.2%}"
    except (TypeError, ValueError):
        return "-"


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return "_暂无数据_\n"
    output = ["| " + " | ".join(_text(item) for item in headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    output.extend("| " + " | ".join(_text(cell) for cell in row) + " |" for row in rows)
    return "\n".join(output) + "\n"


def _bullets(items: Any) -> str:
    values = items if isinstance(items, list) else []
    return "\n".join(f"- {_text(item)}" for item in values) + "\n" if values else "_暂无数据_\n"

def _audit_line(name: str, audit: dict[str, Any] | None) -> str:
    item = audit or {}
    status = "通过" if item.get("passed") else "被阻断"
    issues = item.get("issues") or []
    detail = "；".join(_text(issue) for issue in issues) if issues else "无"
    return f"- **{name}：{status}** — {detail}"


def render_research_markdown(state: ResearchState | dict[str, Any]) -> str:
    data = state.model_dump(mode="json") if isinstance(state, ResearchState) else dict(state)
    request = data.get("request") or {}
    research = data.get("research") or {}
    valuation = data.get("valuation") or {}
    judge = data.get("judge") or {}
    expectation = data.get("expectation") or {}
    catalyst = data.get("catalyst") or {}
    bear = data.get("bear") or {}
    mispricing = data.get("mispricing") or {}
    verdict = judge.get("investment_verdict") or {}
    mispricing_verdict = mispricing.get("verdict") or {}
    evidence_status_layer = mispricing.get("evidence_status_layer") or {}
    evidence_acquisition_plan = mispricing.get("evidence_acquisition_plan") or {}
    mispricing_gate = mispricing.get("gate") or {}
    narrative = mispricing.get("narrative_draft") or {}
    edge = mispricing.get("edge") or {}
    classification = mispricing.get("classification") or {}
    presentation = data.get("presentation") or build_natural_language_presentation(data).model_dump(mode="json")
    snapshot = research.get("market_snapshot") or {}
    scenarios = valuation.get("scenarios") or []

    # 用户层只聚合现有校验结果，不创建新 Gate，也不改变底层放行条件。
    consistency = snapshot.get("consistency_report") or {}
    forecast = valuation.get("forecast_reasonableness_audit") or {}
    scenario_trigger = valuation.get("scenario_trigger_audit") or {}
    comparable = valuation.get("comparable_valuation_evidence_audit") or {}
    method_applicability = valuation.get("valuation_method_applicability_audit") or {}
    double_counting = valuation.get("double_counting_audit") or {}
    valuation_validity = valuation.get("valuation_audit") or {}
    gap = mispricing.get("maximum_gap") or {}
    gap_gate = mispricing.get("gate") or {}
    evidence_verdict = (
        judge.get("evidence_based_verdict")
        or mispricing.get("evidence_based_verdict")
        or {}
    )

    def _first_issue(groups: list[Any], fallback: str) -> str:
        for group in groups:
            if isinstance(group, list):
                value = next((str(item) for item in group if str(item).strip()), "")
                if value:
                    return value
        return fallback

    forecast_checks = [
        forecast.get("historical_anchor_passed"),
        forecast.get("margin_bridge_passed"),
        forecast.get("scenario_separation_passed"),
        forecast.get("margin_expansion_evidence_passed"),
        forecast.get("passed"),
        scenario_trigger.get("passed"),
    ]
    forecast_available = bool(forecast or scenario_trigger)
    forecast_passed = bool(forecast.get("passed") and scenario_trigger.get("passed"))
    valuation_checks = [
        comparable.get("passed"), method_applicability.get("passed"),
        double_counting.get("passed"), valuation_validity.get("passed"),
    ]
    valuation_available = bool(valuation_validity)
    valuation_passed = bool(valuation_available and all(value is True for value in valuation_checks))
    gap_available = bool(
        gap_gate.get("benchmark_available")
        and gap.get("agent_value") is not None
        and gap.get("benchmark_value") is not None
    )
    gap_status = (
        "暂不可用" if not gap_available
        else "存在显著差距" if gap.get("significant")
        else "未达到显著阈值"
    )
    deterministic_verdict = judge.get("investment_verdict") or verdict
    verdict_status = (
        evidence_verdict.get("status")
        or deterministic_verdict.get("label")
        or deterministic_verdict.get("status")
        or "暂不能判断"
    )
    five_layer_rows = [
        [
            "1. 数据有效性",
            "暂不可用" if not consistency else "通过" if consistency.get("passed") else "阻断",
            consistency.get("summary") or "尚未执行数据一致性检查。",
            f"数据问题{len(consistency.get('issues') or [])}项；估值数据{'允许使用' if consistency.get('valuation_allowed') else '保持关闭'}。",
            "现有数据未通过时，暂不进入可靠估值。",
        ],
        [
            "2. 预测有效性",
            "暂不可用" if not forecast_available else "通过" if forecast_passed else "阻断",
            "现有预测检查均已通过。" if forecast_passed else _first_issue(
                [forecast.get("issues"), scenario_trigger.get("issues")],
                "至少一项现有预测检查未通过。",
            ),
            f"底层预测检查{sum(value is True for value in forecast_checks)}/{len(forecast_checks)}项通过。",
            "不修改预测来换取通过；未通过时目标价和收益率继续保持关闭。",
        ],
        [
            "3. 预期差",
            gap_status,
            (
                f"{gap.get('period') or ''} {gap.get('metric') or ''}的模型值与比较基准存在差异。"
                if gap_available else "缺少可用比较基准，暂不能判断预期差。"
            ),
            (
                f"比较基准{_number(gap.get('benchmark_value'))}；模型{_number(gap.get('agent_value'))}；"
                f"差值{_number(gap.get('absolute_gap'))}。"
                if gap_available else "不使用猜测补充 Market Consensus 或价格反推基准。"
            ),
            "预期差只描述已有比较结果，不代表自动达到可投资条件。",
        ],
        [
            "4. 估值有效性",
            "暂不可用" if not valuation_available else "通过" if valuation_passed else "阻断",
            "现有估值证据、方法和数学检查均已通过。" if valuation_passed else _first_issue(
                [comparable.get("issues"), method_applicability.get("issues"),
                 double_counting.get("issues"), valuation_validity.get("issues")],
                "至少一项现有估值检查未通过。",
            ),
            (
                f"底层估值检查{sum(value is True for value in valuation_checks)}/{len(valuation_checks)}项通过；"
                f"概率加权目标价{_number(valuation.get('probability_weighted_target_price'))}；"
                f"预期收益率{_percent(valuation.get('expected_price_return'))}。"
            ),
            "不调整倍数或结果；未通过时明确显示估值无法确认。",
        ],
        [
            "5. 投资判断",
            verdict_status,
            evidence_verdict.get("one_sentence_conclusion") or deterministic_verdict.get("summary") or "当前没有可用投资判断。",
            evidence_verdict.get("data") or (
                f"概率加权目标价{_number(valuation.get('probability_weighted_target_price'))}；"
                f"预期收益率{_percent(valuation.get('expected_price_return'))}。"
            ),
            evidence_verdict.get("implication") or deterministic_verdict.get("valuation_block_reason") or "保持现有判断，不自动生成交易建议。",
        ],
    ]
    investor_rows = [
        [item.get("question"), item.get("conclusion"), item.get("why"),
         item.get("data"), item.get("implication")]
        for item in (presentation.get("quick_conclusions") or [])[:8]
    ]
    lines = [
        f"# {_text(request.get('company'))} 公司研究报告",
        "",
        f"> 分析 ID：`{_text(data.get('analysis_id'))}`  ",
        f"> 股票代码：`{_text(request.get('symbol'))}`  ",
        f"> 分析日：{_text(request.get('analysis_date'))}  ",
        f"> 完成时间：{_text(data.get('completed_at'))}  ",
        "> 本报告由结构化研究结果确定性生成，仅供研究，不构成投资建议。",
        "",
        "## 投资者先看",
        "",
        _table(
            ["最关心的问题", "一句话回答", "为什么", "关键数据", "下一步或影响"],
            investor_rows,
        ),
        "## 五层核心判断",
        "",
        "> 本节只汇总现有结果，不改变底层研究、概率、预测、估值和安全停止机制。完整计算与证据默认折叠。",
        "",
        _table(
            ["核心层级", "状态", "结论", "关键数据", "对投资判断的影响"],
            five_layer_rows,
        ),
        "<details>",
        "<summary><strong>查看完整专业计算、证据和检查明细</strong></summary>",
        "",
        "## 证据充分性与最终判断",
        "",
        f"**一句话结论：{_text(evidence_verdict.get('one_sentence_conclusion') or evidence_verdict.get('status'))}**",
        "",
        f"- **为什么：** {_text(evidence_verdict.get('why'))}",
        f"- **数据：** {_text(evidence_verdict.get('data'))}",
        f"- **这意味着什么：** {_text(evidence_verdict.get('implication'))}",
        f"- **主要阻断类型：** {_text(evidence_verdict.get('primary_reason_type'))}",
        "",
        "### 统一证据状态",
        "",
        _table(
            ["假设", "类别", "当前状态", "逻辑状态", "支持证据", "反方证据", "缺失证据", "下一步验证"],
            [[item.get("title"), item.get("category"), item.get("current_status"),
              item.get("logic_status"),
              "；".join(ref.get("evidence_id", "") for ref in item.get("support_evidence") or []),
              "；".join(ref.get("evidence_id", "") for ref in item.get("counter_evidence") or []),
              "；".join(item.get("missing_evidence") or []), item.get("next_verification")]
             for item in evidence_status_layer.get("items") or []],
        ),
        "### 证据获取计划（按解除阻断价值排序）",
        "",
        _table(
            ["优先级", "关键假设", "当前状态", "最值得寻找的证据", "优先来源", "为什么价值高", "预计解除"],
            [[item.get("rank"), item.get("assumption_title"), item.get("current_status"),
              item.get("evidence_to_find"), "；".join(item.get("preferred_sources") or []),
              item.get("why_high_value"), "；".join(item.get("expected_unblock") or [])]
             for item in evidence_acquisition_plan.get("items") or []],
        ),
        "## 专业分析明细（保留全部计算和限制条件）",
        "",
        "### 投资优势与预期差判断",
        "",
        f"**{_text(mispricing_verdict.get('status') or 'NO EDGE')}**",
        "",
        f"- ① **市场现在在赌什么：** {_text(narrative.get('market_bet') or mispricing.get('market_expectation_summary'))}",
        f"- ② **Agent 认为哪里错：** {_text(narrative.get('agent_disagreement'))}",
        f"- ③ **为什么认为错：** {_text(narrative.get('why_market_may_be_wrong'))}",
        f"- ④ **最大预期差：** {_text(mispricing_verdict.get('maximum_gap'))}",
        f"- ⑤ **重新定价事件：** {_text(mispricing_verdict.get('repricing_event'))}",
        f"- ⑥ **验证时间：** {_text(mispricing_verdict.get('verification_deadline'))}",
        f"- ⑦ **判断错误的失败点：** {_text(mispricing_verdict.get('failure_point'))}",
        f"- ⑧ **当前是否达到可投资状态：** {'是' if mispricing_verdict.get('investable') else '否'} — {_text(mispricing_verdict.get('summary'))}",
        "",
        "### 三类预期（严格分离）",
        "",
        _table(
            ["序列", "结果"],
            [
                ["Market Consensus", mispricing.get("market_expectation_summary") or "unavailable"],
                ["Model-Implied Expectation", mispricing.get("model_implied_summary") or "unavailable"],
                ["Agent Forecast · Bear/Base/Bull", mispricing.get("agent_forecast_summary") or "unavailable"],
            ],
        ),
        "### 最大 Gap 与完整验证链", "",
        _table(
            ["Driver", "Market Expectation", "Agent Expectation", "Gap", "Evidence", "Catalyst", "Verification", "Time", "Re-rating", "完整"],
            [[row.get("driver_id"), row.get("market_expectation"), row.get("agent_expectation"),
              row.get("gap_id"), ", ".join(row.get("evidence_refs") or []),
              ", ".join(row.get("catalyst_ids") or []), row.get("verification_metric"),
              row.get("time_window"), row.get("repricing_mechanism"),
              "YES" if row.get("complete") else "NO"]
             for row in mispricing.get("chains") or []],
        ),
        "### Gap Drivers 与证据强度", "",
        _table(
            ["Driver", "类型", "判断", "机制", "状态", "验证指标", "Evidence", "Counter Evidence"],
            [[row.get("driver_id"), row.get("driver_type"), row.get("claim"), row.get("mechanism"),
              "SUPPORTED" if row.get("supported") else "BLOCKED", row.get("verification_metric"),
              ", ".join(row.get("evidence_refs") or []),
              ", ".join(row.get("counter_evidence_refs") or [])]
             for row in mispricing.get("drivers") or []],
        ),
        "### 可证伪条件", "",
        _table(
            ["Condition", "Driver", "当前判断", "关键变量", "观察指标", "判断正确", "判断错误", "最晚验证", "当前状态"],
            [[row.get("condition_id"), row.get("driver_id"), row.get("current_judgment"),
              "；".join(row.get("key_variables") or []), row.get("verification_metric"),
              row.get("confirms_if"), row.get("invalidates_if"), row.get("latest_verification_time"),
              "INVALIDATED" if row.get("currently_invalidated") else row.get("current_status")]
             for row in mispricing.get("falsifiable_conditions") or []],
        ),
        "### Investment Edge（确定性六分项）", "",
        f"- Edge available：{'YES' if edge.get('available') else 'NO'}",
        f"- Total score：{_number(edge.get('total_score')) if edge.get('available') else 'null'}",
        f"- 公式：{_text(edge.get('formula'))}",
        _table(
            ["分项", "Available", "Score", "Formula", "Inputs", "Issues"],
            [[row.get("name"), row.get("available"), row.get("score"), row.get("formula"),
              row.get("inputs"), "；".join(row.get("issues") or [])]
             for row in edge.get("components") or []],
        ),
        "### 五类状态（独立字段）", "",
        _table(
            ["价格便宜", "市场预期过高", "市场预期过低", "基本面正在改善", "估值正在重估"],
            [[classification.get("price_cheap"), classification.get("market_expectation_too_high"),
              classification.get("market_expectation_too_low"), classification.get("fundamentals_improving"),
              classification.get("valuation_rerating")]],
        ),
        f"- Mispricing Gate：{'PASS' if mispricing_gate.get('passed') else 'BLOCKED'}",
        f"- Gate issues：{_text(mispricing_gate.get('issues'))}",
        "",
        "### 原程序化估值结论（保留审计语义）",
        "",
        f"**{_text(verdict.get('label') or verdict.get('status'))}**",
        "",
        f"- 失败层级：{_text(verdict.get('failed_layer'))}",
        f"- 估值阻断原因：{_text(verdict.get('valuation_block_reason'))}",
        f"- 当前价格盈利要求：{_text(verdict.get('current_price_required_earnings'))}",
        f"- Base 判断：{_text(verdict.get('base_forecast_assessment'))}",
        f"- Bull 判断：{_text(verdict.get('bull_forecast_assessment'))}",
        f"- 关键财务变量：{_text(verdict.get('key_financial_driver'))}",
    ]
    lines += [
        "", "### 复杂指标怎么理解", "",
        _table(
            ["指标", "用普通话解释", "计算方法", "原始输入"],
            [[item.get("name"), item.get("explanation"), item.get("formula"), item.get("inputs")]
             for item in presentation.get("metric_explanations") or []],
        ),
        "### 投资逻辑因果链", "",
        *[
            " → ".join(f"{index}. {_text(step)}" for index, step in enumerate(chain, start=1))
            for chain in presentation.get("causal_chains") or []
        ],
        "", "### 核心概率与回报", "",
        _table(
            ["指标", "结果"],
            [
                ["Bull Thesis · P(Bull)", _percent(judge.get("bull_thesis_probability"))],
                ["Downside Event · P(Bear)", _percent(judge.get("downside_risk_probability"))],
                ["负回报概率", _percent(judge.get("probability_of_negative_return"))],
                ["Story Chain（仅诊断）", _percent(judge.get("story_chain_probability"))],
                ["当前价格", _number(valuation.get("current_price"))],
                ["概率加权目标价", _number(valuation.get("probability_weighted_target_price"))],
                ["Expected Return", _percent(valuation.get("expected_price_return"))],
                ["年化 Expected Return", _percent(valuation.get("annualized_expected_return"))],
            ],
        ),
        "## 2. 当前市场与预期", "",
        _table(
            ["价格", "市值(亿元)", "PE TTM", "PB", "总股本(亿股)", "数据日期"],
            [[snapshot.get("price"), snapshot.get("market_cap"), snapshot.get("pe_ttm"),
              snapshot.get("pb"), snapshot.get("shares_outstanding"), snapshot.get("as_of")]],
        ),
        "### 市场正在交易什么", "", _bullets(expectation.get("current_market_story")),
        "### Model-Implied / Consensus / Agent Forecast", "",
        _table(
            ["年份", "Model-Implied Earnings", "Market Consensus", "Agent Forecast", "方向", "说明"],
            [[row.get("year"), row.get("model_implied_profit"),
              row.get("consensus_profit") if row.get("consensus_available") else "Unavailable",
              row.get("agent_forecast_profit"), row.get("direction"), row.get("explanation")]
             for row in valuation.get("forecast_comparison") or []],
        ),
        "## 3. Bear / Base / Bull 情景", "",
        _table(
            ["情景", "Base P", "Final P", "目标年", "收入", "净利润", "方法", "倍数", "目标价", "回报"],
            [[s.get("name"), _percent(s.get("base_probability")), _percent(s.get("probability")),
              s.get("target_year"), s.get("revenue"), s.get("net_profit"), s.get("valuation_method"),
              s.get("valuation_multiple"), s.get("target_price"), _percent(s.get("price_return"))]
             for s in scenarios],
        ),
    ]
    probability = valuation.get("pre_valuation_probability_audit") or {}
    forecast = valuation.get("forecast_reasonableness_audit") or {}
    margin = forecast.get("margin_expansion_evidence") or {}
    comparable = valuation.get("comparable_valuation_evidence_audit") or {}
    lines += [
        "## 4. 审计与 Fail-Closed 状态", "",
        _audit_line("Pre-Valuation Probability Gate", probability),
        _audit_line("Forecast Reasonableness", forecast),
        _audit_line("Margin Expansion Evidence", margin),
        _audit_line("Comparable Valuation Evidence", comparable),
        _audit_line("Scenario Valuation / Expected Return", valuation.get("valuation_audit")),
        "", "### Probability Agent 原始情景", "",
        _table(
            ["情景", "概率", "事件定义", "边界", "Horizon", "Reasoning", "Evidence"],
            [[row.get("name"), _percent(row.get("base_probability")), row.get("event_definition"),
              "；".join(row.get("boundary_conditions") or []), row.get("horizon"), row.get("reasoning"),
              ", ".join(row.get("evidence_refs") or [])]
             for row in probability.get("scenarios") or []],
        ),
        "### Bull 毛利率扩张证据桥", "",
        _table(
            ["年份", "Base GM", "Bull GM", "ΔGM", "Driver Σ", "净利润影响", "主要驱动", "状态", "因果链"],
            [[row.get("year"), _percent(row.get("base_gross_margin")), _percent(row.get("bull_gross_margin")),
              _percent(row.get("gross_margin_delta")), _percent(row.get("driver_impact_sum")),
              row.get("net_profit_impact"), "是" if row.get("primary_profit_driver") else "否",
              "SUPPORTED" if row.get("supported") else "BLOCKED", row.get("causal_chain")]
             for row in margin.get("items") or []],
        ),
        "### 可比估值证据", "",
        _table(
            ["方法", "关系", "标的", "值", "区间", "As Of", "期间", "口径", "适用性", "状态", "Evidence"],
            [[row.get("method"), row.get("relationship"), row.get("subject") or row.get("symbol"),
              row.get("value"), f"{_number(row.get('range_low'))} ~ {_number(row.get('range_high'))}",
              row.get("as_of"), row.get("period"), row.get("accounting_basis"), row.get("applicability"),
              "VALID" if row.get("valid") else "INVALID", row.get("evidence_ref")]
             for row in comparable.get("evidence_items") or []],
        ),
    ]
    lines += [
        "## 5. 催化剂", "",
        _table(
            ["时间", "事件", "方向", "概率", "观察指标", "超预期阈值", "重定价机制", "Evidence"],
            [[row.get("timeframe"), row.get("event"), row.get("direction"), _percent(row.get("probability")),
              row.get("observable_metric"), row.get("surprise_threshold"), row.get("repricing_mechanism"),
              ", ".join(row.get("evidence_refs") or [])]
             for row in catalyst.get("catalysts") or []],
        ),
        "## 6. Red Team / Bear Case", "",
        f"**最强反证：** {_text(bear.get('strongest_disproof'))}", "",
        _table(
            ["风险", "类别", "概率", "影响", "验证方式", "Evidence"],
            [[row.get("risk"), row.get("category"), _percent(row.get("probability")), row.get("impact"),
              row.get("verification"), ", ".join(row.get("evidence_refs") or [])]
             for row in bear.get("risks") or []],
        ),
        "## 7. 下一步观察", "",
        "### 未来30天", _bullets((judge.get("tracking_plan") or {}).get("next_30_days")),
        "### 下一季度", _bullets((judge.get("tracking_plan") or {}).get("next_quarter")),
        "### 未来半年", _bullets((judge.get("tracking_plan") or {}).get("next_half_year")),
        "### 解除阻断所需证据", _bullets(verdict.get("evidence_needed")),
        "## 8. 统一术语表", "",
        _table(
            ["专业写法", "本报告统一使用的中文说法"],
            [[key, value] for key, value in presentation.get("terminology", {}).items()],
        ),
        "## 9. 证据附录", "",
        _table(
            ["证据编号", "事实", "来源", "等级", "日期", "当前有效", "网址"],
            [[row.get("id"), row.get("fact"), row.get("source"), row.get("source_grade"),
              row.get("publication_date") or row.get("information_date") or row.get("available_at"),
              "是" if row.get("supports_current_claim") else "否", row.get("source_url")]
             for row in research.get("evidence") or []],
        ),
        "</details>", "",
        "---", "",
        _text(judge.get("disclaimer") or "仅供研究，不构成投资建议。"), "",
    ]
    rendered = "\n".join(lines)
    summary, separator, details = rendered.partition("<details>")
    return summary + separator + humanize_text(details)


def _safe_slug(value: Any, fallback: str) -> str:
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", str(value or "")).strip(" ._")
    text = re.sub(r"\s+", "_", text)[:48]
    return text or fallback


def export_research_markdown(
    state: ResearchState,
    root: str | Path | None = None,
) -> Path:
    if state.status != "completed":
        raise ValueError("仅为已完成的公司研究生成Markdown报告")
    project_root = Path(__file__).resolve().parents[1]
    reports_root = Path(root) if root else project_root / "data" / "research_reports"
    day = state.request.analysis_date
    folder = (reports_root / f"{day:%Y}" / f"{day:%m}").resolve()
    root_resolved = reports_root.resolve()
    if root_resolved != folder and root_resolved not in folder.parents:
        raise ValueError("报告路径越界")
    folder.mkdir(parents=True, exist_ok=True)
    company = _safe_slug(state.request.company, "company")
    symbol = _safe_slug(state.request.symbol, "no-symbol")
    analysis = _safe_slug(state.analysis_id[:16], "analysis")
    target = folder / f"{day.isoformat()}_{company}_{symbol}_{analysis}.md"
    temporary = target.with_suffix(".md.tmp")
    temporary.write_text(render_research_markdown(state), encoding="utf-8")
    temporary.replace(target)
    return target
