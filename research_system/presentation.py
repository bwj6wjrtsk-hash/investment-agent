from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel

from .schemas import (
    NaturalLanguagePresentation,
    ReadableBlockExplanation,
    ReadableConclusion,
    ReadableMetricExplanation,
)


TERMINOLOGY: dict[str, str] = {
    "Pre-Valuation Probability Gate": "估值前情景概率检查",
    "Probability of Negative Return": "出现负收益的概率",
    "Annualized Expected Return": "年化预期收益率",
    "Required Return": "要求达到的回报率",
    "Market Consensus": "市场一致预期",
    "Model-Implied Earnings": "当前股价隐含的盈利要求",
    "Model-Implied Expectation": "当前股价隐含的经营要求",
    "Model-Implied的": "当前股价隐含的",
    "Model-Implied": "当前股价隐含的要求",
    "Agent Forecast": "模型预测",
    "Agent Expectation": "模型预期",
    "Market Expectation": "市场预期",
    "Expected Return": "预期收益率",
    "Investment Edge": "投资优势/预期差",
    "Downside Event": "悲观事件",
    "Downside Risk": "下行风险",
    "Catalyst": "催化因素（可能推动股价变化的事件）",
    "Counter Evidence": "反对证据",
    "Evidence Gate": "证据检查关卡",
    "Evidence": "证据",
    "Forecast Reasonableness": "预测是否合理",
    "Scenario Separation": "不同情景是否由真实经营变量区分",
    "Comparable Valuation": "可比公司估值",
    "Margin Expansion": "毛利率提升",
    "Historical Anchor": "历史数据参照",
    "Margin Bridge": "利润率传导检查",
    "Valuation Re-rating": "估值提升",
    "Target Price": "目标价",
    "Probability": "概率",
    "Revenue Growth": "营业收入增长率",
    "Revenue": "营业收入",
    "Gross Margin": "毛利率",
    "Gross Profit": "毛利润",
    "Net Profit": "净利润",
    "Net Margin": "净利率",
    "D&A": "折旧与摊销",
    "FCF": "自由现金流",
    "EV/Sales": "企业价值/营业收入",
    "EV/EBITDA": "企业价值/息税折旧摊销前利润",
    "EBITDA": "息税折旧摊销前利润",
    "EBIT Margin": "营业利润率",
    "EBIT": "营业利润",
    "PE TTM": "滚动市盈率",
    "PE": "市盈率",
    "PB": "市净率",
    "Gap Drivers": "差距驱动因素",
    "Gap Driver": "差距驱动因素",
    "Gap": "差距",
    "Gate": "检查关卡",
    "Fail-closed": "条件不满足就不输出结论",
    "Price In": "被市场计价",
    "Agent": "模型",
    "Driver": "驱动因素",
    "Condition": "条件",
    "Reasoning": "判断依据",
    "Horizon": "预测期限",
    "Audit": "检查",
    "Score": "得分",
    "Formula": "计算方法",
    "Inputs": "输入数据",
    "Issues": "问题",
    "Available": "是否有数据",
    "Edge available": "投资优势是否可计算",
    "Total score": "总分",
    "Story Chain": "故事因果链",
    "Red Team": "反方检验",
    "Bear Case": "悲观情景检验",
    "Scenario Valuation": "情景估值",
    "Probability Agent": "概率模型",
    "Bull Thesis": "乐观情景成立概率",
    "Equity Value": "股东价值",
    "Mispricing": "预期差",
    "Valuation": "估值",
    "Expectation": "预期",
    "Verification": "验证指标",
    "Re-rating": "估值变化",
    "Earnings": "盈利",
    "Benchmark": "比较基准",
    "Time": "时间",
    "As Of": "数据日期",
    "Case": "检验",
    "Ref": "证据编号",
    "business_mix": "业务结构",
    "product_mix": "产品结构",
    "valuation_multiple": "估值倍数",
    "other": "其他",
    "competition": "竞争",
    "strategy": "战略",
    "governance": "公司治理",
    "valuation": "估值",
    "demand": "需求",
    "cash_flow": "现金流",
    "receivables": "应收账款",
    "inventory": "存货",
    "technology": "技术",
    "critical": "严重",
    "high": "高",
    "medium": "中",
    "low": "低",
    "one_year": "未来一年",
    "half_year": "未来半年",
    "quarter": "下一季度",
    "30d": "未来30天",
    "two_sided": "双向",
    "YES": "是",
    "NO": "否",
    "true": "是",
    "false": "否",
    "supported_drivers": "有证据支持的驱动因素数量",
    "all_drivers": "全部驱动因素数量",
    "historical_anchor": "历史数据参照",
    "margin_bridge": "利润率传导检查",
    "scenario_separation": "不同情景是否由真实经营变量区分",
    "margin_expansion_evidence": "毛利率提升证据",
    "valid_catalysts": "有效催化因素",
    "annualized_expected_return": "年化预期收益率",
    "required_return": "要求达到的回报率",
    "p_bear": "悲观情景概率",
    "probability_of_negative_return": "出现负收益的概率",
    "maximum_absolute_gap_ratio": "最大绝对差距比例",
    "probability_generation": "概率生成阶段",
    "model_implied": "当前股价隐含的要求",
    "agent_forecast": "模型预测",
    "market_consensus": "市场一致预期",
    "De-rating": "估值下调",
    "de-rating": "估值下调",
    "ratio": "比例",
    "vs": "对比",
    "available": "有可靠数据",
    "market_vs_agent_earnings_gap": "市场与模型盈利差距",
    "evidence_strength": "证据强度",
    "forecast_confidence": "预测可信度",
    "catalyst_proximity": "催化因素临近程度",
    "valuation_gap": "估值差距",
    "downside_risk": "下行风险承受度",
    "gross_margin": "毛利率",
    "ebit_margin": "营业利润率",
    "net_profit": "净利润",
    "revenue": "营业收入",
}

STATUS_LABELS: dict[str, str] = {
    "PASS": "通过",
    "BLOCKED": "被阻断",
    "FAILED": "未通过",
    "UNAVAILABLE": "暂无可靠数据",
    "SUPPORTED": "有证据支持",
    "UNSUPPORTED": "缺少证据",
    "NO EDGE": "暂未发现明确投资优势",
    "WATCH": "值得观察",
    "EMERGING EDGE": "开始出现投资机会",
    "ACTIONABLE EDGE": "已经形成比较明确的投资机会",
    "INVALIDATED": "原来的判断已经被证据推翻",
    "VALID": "有效",
    "INVALID": "无效",
    "Available": "有可靠数据",
    "Unavailable": "暂无可靠数据",
    "positive": "偏高",
    "negative": "偏低",
    "neutral": "接近",
    "supporting": "正在获得数据支持",
    "untested": "尚未验证",
    "refuted": "已被反证",
    "True": "是",
    "False": "否",
    "bear": "悲观情景",
    "base": "基准情景",
    "bull": "乐观情景",
}


def _dict(value: Any) -> dict[str, Any]:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return dict(value or {})


def humanize_text(value: Any) -> str:
    text = str(value or "").strip()
    protected_urls: list[str] = []

    def protect_url(match: re.Match[str]) -> str:
        protected_urls.append(match.group(0))
        return f"\uf000{len(protected_urls) - 1}\uf001"

    text = re.sub(r"https?://[^\s<>]+", protect_url, text)
    replacements = {**TERMINOLOGY, **STATUS_LABELS}
    for source in sorted(replacements, key=len, reverse=True):
        pattern = rf"(?<![A-Za-z0-9_]){re.escape(source)}(?![A-Za-z0-9_])"
        text = re.sub(pattern, lambda _match, target=replacements[source]: target, text)
    text = re.sub(r"(\d+(?:\.\d+)?)\s*pct\b", r"\1个百分点", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<![A-Za-z0-9_])Base(?![A-Za-z0-9_])", "基准情景", text)
    text = re.sub(r"(?<![A-Za-z0-9_])Bear(?![A-Za-z0-9_])", "悲观情景", text)
    text = re.sub(r"(?<![A-Za-z0-9_])Bull(?![A-Za-z0-9_])", "乐观情景", text)
    text = re.sub(r"(?<![A-Za-z0-9_])unavailable(?![A-Za-z0-9_])", "暂无可靠数据", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<![A-Za-z0-9_])null(?![A-Za-z0-9_])", "暂时无法计算", text, flags=re.IGNORECASE)
    for phrase in ("值得注意的是，", "需要强调的是，", "在此基础上，", "综合来看，"):
        text = text.replace(phrase, "")
    for index, url in enumerate(protected_urls):
        text = text.replace(f"\uf000{index}\uf001", url)
    return text or "暂无可靠数据"


def _number(value: Any, digits: int = 2) -> str:
    try:
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return "暂无可靠数据"


def _percent(value: Any, digits: int = 1) -> str:
    try:
        return f"{float(value):.{digits}%}"
    except (TypeError, ValueError):
        return "暂无可靠数据"


def _scenario_assessment(valuation: dict[str, Any], scenario_name: str) -> ReadableConclusion:
    label = STATUS_LABELS[scenario_name]
    audit = valuation.get("required_earnings_gap") or {}
    entry = next(
        (item for item in audit.get("entries") or [] if item.get("scenario") == scenario_name),
        None,
    )
    question = f"{label}能不能支撑当前股价？"
    if not entry or entry.get("model_implied_earnings") is None or entry.get("agent_forecast_earnings") is None:
        return ReadableConclusion(
            question=question,
            conclusion=f"目前无法判断{label}能否支撑当前股价。",
            why="当前股价要求的盈利或模型预测没有通过前置检查，不能进行同年份比较。",
            data="至少有一项关键数字暂无可靠数据。",
            implication="继续给出‘能支撑’或‘不能支撑’会造成虚假的确定性，因此系统不下结论。",
        )
    required = float(entry["model_implied_earnings"])
    forecast = float(entry["agent_forecast_earnings"])
    gap = required - forecast
    year = str(entry.get("year") or audit.get("year") or "目标年份")
    if gap > 0:
        conclusion = f"{label}不能支撑当前股价。"
        why = "按照现在的股价反推，公司需要赚到的利润高于模型预测。"
        implication = f"公司实际利润还需要比{label}预测高约{_number(gap)}亿元，当前股价才更容易得到支撑。"
    else:
        conclusion = f"{label}可以覆盖当前股价对盈利的要求。"
        why = "模型预测利润不低于当前股价反推的利润要求。"
        implication = f"{label}比当前股价要求多预测约{_number(abs(gap))}亿元利润，但仍需确认估值和风险检查是否通过。"
    return ReadableConclusion(
        question=question,
        conclusion=conclusion,
        why=why,
        data=(
            f"{year}年，当前股价要求公司净利润约{_number(required)}亿元；"
            f"{label}预测约{_number(forecast)}亿元；两者相差约{_number(abs(gap))}亿元。"
        ),
        implication=implication,
    )


def _return_block(valuation: dict[str, Any]) -> ReadableBlockExplanation:
    if valuation.get("expected_price_return") is not None and (valuation.get("valuation_audit") or {}).get("passed"):
        return ReadableBlockExplanation(
            blocked=False,
            headline=f"目前可以计算预期收益率，结果为{_percent(valuation.get('expected_price_return'))}。",
            reasons=["三个情景都有可比较的目标价，概率和估值检查均已通过。"],
            why_stop="",
        )
    reasons: list[str] = []
    if not (valuation.get("pre_valuation_probability_audit") or {}).get("passed"):
        reasons.append("悲观、基准、乐观三个情景的概率依据不完整，无法确认它们互不重叠且合计为100%。")
    forecast = valuation.get("forecast_reasonableness_audit") or {}
    if not forecast.get("passed"):
        reasons.append("模型对收入、毛利率或利润的预测没有通过‘是否符合历史和经营证据’的检查。")
    if not (valuation.get("comparable_valuation_evidence_audit") or {}).get("passed"):
        reasons.append("至少一个情景缺少可靠的可比公司或公司历史估值证据，无法得到可信目标价。")
    scenarios = valuation.get("scenarios") or []
    missing = [STATUS_LABELS.get(item.get("name"), str(item.get("name"))) for item in scenarios if item.get("target_price") is None]
    if missing:
        reasons.append(f"{'、'.join(missing)}没有可靠目标价，三个情景无法形成完整比较。")
    if not scenarios:
        reasons.append("前置检查已经停止了财务预测和估值，因此没有生成三个可比较的情景。")
    if not reasons:
        reasons.extend(humanize_text(item) for item in (valuation.get("valuation_audit") or {}).get("issues", [])[:3])
    return ReadableBlockExplanation(
        blocked=True,
        headline="目前无法计算预期收益率。",
        reasons=list(dict.fromkeys(reasons)),
        why_stop="在这些条件没有满足时继续计算概率加权目标价，会制造看似精确但不可靠的数字，所以系统主动停止计算。",
    )


def _metric_explanations(valuation: dict[str, Any]) -> list[ReadableMetricExplanation]:
    output: list[ReadableMetricExplanation] = []
    earnings = valuation.get("required_earnings_gap") or {}
    for item in earnings.get("entries") or []:
        required = item.get("model_implied_earnings")
        forecast = item.get("agent_forecast_earnings")
        gap = item.get("required_earnings_gap")
        if None in {required, forecast, gap}:
            continue
        label = STATUS_LABELS.get(item.get("scenario"), str(item.get("scenario")))
        if gap > 0:
            explanation = (
                f"当前股价要求的盈利减去{label}预测盈利，结果为{_number(gap)}亿元。"
                f"也就是说，公司实际利润还需要比{label}预测高约{_number(gap)}亿元，当前股价才更容易得到支撑。"
            )
        else:
            explanation = (
                f"{label}预测盈利比当前股价要求高约{_number(abs(gap))}亿元，"
                "说明盈利这一项可以覆盖股价要求，但不代表其他估值和风险条件已经通过。"
            )
        output.append(ReadableMetricExplanation(
            name=f"{label}盈利差距",
            explanation=explanation,
            formula="当前股价要求的净利润－模型预测净利润",
            inputs={"当前股价要求的净利润": required, "模型预测净利润": forecast, "差距": gap},
        ))

    valuation_gap = valuation.get("valuation_gap") or {}
    for item in valuation_gap.get("entries") or []:
        if item.get("scenario") != "base":
            continue
        scenario_multiple = item.get("scenario_multiple")
        current_multiple = item.get("current_comparable_multiple")
        change = item.get("multiple_change_percentage")
        if None in {scenario_multiple, current_multiple, change}:
            break
        direction = "下降" if change < 0 else "提高"
        output.append(ReadableMetricExplanation(
            name="基准情景估值差距",
            explanation=(
                f"基准情景使用约{_number(scenario_multiple)}倍估值，而当前市场同口径约为{_number(current_multiple)}倍。"
                f"这意味着基准情景假设市场愿意给公司的估值{direction}约{_percent(abs(change))}。"
            ),
            formula="基准情景估值倍数－当前同口径估值倍数",
            inputs={"基准情景倍数": scenario_multiple, "当前倍数": current_multiple, "变化比例": change},
        ))
        break
    return output


def _causal_chains(valuation: dict[str, Any], mispricing: dict[str, Any]) -> list[list[str]]:
    chains: list[list[str]] = []
    margin = ((valuation.get("forecast_reasonableness_audit") or {}).get("margin_expansion_evidence") or {})
    item = next((row for row in margin.get("items") or [] if row.get("material")), None)
    if item:
        chains.append([
            f"毛利率提高约{_percent(item.get('gross_margin_delta'))}",
            "每卖出100元产品，公司留下的毛利润更多",
            f"按模型测算，净利润因此变化约{_number(item.get('net_profit_impact'))}亿元",
            "如果利润提高得到财报验证，市场可能愿意给更高估值",
            "只有前述步骤都发生，股价才可能进一步上涨",
        ])
    for chain in (mispricing.get("chains") or [])[:3]:
        chains.append([
            humanize_text(chain.get("market_expectation")),
            humanize_text(chain.get("agent_expectation")),
            f"两者形成{humanize_text(chain.get('gap_id'))}",
            f"需要证据支持的原因：{humanize_text(chain.get('driver_id'))}",
            f"观察事件：{humanize_text('、'.join(chain.get('catalyst_ids') or []))}",
            f"验证数据：{humanize_text(chain.get('verification_metric'))}；时间：{humanize_text(chain.get('time_window'))}",
            f"如果得到验证：{humanize_text(chain.get('repricing_mechanism'))}",
        ])
    return chains


def build_natural_language_presentation(data: Any) -> NaturalLanguagePresentation:
    root = _dict(data)
    if isinstance(root.get("stages"), dict):
        root = {**root, **root["stages"]}
    research = root.get("research") or {}
    expectation = root.get("expectation") or {}
    valuation = root.get("valuation") or {}
    mispricing = root.get("mispricing") or {}
    judge = root.get("judge") or {}
    bear = root.get("bear") or {}
    catalyst = root.get("catalyst") or {}
    snapshot = research.get("market_snapshot") or {}
    implied_path = valuation.get("reverse_valuation") or expectation.get("price_implied_path") or []
    implied = next((item for item in implied_path if item.get("implied_net_profit") is not None), None)
    consensus_available = bool((mispricing.get("gate") or {}).get("consensus_gate_passed"))

    if consensus_available:
        market = ReadableConclusion(
            question="这家公司现在市场在期待什么？",
            conclusion="已有可靠的机构一致预期，可以用来判断市场普遍期待。",
            why="一致预期有当前有效的外部来源，并通过证据检查。",
            data=humanize_text(mispricing.get("market_expectation_summary")),
            implication="后续可以直接比较市场一致预期与模型预测的差距。",
        )
    elif implied:
        market = ReadableConclusion(
            question="这家公司现在市场在期待什么？",
            conclusion="目前没有可靠的机构一致预期，只能根据当前股价反推市场对盈利的要求。",
            why="现有资料没有满足机构来源数量、日期和证据有效性的要求。",
            data=(
                f"当前股价约{_number(snapshot.get('price') or valuation.get('current_price'))}元。"
                f"在模型披露的回报率和估值假设下，股价要求公司{implied.get('year')}年"
                f"净利润约{_number(implied.get('implied_net_profit'))}亿元"
                + (f"、营业收入约{_number(implied.get('implied_revenue'))}亿元。" if implied.get("implied_revenue") is not None else "。")
            ),
            implication="这些数字只是股价在给定假设下提出的要求，不代表机构真的形成了这个一致预期，也不代表已经被市场完全计价。",
        )
    else:
        market = ReadableConclusion(
            question="这家公司现在市场在期待什么？",
            conclusion="目前无法可靠判断市场在期待什么。",
            why="既没有通过证据检查的机构一致预期，也缺少可用的股价反推结果。",
            data="市场预期相关数字暂无可靠数据。",
            implication="不能用猜测补充市场预期，因此暂时不比较市场与模型的分歧。",
        )

    maximum_gap = mispricing.get("maximum_gap") or {}
    narrative = mispricing.get("narrative_draft") or {}
    if maximum_gap.get("agent_value") is not None and maximum_gap.get("benchmark_value") is not None:
        difference = maximum_gap.get("absolute_gap")
        disagreement = ReadableConclusion(
            question="模型认为市场可能错在哪里？",
            conclusion=(
                "模型认为市场要求偏低。" if difference and difference > 0
                else "模型认为市场要求偏高。" if difference and difference < 0
                else "模型与市场要求接近。"
            ),
            why=humanize_text(narrative.get("why_market_may_be_wrong")),
            data=(
                f"{maximum_gap.get('period')}年{humanize_text(maximum_gap.get('metric'))}："
                f"市场基准约{_number(maximum_gap.get('benchmark_value'))}，模型约{_number(maximum_gap.get('agent_value'))}，"
                f"相差约{_number(abs(difference or 0))}"
                + (f"，相当于{_percent(abs(maximum_gap.get('gap_percentage')))}。" if maximum_gap.get("gap_percentage") is not None else "。")
            ),
            implication="只有差距的原因有证据、未来有数据可以验证，这个分歧才可能成为投资机会。",
        )
    else:
        disagreement = ReadableConclusion(
            question="模型认为市场可能错在哪里？",
            conclusion="目前还不能把模型与市场的分歧量化。",
            why=humanize_text(narrative.get("agent_disagreement") or "模型预测或可靠市场基准缺失"),
            data="没有通过检查的同年份市场基准与模型预测，最大差距暂时无法计算。",
            implication="可以保留观察假设，但不能把它称为已经成立的投资机会。",
        )

    forecast = valuation.get("forecast_reasonableness_audit") or {}
    supported_driver = next((item for item in mispricing.get("drivers") or [] if item.get("supported")), None)
    key_variable = humanize_text(
        forecast.get("key_driver")
        or (supported_driver or {}).get("claim")
        or "关键经营变量尚未识别"
    )
    base = _scenario_assessment(valuation, "base")
    bull = _scenario_assessment(valuation, "bull")

    legacy_status = ((mispricing.get("verdict") or {}).get("status") or "NO EDGE")
    evidence_verdict = mispricing.get("evidence_based_verdict") or {}
    verdict_label = humanize_text(
        evidence_verdict.get("status") or STATUS_LABELS.get(legacy_status, legacy_status)
    )
    blockers = (
        (mispricing.get("verdict") or {}).get("blockers")
        or (mispricing.get("gate") or {}).get("issues")
        or []
    )
    action = ReadableConclusion(
        question="现在是否已经达到值得投资的条件？",
        conclusion=evidence_verdict.get("one_sentence_conclusion") or f"当前研究状态：{verdict_label}。",
        why=evidence_verdict.get("why") or humanize_text(
            (mispricing.get("verdict") or {}).get("summary") or "关键条件尚未全部满足"
        ),
        data=evidence_verdict.get("data") or (
            "；".join(humanize_text(item) for item in blockers[:3])
            if blockers else "关键条件已有结果，具体见预测、估值和风险收益数据。"
        ),
        implication=evidence_verdict.get("implication") or (
            "目前没有足够证据形成明确投资优势。"
            if legacy_status in {"NO EDGE", "WATCH", "INVALIDATED"}
            else "条件有所改善，但仍不等于自动买入建议。"
        ),
    )
    headline = ReadableConclusion(
        question="一句话结论",
        conclusion=action.conclusion,
        why=action.why,
        data=action.data,
        implication=action.implication,
    )

    gap_focus = ReadableConclusion(
        question="最大的预期差是什么？",
        conclusion=(
            f"最大预期差集中在{maximum_gap.get('period')}年"
            f"{humanize_text(maximum_gap.get('metric'))}。"
            if maximum_gap.get("metric") else disagreement.conclusion
        ),
        why=humanize_text(
            (supported_driver or {}).get("mechanism")
            or narrative.get("why_market_may_be_wrong")
            or disagreement.why
        ),
        data=f"{disagreement.data} 关键经营变量：{key_variable}。",
        implication=humanize_text(
            (supported_driver or {}).get("rerating_mechanism")
            or disagreement.implication
        ),
    )

    falsifier = next(
        (item for item in mispricing.get("falsifiable_conditions") or [] if item.get("complete")),
        None,
    )
    next_catalyst = next(
        (item for item in catalyst.get("catalysts") or [] if (item.get("evidence_gate") or {}).get("passed")),
        None,
    )
    acquisition_item = next(
        iter((mispricing.get("evidence_acquisition_plan") or {}).get("items") or []),
        None,
    )
    if next_catalyst:
        verification_event = ReadableConclusion(
            question="接下来最重要的验证事件是什么？",
            conclusion=f"重点等待：{humanize_text(next_catalyst.get('event'))}",
            why=humanize_text(
                next_catalyst.get("observable_metric")
                or "该事件最能直接验证核心经营变量"
            ),
            data=(
                f"时间：{humanize_text(next_catalyst.get('expected_date') or next_catalyst.get('timeframe'))}；"
                f"正向结果：{humanize_text(next_catalyst.get('favorable_result'))}；"
                f"负向结果：{humanize_text(next_catalyst.get('adverse_result'))}。"
            ),
            implication=(
                "如果正向结果出现，"
                + humanize_text(next_catalyst.get("repricing_mechanism") or "市场可能重新评估盈利与估值")
                + "；如果未出现，则不能继续沿用乐观解释。"
            ),
        )
    elif acquisition_item:
        verification_event = ReadableConclusion(
            question="接下来最重要的验证事件是什么？",
            conclusion=f"优先补充：{humanize_text(acquisition_item.get('evidence_to_find'))}",
            why=humanize_text(acquisition_item.get("why_high_value")),
            data=(
                f"优先来源：{'、'.join(acquisition_item.get('preferred_sources') or [])}；"
                f"证据要求：{acquisition_item.get('required_quality')}。"
            ),
            implication="找到资料不代表自动通过，仍需按现有来源、日期和独立性要求验证。",
        )
    else:
        verification_event = ReadableConclusion(
            question="接下来最重要的验证事件是什么？",
            conclusion="目前尚未形成可靠的下一项验证事件。",
            why="现有催化因素或证据获取计划不足。",
            data="下一期营业收入、毛利率、净利润和经营现金流仍是基础观察项。",
            implication="在验证事件明确前，不把预期差升级为投资优势。",
        )

    top_risk = next(
        (item for item in bear.get("risks") or [] if item.get("impact") in {"critical", "high"}),
        None,
    )
    risk_text = humanize_text(
        bear.get("strongest_disproof")
        or (top_risk or {}).get("risk")
        or "尚未形成可靠的最大风险判断"
    )
    risk_data = humanize_text(
        (top_risk or {}).get("verification") or "需要等待后续经营和财务数据验证"
    )
    falsification = ReadableConclusion(
        question="什么情况出现后，投资逻辑会被证伪？",
        conclusion=humanize_text(
            (falsifier or {}).get("invalidates_if")
            or bear.get("strongest_disproof")
            or "尚未定义满足要求的事实证伪条件"
        ),
        why=humanize_text(
            (falsifier or {}).get("current_judgment")
            or risk_text
        ),
        data=(
            f"观察指标：{humanize_text((falsifier or {}).get('verification_metric') or risk_data)}；"
            f"最晚验证：{humanize_text((falsifier or {}).get('latest_verification_time') or '尚未明确')}。"
        ),
        implication="只有上述事实条件真正出现，才写成投资逻辑被证伪；证据不足本身不等于证伪。",
    )
    blocker = ReadableConclusion(
        question="为什么现在还不能确认 / 为什么值得关注？",
        conclusion=action.conclusion,
        why=action.why,
        data=(
            "；".join(humanize_text(item) for item in blockers[:3])
            if blockers else action.data
        ),
        implication=action.implication,
    )

    return_block = _return_block(valuation)
    risks = [risk_text]
    risks.extend(humanize_text(item.get("risk")) for item in (bear.get("risks") or [])[:3])
    observations = [verification_event.conclusion]
    observations.extend(humanize_text(item.get("observable_metric")) for item in (catalyst.get("catalysts") or [])[:3])
    limits = [humanize_text(item) for item in (mispricing.get("limitations") or [])]
    limits.extend(humanize_text(item) for item in (valuation.get("limitations") or []))
    return NaturalLanguagePresentation(
        verdict_label=verdict_label,
        verdict_explanation=action.implication,
        quick_conclusions=[
            headline, market, base, bull, gap_focus, blocker,
            verification_event, falsification,
        ],
        return_block=return_block,
        metric_explanations=_metric_explanations(valuation),
        causal_chains=_causal_chains(valuation, mispricing),
        important_risks=list(dict.fromkeys(item for item in risks if item and item != "暂无可靠数据")),
        next_observations=list(dict.fromkeys(item for item in observations if item and item != "暂无可靠数据")),
        terminology=TERMINOLOGY,
        limitations=list(dict.fromkeys(limits)),
    )
