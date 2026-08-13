from __future__ import annotations

import inspect
import json
import re
from typing import Any

from pydantic import BaseModel

from .llm import DeepSeekClient
from .calculations import (
    build_gap_audits,
    build_pre_valuation_probability_audit,
    calculate_extended_valuation_audits,
    calculate_price_implied_path,
    calculate_sensitivity_audit,
    calculate_unified_valuation_engine,
    evidence_probability_adjustment,
    evaluate_evidence_gate,
    finalize_risk_reward_and_decision,
    finalize_valuation_probabilities,
)
from .mispricing import calculate_mispricing_output
from .schemas import (
    BearOutput,
    CatalystOutput,
    ComparableValuationEvidenceItem,
    CompanyOutput,
    Evidence,
    EvidenceGateResult,
    ExpectationGap,
    ExpectationOutput,
    FinancialOutput,
    FinancialPeriod,
    ForecastComparison,
    IndustryOutput,
    JudgeOutput,
    JudgmentAnswer,
    LogicTree,
    LogicTreeNode,
    MarketImpliedYear,
    MarketSnapshot,
    MispricingNarrativeDraft,
    MispricingOutput,
    ResearchOutput,
    ScenarioAssumption,
    ScenarioProbabilityOutput,
    StoryOutput,
    TrackingPlan,
    ValuationAssumptionOutput,
    ValuationOutput,
)


COMMON_RULES = """
你是上市公司预期变化检测系统中的专业 Agent。准确性优先：
1. 只能基于输入事实，不得伪造数据、来源、市场一致预期或证据 ID。
2. 每个关键判断引用输入中真实存在的 evidence id；证据不足时写入 limitations。
3. A级=公司/交易所原始资料，B级=可靠行情财务平台，C级=二手新闻研报。
4. 核心当前结论必须通过 Evidence Gate：至少1条当前有效A级证据，或来自2个独立来源的当前有效B级证据；否则只能标记为假设/未知。
5. 分析日之后才可获得的行情、财务、新闻或研报属于未来证据，严禁引用，避免 Look-ahead Bias。
6. 所有增长率、利润率、费用率和概率内部统一使用0~1小数，例如28%=0.28；不得混用28与0.28。
7. 严格区分 Market Consensus（真实机构一致预期）、Model-Implied Earnings（价格反推）和 Agent Forecast（本模型预测）；Model-Implied 不代表市场已经 Price In。
8. 区分事实、推断与假设；这不是荐股，不输出买入/卖出建议。
"""


def _evidence_index(payload: dict[str, Any]) -> dict[str, dict]:
    research = payload.get("research") or {}
    return {item.get("id", ""): item for item in research.get("evidence", []) if item.get("id")}


def _claim_gate(refs: list[str], evidence: dict[str, dict]):
    return evaluate_evidence_gate(list(dict.fromkeys(refs)), evidence)


class StructuredAgent:
    output_schema: type[BaseModel]
    role_prompt: str

    def __init__(self, llm: DeepSeekClient) -> None:
        self.llm = llm

    def run(self, payload: dict[str, Any]) -> BaseModel:
        return self.llm.generate(self.output_schema, COMMON_RULES + self.role_prompt, payload)


class ResearchAgent:
    def __init__(self, llm: DeepSeekClient, collector: Any) -> None:
        self.llm = llm
        self.collector = collector

    def run(self, request: Any) -> ResearchOutput:
        materials = self.collector.collect(request)
        # The collector remains authoritative for evidence. Give the LLM a compact
        # evidence index so it synthesizes facts instead of re-emitting long source text.
        llm_materials = dict(materials)
        llm_materials["evidence"] = [
            {
                "id": item.get("id"),
                "fact": item.get("fact"),
                "source": item.get("source"),
                "source_type": item.get("source_type"),
                "source_grade": item.get("source_grade"),
                "information_date": item.get("information_date"),
                "supports_current_claim": item.get("supports_current_claim"),
                "excerpt": str(item.get("excerpt") or "")[:600],
            }
            for item in materials.get("evidence", [])
        ]
        output = self.llm.generate(
            ResearchOutput,
            COMMON_RULES + """
你是 Research Agent，只回答“发生了什么”。整理公司、事件、产品、客户、合作伙伴、产能和行业事件，
不做投资判断。明确区分历史事实和当前事实；不得把旧研报预测当成当前进展。
必须返回 evidence=[]、market_snapshot=null；系统会在校验后注入采集器的完整原始结构化数据，禁止复制证据原文。
""",
            llm_materials,
        )
        output.symbol = materials["symbol"]
        output.market_snapshot = (
            MarketSnapshot.model_validate(materials["market_snapshot"])
            if materials.get("market_snapshot") else None
        )
        output.evidence = [Evidence.model_validate(item) for item in materials["evidence"]]
        output.comparable_valuation_catalog = [
            ComparableValuationEvidenceItem.model_validate(item)
            for item in materials.get("comparable_valuation_catalog", [])
        ]
        output.limitations = list(dict.fromkeys(output.limitations + materials["limitations"]))
        return ResearchOutput.model_validate(output.model_dump())


class FinancialAgent(StructuredAgent):
    output_schema = FinancialOutput
    role_prompt = """
你是 Financial Agent，只回答过去赚了多少钱、如何赚。准确提取历史收入、毛利率、营业利润、净利润、现金流、
ROE、CapEx、应收、存货和负债。不得预测未来。识别季报累计口径，不能把Q1和全年直接比较。
指出收入、毛利、期间费用、净利润、现金流之间的历史传导与异常，单位无法确认时留空。
EBIT、EBITDA和D&A历史值由程序从结构化年度报表注入，不得自行补造。
"""

    def run(self, payload: dict[str, Any]) -> FinancialOutput:
        output = FinancialOutput.model_validate(super().run(payload).model_dump())
        evidence = (payload.get("research") or {}).get("evidence", [])
        statement = next(
            (item for item in evidence if item.get("source_type") == "financial_statement_data"),
            None,
        )
        if not statement:
            output.limitations.append("缺少年度利润表/现金流量表结构化证据，历史EBIT、EBITDA及D&A锚点不可用")
            return output
        try:
            records = json.loads(statement.get("excerpt") or "[]")
        except (TypeError, json.JSONDecodeError):
            records = []
        evidence_id = str(statement.get("id") or "")
        fields = (
            "revenue", "gross_profit", "operating_profit", "ebit", "ebitda",
            "depreciation_amortization", "net_profit",
        )
        periods = list(output.historical)
        for record in records:
            period = str(record.get("period") or "")
            year = period[:4]
            existing = next((
                item for item in periods
                if str(item.period).startswith(year)
                and ("12-31" in str(item.period) or str(item.period).strip() == year)
            ), None)
            values = existing.model_dump() if existing else {"period": period}
            values.update({key: record.get(key) for key in fields if record.get(key) is not None})
            values["period"] = period
            values["ebit_basis"] = str(record.get("ebit_basis") or "")
            values["evidence_refs"] = list(dict.fromkeys(values.get("evidence_refs", []) + ([evidence_id] if evidence_id else [])))
            field_refs = dict(values.get("field_evidence_refs") or {})
            for key in fields:
                if record.get(key) is not None and evidence_id:
                    field_refs[key] = [evidence_id]
            values["field_evidence_refs"] = field_refs
            normalized = FinancialPeriod.model_validate(values)
            if existing:
                periods[periods.index(existing)] = normalized
            else:
                periods.append(normalized)
        output.historical = sorted(periods, key=lambda item: str(item.period))
        if evidence_id:
            output.evidence_refs = list(dict.fromkeys(output.evidence_refs + [evidence_id]))
        output.quality_flags = list(dict.fromkeys(output.quality_flags + [
            "历史EBIT使用东方财富营业利润proxy；Forecast Reasonableness Audit中披露口径差异",
            "历史D&A来自现金流量表固定资产折旧、无形资产及长期待摊费用摊销等项目之和",
        ]))
        return FinancialOutput.model_validate(output.model_dump())


class IndustryAgent(StructuredAgent):
    output_schema = IndustryOutput
    role_prompt = """
你是 Industry Agent，分析宏观→行业→产业链→公司环节→上下游→竞争格局。
重点识别会让收入或利润非线性变化的驱动，并区分当前证据与历史背景。
"""


class CompanyAgent(StructuredAgent):
    output_schema = CompanyOutput
    role_prompt = """
你是 Company Agent，把行业机会映射到公司能力，但行业增长绝不自动等于公司增长。
每条 opportunity_mapping 必须尽量量化：行业增速、公司收入增速、当前/目标市场份额、毛利率影响、
净利润影响、传导时滞和目标期，所有比例使用0~1小数。必须解释：行业需求→公司可服务市场→
Company Capture Rate/份额变化→公司收入→毛利率→净利润；数据不足时留空并列出 blockers。
"""

    def run(self, payload: dict[str, Any]) -> CompanyOutput:
        output = super().run(payload)
        evidence = _evidence_index(payload)
        for mapping in output.opportunity_mapping:
            mapping.evidence_gate = _claim_gate(mapping.evidence_refs, evidence)
            if mapping.industry_growth_rate not in {None, 0} and mapping.company_revenue_growth_rate is not None:
                mapping.company_capture_rate = mapping.company_revenue_growth_rate / mapping.industry_growth_rate
            if mapping.current_market_share is not None and mapping.target_market_share is not None:
                mapping.market_share_change = mapping.target_market_share - mapping.current_market_share
            complete = all(value is not None for value in (
                mapping.industry_growth_rate, mapping.company_revenue_growth_rate,
                mapping.company_capture_rate, mapping.gross_margin_impact,
            ))
            mapping.validation_status = "validated" if complete and mapping.evidence_gate.passed else "hypothesis"
        return CompanyOutput.model_validate(output.model_dump())


class StoryAgent(StructuredAgent):
    output_schema = StoryOutput
    role_prompt = """
你是 Story Evolution Agent，只建立“为什么可能上涨”的投资逻辑树。诉讼、现金流、应收、存货、竞争恶化、战略失败等必须留给 Risk Tree，不得混入 Story Probability。
要求：
- node_id 使用 S1/S2...；depends_on 只填真正的上行因果前提；overlaps_with 标出重复或嵌套节点，禁止依赖环。
- 推荐链路：行业变化→公司捕获率/份额→收入→毛利/费用→净利润→估值，不要强行补齐无证据节点。
- 每个节点的 base conditional probability 表示 P(本节点成立|全部父节点成立)，条件按P(C1)、P(C2|C1)…填写。
- 程序用“基础概率×证据修正因子”调整节点概率，再对祖先闭包联合计算；不得机械封顶30%。
- key_node_ids 只保留同一核心上行链上真正决定结果的2~3个节点，并指出 critical_uncertainty。
"""

    def run(self, payload: dict[str, Any]) -> StoryOutput:
        output = super().run(payload)
        evidence = _evidence_index(payload)
        adjustments: dict[str, tuple[float, float, str]] = {}
        for node in output.stories:
            refs = node.evidence_for + [ref for condition in node.conditions for ref in condition.evidence_refs]
            gate, factor, confidence, rationale = evidence_probability_adjustment(refs, evidence)
            node.evidence_gate = gate
            node.evidence_factor = factor
            node.claim_status = "supported" if gate.passed else "hypothesis"
            adjustments[node.node_id] = (factor, confidence, rationale)
            if not gate.passed:
                output.limitations.append(f"{node.node_id}未通过 Evidence Gate，证据因子调整为{factor:.2f}，但不做机械封顶")
        result = StoryOutput.model_validate(output.model_dump())
        for node in result.stories:
            factor, confidence, rationale = adjustments.get(node.node_id, (1.0, 0.5, ""))
            node.probability_assessment.evidence_factor = factor
            node.probability_assessment.confidence = confidence
            node.probability_assessment.rationale = rationale
        return result


class ExpectationAgent(StructuredAgent):
    output_schema = ExpectationOutput
    role_prompt = """
你是 Model-Implied Expectations Agent。严格拆分三类序列：
- Model-Implied Earnings：只填写要求回报率、退出PE、净利率假设，派生值由程序从当前价格反推；它不等于市场已计价。
- Market Consensus：只有真实券商/机构当前预测、预测日期和至少两个独立来源均有证据时才填写；否则 available=false。
- Agent Forecast：本阶段禁止填写，由后续程序化估值引擎生成。
不得输出“Price In/已计价/未计价”判断。所有比例使用0~1小数。
"""

    def run(self, payload: dict[str, Any]) -> ExpectationOutput:
        output = super().run(payload)
        research = payload.get("research") or {}
        financial = FinancialOutput.model_validate(payload.get("financial") or {})
        output = calculate_price_implied_path(output, research, financial)
        snapshot = research.get("market_snapshot") or {}
        snapshot_refs = snapshot.get("evidence_refs") or []
        evidence = _evidence_index(payload)
        for item in output.price_implied_path:
            item.evidence_refs = list(dict.fromkeys(item.evidence_refs + snapshot_refs))

        reliable_consensus = False
        consensus_by_year: dict[str, Any] = {}
        for item in output.consensus_estimates:
            gate = _claim_gate(item.evidence_refs, evidence)
            item.available = bool(
                item.available and item.source_count >= 2 and item.as_of is not None and gate.passed
            )
            if item.available:
                consensus_by_year[item.year] = item
                reliable_consensus = True
            else:
                item.revenue = None
                item.net_profit = None
                output.data_gaps.append(
                    f"{item.year} Market Consensus 缺少真实机构预测日期、两个独立来源或未通过 Evidence Gate"
                )
        output.consensus_available = reliable_consensus

        previous_actual = {
            (item.metric, item.period): item.actual_value for item in output.expectation_gaps
            if item.actual_value is not None
        }
        gaps: list[ExpectationGap] = []
        for item in output.price_implied_path:
            consensus = consensus_by_year.get(item.year)
            for metric, implied, consensus_value, unit in (
                ("net_profit", item.implied_net_profit, getattr(consensus, "net_profit", None), "亿元"),
                ("revenue", item.implied_revenue, getattr(consensus, "revenue", None), "亿元"),
                ("net_profit_growth", item.implied_net_profit_growth, None, "ratio"),
                ("revenue_growth", item.implied_revenue_growth, None, "ratio"),
            ):
                if implied is None and consensus_value is None:
                    continue
                gaps.append(ExpectationGap(
                    metric=metric,
                    period=item.year,
                    model_implied_value=implied,
                    market_consensus_value=consensus_value,
                    market_consensus_available=bool(consensus and consensus.available and consensus_value is not None),
                    agent_forecast_value=None,
                    actual_value=previous_actual.get((metric, item.year)),
                    unit=unit,
                    gap_direction="unknown",
                    inference_method="Model-Implied由当前价格、要求回报率、退出PE和净利率假设反推；不代表市场已计价",
                    evidence_refs=list(dict.fromkeys(item.evidence_refs + (consensus.evidence_refs if consensus else []))),
                ))
        output.expectation_gaps = gaps
        output.price_in_assessments = []
        output.priced_in = []
        output.not_priced_in = []
        output.limitations = list(dict.fromkeys(output.limitations + [
            "系统不把 Model-Implied Earnings 表述为 Price In；是否形成市场共识必须由独立机构预测证据证明。"
        ]))
        return ExpectationOutput.model_validate(output.model_dump())


class ValuationAgent(StructuredAgent):
    output_schema = ValuationAssumptionOutput
    role_prompt = """
你是 Valuation Assumption Agent。Probability Agent已经独立生成并通过程序门控的同一概率空间；你不得重估或修改概率，输出中的概率字段会被程序覆盖。
1. 系统会按悲观、基准、乐观情景分别调用你；每次只输出指定的一个情景，并使用输入要求的相同连续3个预测年份，严格采用输入 probability_output 对应情景的事件定义、边界和horizon。
2. 所有增长率、毛利率、费用率、税率和概率使用0~1小数，例如28%=0.28。
3. 每期只填写增长率、利润率、费用率、估值倍数及现金、债务、账面净资产假设；倍数只能来自输入中的 comparable_valuation Evidence catalog，缺乏证据必须填0，禁止自行拍倍数。
4. assumption_evidence_refs必须按变量名分别映射真实Evidence ID。倍数键必须分别使用pe_multiple、ev_ebitda_multiple、ev_sales_multiple、pb_multiple；一种metric的证据不得支持另一种方法。
5. 如果增长率、利润率或D&A/Revenue突破历史区间，必须用变量级证据解释结构性变化；证据不足时系统将标记unsupported_forecast并禁止估值。
6. Bear/Base/Bull每个情景都必须填写business_triggers。每个trigger必须包含trigger_id、真实业务变化business_change、精确financial_variable、相对Base的direction、传导mechanism、当前有效Evidence Ref和counter_evidence_refs；没有反方证据时使用空列表，禁止把“未找到反方证据”写成假设已验证。
7. Bear或Bull相对Base变化的每个revenue_growth、gross_margin、rd_expense_ratio、selling_expense_ratio、admin_expense_ratio、other_operating_expense_ratio、financial_expense_ratio、depreciation_amortization_ratio、tax_rate，都必须逐年有对应business_trigger，说明真实业务变化→变量方向→财务传导机制并引用证据；禁止仅为拉开情景而机械调整收入增速、毛利率或费用率。
8. 任何估值倍数都必须有独立、同口径Comparable Evidence支持，并说明业务/风险变化为何影响倍数；经营变量证据不能替代倍数证据，禁止机械调倍数。Bull相对Base存在重大毛利率提升时还必须填写margin_expansion_drivers，其影响合计应解释毛利率差，并分别填写支持证据与counter_evidence_refs。
9. 禁止自行填写 Revenue、Gross Profit、Opex、Net Profit、EBITDA、EV、股权价值、EPS或目标价。
10. 净利润>0才允许PE；净利润≤0时程序优先EV/EBITDA，其次EV/Sales，必要时PB。无可靠Comparable Evidence时保持0/null。
11. 禁止人为Bear/Base/Bull排序缩放，禁止为了得到目标价修改倍数或结果；跨方法排序反转必须fail-closed。
12. 只提供假设；Python负责财务、概率、估值、日期、证据与Expected Return规则。
"""

    def run(self, payload: dict[str, Any]) -> ValuationOutput:
        evidence = _evidence_index(payload)
        probability = self.llm.generate(
            ScenarioProbabilityOutput,
            COMMON_RULES + """
你是独立的Pre-Valuation Probability Agent。只定义一个共同状态变量S，并在相同horizon下生成且仅生成Bear/Base/Bull三个互斥、穷尽情景。
- 三个base_probability必须在生成阶段精确合计1.0，不得先独立估计后接受异常总和。
- 每个情景必须分别填写非空reasoning和probability_basis_refs；probability_basis_refs只能引用输入中当前有效、非未来、支持当前结论的A级或B级Evidence ID。
- reasoning必须明确写出base prior如何建立、每条有效证据相对base导致何种上调/下调/不变、调整方向与机制，以及为何得到最终base_probability；禁止无依据拍概率、只报数字或用泛化措辞替代base→evidence adjustment依据。
- evidence_refs用于情景事实边界，probability_basis_refs用于概率数值依据；两者都不得伪造，C级、未来证据或无效ref不能作为概率依据。证据不足时必须在limitations如实说明，不能凑概率理由。
- 每个情景必须给不重叠的event_definition、可验证boundary_conditions，并解释为什么三个区间互斥且穷尽共同状态变量全部结果。
- 不生成财务预测、估值倍数、目标价或Expected Return。无法形成合规概率空间时如实输出，程序将在本层阻断。
""",
            payload,
        )
        pre_audit = build_pre_valuation_probability_audit(probability, evidence)
        expectation = payload.get("expectation") or {}
        path = expectation.get("price_implied_path") or expectation.get("market_implied_path") or []
        if not pre_audit.passed:
            blocked = ValuationOutput(
                reverse_valuation=[MarketImpliedYear.model_validate(item) for item in path],
                pre_valuation_probability_audit=pre_audit,
                limitations=list(dict.fromkeys(probability.limitations + pre_audit.issues + [
                    "Probability Agent层已fail-closed；未调用Valuation Assumption Agent，未生成财务预测或估值。"
                ])),
            )
            blocked = calculate_extended_valuation_audits(
                blocked, evidence, FinancialOutput.model_validate(payload.get("financial") or {}),
                payload.get("research") or {},
            )
            blocked = finalize_valuation_probabilities(blocked)
            blocked = build_gap_audits(blocked)
            blocked = calculate_sensitivity_audit(blocked)
            return finalize_risk_reward_and_decision(
                blocked,
                FinancialOutput.model_validate(payload.get("financial") or {}),
                payload.get("research") or {},
            )

        valuation_payload = dict(payload)
        valuation_payload["probability_output"] = probability.model_dump(mode="json")
        horizon_text = " ".join(item.horizon for item in probability.scenarios)
        horizon_years = [int(value) for value in re.findall(r"20\d{2}", horizon_text)]
        as_of_text = str(((payload.get("research") or {}).get("market_snapshot") or {}).get("as_of") or "")
        start_year = (
            int(as_of_text[:4]) if re.fullmatch(r"20\d{2}.*", as_of_text)
            else (min(horizon_years) if horizon_years else 2026)
        )
        required_years = [str(start_year + offset) for offset in range(3)]
        scenario_assumptions: list[ScenarioAssumption] = []
        generation_limitations: list[str] = []
        for source in probability.scenarios:
            scenario_payload = dict(valuation_payload)
            scenario_payload["required_scenario"] = {
                "name": source.name,
                "event_definition": source.event_definition,
                "boundary_conditions": source.boundary_conditions,
                "horizon": source.horizon,
                "required_years": required_years,
            }
            scenario = self.llm.generate(
                ScenarioAssumption,
                COMMON_RULES + self.role_prompt + f"""
本次只返回{source.name}一个情景的ScenarioAssumption对象，不得返回其他情景或外层scenarios数组。
name必须为{source.name}；projections必须且只能包含{', '.join(required_years)}三个年份。
内容保持精炼，每个assumptions、mechanism和description字段只写一句话；不要重复输入事实。
""",
                scenario_payload,
            )
            if scenario.name != source.name:
                generation_limitations.append(
                    f"Valuation Agent返回{scenario.name}而不是指定的{source.name}；该情景已丢弃"
                )
                continue
            scenario_assumptions.append(scenario)
        assumptions = ValuationAssumptionOutput(
            scenarios=scenario_assumptions,
            limitations=generation_limitations,
        )
        business_triggers_by_name = {
            scenario.name: list(scenario.business_triggers) for scenario in assumptions.scenarios
        }
        probability_by_name = {item.name: item for item in probability.scenarios}
        for scenario in assumptions.scenarios:
            source = probability_by_name.get(scenario.name)
            if source:
                scenario.probability = source.base_probability
                scenario.base_probability = source.base_probability
                scenario.event_definition = source.event_definition
                scenario.boundary_conditions = source.boundary_conditions
                scenario.horizon = source.horizon
                scenario.probability_basis = [source.reasoning]
                scenario.evidence_refs = list(source.evidence_refs)
        output = ValuationOutput.model_validate(assumptions.model_dump())
        for scenario in output.scenarios:
            scenario.business_triggers = list(business_triggers_by_name.get(scenario.name, []))
        output.pre_valuation_probability_audit = pre_audit
        output.reverse_valuation = [MarketImpliedYear.model_validate(item) for item in path]
        research = payload.get("research") or {}
        financial = FinancialOutput.model_validate(payload.get("financial") or {})

        unique: dict[str, Any] = {}
        for scenario in output.scenarios:
            if scenario.name not in unique:
                unique[scenario.name] = scenario
        output.scenarios = [unique[name] for name in ("bear", "base", "bull") if name in unique]
        missing = [name for name in ("bear", "base", "bull") if name not in unique]
        if missing:
            output.limitations.append(f"缺少必需估值情景：{', '.join(missing)}；程序不伪造缺失情景")
        year_sets = [{item.year for item in scenario.projections} for scenario in output.scenarios]
        if year_sets and (any(len(years) != 3 for years in year_sets) or len({tuple(sorted(years)) for years in year_sets}) != 1):
            output.limitations.append("Bear/Base/Bull 未提供完全一致的三个预测年份，跨情景比较受限")

        evidence = _evidence_index(payload)
        output = calculate_unified_valuation_engine(output, research, financial, evidence)
        for scenario in output.scenarios:
            refs = list(dict.fromkeys(
                scenario.evidence_refs + [ref for item in scenario.projections for ref in item.evidence_refs]
            ))
            gate, factor, confidence, rationale = evidence_probability_adjustment(refs, evidence)
            scenario.evidence_refs = refs
            scenario.evidence_gate = gate
            scenario.evidence_factor = factor
            raw = scenario.base_probability * factor
            scenario.probability = min(1.0, raw)
            scenario.probability_assessment.base_probability = scenario.base_probability
            scenario.probability_assessment.evidence_factor = factor
            scenario.probability_assessment.raw_adjusted_probability = raw
            scenario.probability_assessment.adjusted_probability = raw
            scenario.probability_assessment.normalized_probability = scenario.probability
            scenario.probability_assessment.confidence = confidence
            scenario.probability_assessment.rationale = rationale
            if not gate.passed:
                output.limitations.append(
                    f"{scenario.name}情景未通过 Evidence Gate，概率按证据因子{factor:.2f}下调；不机械封顶"
                )
        output = calculate_extended_valuation_audits(output, evidence, financial, research)
        output = finalize_valuation_probabilities(output)
        output = build_gap_audits(output)
        output = calculate_sensitivity_audit(output)

        base = next((item for item in output.scenarios if item.name == "base"), None)
        base_by_year = {item.year: item for item in (base.projections if base else [])}
        implied_by_year = {item.year: item for item in output.reverse_valuation}
        consensus_by_year = {
            item.get("year"): item for item in expectation.get("consensus_estimates", [])
            if item.get("available") is True
        }
        comparisons: list[ForecastComparison] = []
        for year in sorted(set(base_by_year) | set(implied_by_year) | set(consensus_by_year)):
            agent_profit = base_by_year.get(year).net_profit if year in base_by_year else None
            implied_profit = implied_by_year.get(year).implied_net_profit if year in implied_by_year else None
            consensus = consensus_by_year.get(year) or {}
            consensus_profit = consensus.get("net_profit")
            consensus_available = bool(consensus and consensus_profit is not None)
            gap_implied = agent_profit - implied_profit if None not in {agent_profit, implied_profit} else None
            gap_consensus = agent_profit - consensus_profit if consensus_available and agent_profit is not None else None
            if gap_implied is None or implied_profit in {None, 0}:
                direction = "unknown"
            elif abs(gap_implied / implied_profit) < 0.05:
                direction = "neutral"
            else:
                direction = "positive" if gap_implied > 0 else "negative"
            comparisons.append(ForecastComparison(
                year=year,
                model_implied_profit=implied_profit,
                consensus_profit=consensus_profit,
                consensus_available=consensus_available,
                agent_forecast_profit=agent_profit,
                gap_vs_implied=gap_implied,
                gap_vs_consensus=gap_consensus,
                direction=direction,
                explanation="程序化 Agent Forecast 与 Model-Implied Earnings 及可用 Market Consensus 的同年比较",
            ))
        output.forecast_comparison = comparisons
        return finalize_risk_reward_and_decision(output, financial, research)


class CatalystAgent(StructuredAgent):
    output_schema = CatalystOutput
    role_prompt = """
你是 Catalyst Agent，只研究什么会让市场重新定价。覆盖30天、季度、半年和一年；每项必须给可观察指标、
阈值、时间、关联投资逻辑节点和“事件→新证据→预期调整→估值调整”机制。每个Catalyst必须填写：
expected_date、verifies_assumption_ids、affected_financial_variables、favorable_result、adverse_result、
scenario_probability_effects（明确Bear/Base/Bull各自方向或幅度，仅作未来验证映射）、支持证据和counter_evidence_refs。
没有反方证据时counter_evidence_refs使用空列表；不得把“未找到反方证据”写成假设已验证。缺任一项必须写入limitations，不得编造。
条件概率只表示Catalyst自身发生概率：程序仅以该Catalyst的base_probability乘自身证据因子进行审计；
绝不直接修改、覆盖、归一化或重新派生Valuation的Bear/Base/Bull概率。证据弱只降低自身置信度/概率，不得机械封顶30%。
"""

    def run(self, payload: dict[str, Any]) -> CatalystOutput:
        output = super().run(payload)
        evidence = _evidence_index(payload)
        for item in output.catalysts:
            refs = list(dict.fromkeys(
                item.evidence_refs + [ref for condition in item.conditions for ref in condition.evidence_refs]
            ))
            gate, factor, confidence, rationale = evidence_probability_adjustment(refs, evidence)
            item.evidence_refs = refs
            item.evidence_gate = gate
            item.evidence_factor = factor
            item.probability = min(1.0, item.base_probability * factor)
            item.probability_assessment.base_probability = item.base_probability
            item.probability_assessment.evidence_factor = factor
            item.probability_assessment.adjusted_probability = item.probability
            item.probability_assessment.confidence = confidence
            item.probability_assessment.rationale = rationale
            if not gate.passed:
                output.limitations.append(
                    f"{item.catalyst_id}未通过 Evidence Gate，证据因子调整为{factor:.2f}，未使用30%封顶"
                )
        return CatalystOutput.model_validate(output.model_dump())


class BearAgent(StructuredAgent):
    output_schema = BearOutput
    role_prompt = """
你是独立 Red Team。风险树必须独立于上行投资逻辑树，覆盖诉讼、现金流、应收、存货、竞争、战略失败，
以及需求、客户、技术、估值和治理风险。每个风险给 risk_id、depends_on、基础概率、影响、证据和验证方式。
逐条攻击行业需求→公司捕获→收入→毛利→利润→估值因果链，并给替代解释与断裂条件。
程序按“基础概率×证据修正因子”保留每项风险的诊断概率；风险之间存在依赖，禁止用独立并集公式生成顶部 Downside Probability。
顶部 Downside Event Probability 由 Judge 直接映射为互斥三情景中的 P(Bear)，风险树只解释机制与证伪条件。
"""

    def run(self, payload: dict[str, Any]) -> BearOutput:
        output = super().run(payload)
        evidence = _evidence_index(payload)
        known_ids = {item.risk_id for item in output.risks}
        for item in output.risks:
            item.depends_on = [parent for parent in item.depends_on if parent in known_ids and parent != item.risk_id]
            gate, factor, confidence, rationale = evidence_probability_adjustment(item.evidence_refs, evidence)
            item.evidence_gate = gate
            item.evidence_factor = factor
            item.probability = min(1.0, item.base_probability * factor)
            item.probability_assessment.base_probability = item.base_probability
            item.probability_assessment.evidence_factor = factor
            item.probability_assessment.adjusted_probability = item.probability
            item.probability_assessment.confidence = confidence
            item.probability_assessment.rationale = rationale
            if not gate.passed:
                output.limitations.append(
                    f"{item.risk_id}未通过 Evidence Gate，风险概率按证据因子{factor:.2f}调整"
                )
        material = [item for item in output.risks if item.impact in {"high", "critical"}]
        roots = [item for item in material if not item.depends_on] or material or output.risks
        output.downside_risk_probability = 0
        output.limitations.append(
            "风险项高度相关，未使用1-∏(1-p)聚合；顶部Downside Event Probability由Judge映射为P(Bear)。"
        )
        root = max(roots, key=lambda item: item.probability, default=None)
        output.risk_tree = LogicTree(
            tree_type="risk",
            root_id=root.risk_id if root else "",
            nodes=[LogicTreeNode(
                node_id=item.risk_id,
                label=f"{item.category}: {item.risk}",
                metric="Risk Diagnostic",
                depends_on=item.depends_on,
                probability=item.probability,
                evidence_refs=item.evidence_refs,
            ) for item in output.risks],
            summary="风险树仅作相关风险的机制诊断；不假设独立、不聚合为顶部概率。顶部Downside Event=P(Bear)。",
        )
        return BearOutput.model_validate(output.model_dump())


class MispricingAgent(StructuredAgent):
    output_schema = MispricingNarrativeDraft
    role_prompt = """
你是独立 Expectation / Mispricing Agent，位于 Bear 之后、Judge 之前。你只输出结构化叙事草稿，严禁自行计算或改写任何数字、概率、估值、Gate、Edge分数或Verdict。
1. 识别“当前市场在赌什么”和“Agent认为哪里错”，但必须严格区分Market Consensus、Model-Implied与Agent Forecast。Market Consensus unavailable时不得补造、不得用Model-Implied替代。
2. drivers只描述Revenue、Gross Margin、EBIT Margin、Business Mix、Market Share、Valuation Multiple等Gap Driver；每项必须引用输入中真实Evidence ID，并链接现有Catalyst ID。
3. 每个driver必须给可验证指标、时间窗口和Re-rating/De-rating机制，形成Gap→Driver→Evidence→Catalyst→Verification→Time→Re-rating链。
4. 每个核心driver必须给falsifiable_condition：当前判断、关键变量、验证指标、证实阈值、证伪阈值、最晚验证时间。
5. 只有当前有效反证已明确触发证伪阈值时current_status才可为refuted，并必须引用counter_evidence_refs；未来证据严禁使用。
6. Python将权威拼接三类预期、重算Gap、校验证据与Catalyst、计算六分项Edge和最终Verdict；你的草稿不得影响这些确定性结果。
"""

    def run(self, payload: dict[str, Any]) -> MispricingOutput:
        draft = MispricingNarrativeDraft.model_validate(super().run(payload).model_dump())
        optional_context = {
            "industry": payload.get("industry") or {},
            "company": payload.get("company") or {},
        }
        supported_parameters = inspect.signature(calculate_mispricing_output).parameters
        accepts_extra_context = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in supported_parameters.values()
        )
        return calculate_mispricing_output(
            draft,
            payload.get("expectation") or {},
            payload.get("valuation") or {},
            payload.get("catalyst") or {},
            payload.get("bear") or {},
            payload.get("research") or {},
            payload.get("financial") or {},
            **{
                name: value for name, value in optional_context.items()
                if accepts_extra_context or name in supported_parameters
            },
            agent_output_available=True,
        )


class JudgeAgent(StructuredAgent):
    output_schema = JudgeOutput
    role_prompt = """Judge仅执行确定性复制与文本汇总；run中禁止调用LLM或任何重算函数。"""

    def run(self, payload: dict[str, Any]) -> JudgeOutput:
        story = StoryOutput.model_validate(payload.get("story") or {})
        valuation = ValuationOutput.model_validate(payload.get("valuation") or {})
        catalyst = CatalystOutput.model_validate(payload.get("catalyst") or {})
        bear = BearOutput.model_validate(payload.get("bear") or {})
        mispricing = MispricingOutput.model_validate(payload.get("mispricing") or {})

        decision = valuation.decision_probabilities
        verdict = valuation.deterministic_investment_verdict
        basis_by_metric = {
            item.metric: item for item in valuation.deterministic_probability_basis
        }

        def copied_probability(value: float | None) -> float:
            # Judge schema requires concrete numbers. Zero is display-only when an
            # authoritative deterministic probability is unavailable/blocked.
            return float(value) if value is not None else 0.0

        bull_probability = copied_probability(decision.probability_of_thesis_success)
        fundamental_probability = copied_probability(
            basis_by_metric.get("基本面改善").value
            if basis_by_metric.get("基本面改善") else None
        )
        valuation_probability = copied_probability(
            basis_by_metric.get("估值重估").value
            if basis_by_metric.get("估值重估") else None
        )
        downside_probability = copied_probability(
            basis_by_metric.get("Downside Event").value
            if basis_by_metric.get("Downside Event") else None
        )
        negative_return_probability = copied_probability(
            decision.probability_of_negative_return
        )

        logic_steps = list(mispricing.investment_logic_chain.steps)
        logic_lines = [
            f"{item.step_id}: {item.premise} → {item.conclusion}"
            for item in logic_steps
        ]
        story_lines = [
            f"{item.node_id}: {item.name}"
            for item in story.stories
        ]
        trackable_lines = [
            f"{item.condition_id}: {item.description}；{item.confirms_if}；{item.invalidates_if}"
            for item in mispricing.trackable_conditions
        ]
        validation_lines = [
            f"{item.validation_id}: {item.question}（{item.expected_date}）"
            for item in mispricing.future_validation_checklist.items
        ]

        chain_refs = list(dict.fromkeys(
            ref for item in logic_steps for ref in item.evidence_refs
        ))
        scenario_refs = list(dict.fromkeys(
            ref for item in valuation.scenarios for ref in item.evidence_refs
        ))
        story_refs = list(dict.fromkeys(
            ref for item in story.stories
            for ref in item.evidence_for
            + [ref for condition in item.conditions for ref in condition.evidence_refs]
        ))
        bear_refs = list(dict.fromkeys(
            [ref for item in bear.risks for ref in item.evidence_refs]
            + [ref for item in bear.causal_attacks for ref in item.evidence_refs]
        ))
        catalyst_refs = list(dict.fromkeys(
            ref for item in catalyst.catalysts
            for ref in item.evidence_refs
            + [ref for condition in item.conditions for ref in condition.evidence_refs]
        ))
        all_refs = list(dict.fromkeys(
            chain_refs + scenario_refs + story_refs + bear_refs + catalyst_refs
        ))

        # Copy the most conservative already-computed chain/scenario gate. Judge
        # never calls evaluate_evidence_gate or creates a new gate conclusion.
        existing_gates = (
            [item.evidence_gate for item in logic_steps]
            + [item.evidence_gate for item in valuation.scenarios]
        )
        conservative_gate = next(
            (gate for gate in existing_gates if not gate.passed),
            existing_gates[0] if existing_gates else EvidenceGateResult(),
        )
        answer_status = "supported" if conservative_gate.passed else "hypothesis"
        confidence = copied_probability(decision.confidence)

        bottleneck = next(
            (item for item in story.stories if item.node_id == story.bottleneck_node_id),
            None,
        )
        most_uncertain_link = (
            f"{bottleneck.node_id} {bottleneck.name}：{bottleneck.critical_uncertainty}"
            if bottleneck else (story.key_story_change or story.evolution_summary)
        )
        bear_mechanisms = [
            item.alternative_explanation for item in bear.causal_attacks
            if item.alternative_explanation
        ]
        maximum_falsifier = (
            bear.strongest_disproof
            or next(iter(bear.thesis_breakers), "证据不足，尚无可复制的证伪条件")
        )

        next_30_days = [
            f"{item.catalyst_id}: {item.event}（{item.expected_date or item.timeframe}）"
            for item in catalyst.catalysts if item.timeframe == "30d"
        ]
        next_quarter = [
            f"{item.catalyst_id}: {item.event}（{item.expected_date or item.timeframe}）"
            for item in catalyst.catalysts if item.timeframe == "quarter"
        ]
        next_half_year = [
            f"{item.catalyst_id}: {item.event}（{item.expected_date or item.timeframe}）"
            for item in catalyst.catalysts if item.timeframe in {"half_year", "one_year"}
        ]
        if validation_lines:
            next_half_year.extend(validation_lines)

        market_summary = [
            text for text in (
                mispricing.verdict.summary,
                mispricing.verdict.maximum_gap,
                mispricing.verdict.repricing_event,
            ) if text
        ]
        forecast_summary = [
            text for text in (
                verdict.summary,
                verdict.base_forecast_assessment,
                verdict.bull_forecast_assessment,
            ) if text
        ]
        gap_drivers = [
            item.conclusion for item in logic_steps if item.conclusion
        ]

        return JudgeOutput(
            current_fundamentals=(
                verdict.key_financial_driver or verdict.summary or "确定性估值结论不可用"
            ),
            current_market_story=(
                mispricing.verdict.summary or "Mispricing确定性结论不可用"
            ),
            market_implied_summary=market_summary,
            agent_forecast_summary=forecast_summary,
            expectation_gap_drivers=gap_drivers,
            why_market_underestimates=JudgmentAnswer(
                conclusion=mispricing.verdict.summary,
                mechanisms=gap_drivers,
                most_likely_failure=mispricing.verdict.failure_point,
                falsifier=mispricing.verdict.failure_point,
                confidence=confidence,
                evidence_refs=chain_refs,
                evidence_gate=conservative_gate,
                claim_status=answer_status,
            ),
            agent_most_likely_wrong=JudgmentAnswer(
                conclusion=maximum_falsifier,
                mechanisms=bear_mechanisms,
                most_likely_failure=most_uncertain_link,
                falsifier=maximum_falsifier,
                confidence=confidence,
                evidence_refs=bear_refs,
                evidence_gate=conservative_gate,
                claim_status=answer_status,
            ),
            priced_in=[],
            not_priced_in=[],
            potential_new_stories=story_lines,
            necessary_conditions=trackable_lines,
            story_chain=logic_lines or story_lines,
            most_uncertain_link=most_uncertain_link,
            financial_impact=(
                verdict.key_financial_driver or verdict.base_forecast_assessment or verdict.summary
            ),
            valuation_impact=verdict.summary,
            maximum_falsifier=maximum_falsifier,
            story_probability=copied_probability(story.overall_probability),
            story_chain_probability=copied_probability(story.overall_probability),
            bull_thesis_probability=bull_probability,
            fundamental_delivery_probability=fundamental_probability,
            valuation_expansion_probability=valuation_probability,
            downside_risk_probability=downside_probability,
            probability_of_negative_return=negative_return_probability,
            probability_basis=list(valuation.deterministic_probability_basis),
            investment_verdict=verdict,
            mispricing_verdict=mispricing.verdict,
            evidence_based_verdict=mispricing.evidence_based_verdict,
            positive_expectation_gap_score=mispricing.positive_expectation_gap_score,
            risk_score=mispricing.risk_score,
            tracking_plan=TrackingPlan(
                next_30_days=next_30_days,
                next_quarter=next_quarter,
                next_half_year=next_half_year,
            ),
            evidence_refs=all_refs,
            evidence_gate=conservative_gate,
        )


AGENT_SEQUENCE = [
    ("financial", FinancialAgent, ["research"]),
    ("industry", IndustryAgent, ["research"]),
    ("company", CompanyAgent, ["research", "financial", "industry"]),
    ("story", StoryAgent, ["research", "financial", "industry", "company"]),
    ("expectation", ExpectationAgent, ["research", "financial", "story"]),
    ("valuation", ValuationAgent, ["research", "financial", "story", "expectation"]),
    ("catalyst", CatalystAgent, ["research", "story", "expectation", "valuation"]),
    (
        "bear", BearAgent,
        ["research", "financial", "industry", "company", "story", "expectation", "valuation", "catalyst"],
    ),
    (
        "mispricing", MispricingAgent,
        ["research", "financial", "industry", "company", "story", "expectation", "valuation", "catalyst", "bear"],
    ),
    (
        "judge", JudgeAgent,
        ["research", "financial", "industry", "company", "story", "expectation", "valuation", "catalyst", "bear", "mispricing"],
    ),
]
