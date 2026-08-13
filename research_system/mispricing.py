from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import BaseModel

from .calculations import evaluate_evidence_gate
from .evidence_status import (
    build_evidence_acquisition_plan,
    build_evidence_based_verdict,
    build_evidence_status_layer,
)
from .schemas import (
    BearOutput,
    CatalystOutput,
    CatalystVerificationAudit,
    CatalystVerificationItem,
    CompanyOutput,
    ExpectationGapChain,
    ExpectationOutput,
    ExpectationPoint,
    FalsifiableCondition,
    FinancialOutput,
    FutureValidationChecklist,
    FutureValidationItem,
    IndustryOutput,
    InvestmentEdgeComponent,
    InvestmentEdgeScore,
    InvestmentLogicChainAudit,
    InvestmentLogicStep,
    MispricingClassification,
    MispricingDriver,
    MispricingGap,
    MispricingGate,
    MispricingNarrativeDraft,
    MispricingOutput,
    MispricingVerdict,
    TrackableInvestmentCondition,
    ValuationOutput,
    VariableGapContribution,
)


def _dict(value: Any) -> dict[str, Any]:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return dict(value or {})


def _clamp(value: float) -> float:
    return max(0.0, min(100.0, float(value)))


def _as_date(value: Any) -> date | None:
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value)[:10]) if value else None
    except ValueError:
        return None


def _implied_operating_points(
    implied: dict[str, Any], base_projection: dict[str, Any] | None
) -> list[ExpectationPoint]:
    """Build only disclosed Model-Implied values; derive margins only under explicit Base assumptions."""
    refs = list(implied.get("evidence_refs") or [])
    assumptions = list(implied.get("assumptions") or [])
    period = str(implied.get("year") or "")
    output: list[ExpectationPoint] = []
    direct_fields = (
        ("gross_margin", "implied_gross_margin", "ratio"),
        ("ebit_margin", "implied_ebit_margin", "ratio"),
        ("ebit", "implied_ebit", "亿元"),
        ("ebitda", "implied_ebitda", "亿元"),
        ("net_margin", "implied_net_margin", "ratio"),
        ("valuation_multiple", "implied_valuation_multiple", "x"),
    )
    for metric, field, unit in direct_fields:
        value = implied.get(field)
        if value is None:
            continue
        point_assumptions = list(assumptions)
        derivation = f"MarketImpliedYear.{field} disclosed by the deterministic reverse-valuation path"
        if metric == "valuation_multiple":
            derivation = "Model-Implied valuation assumption used by the reverse-valuation path; not Market Consensus"
            point_assumptions.append("该倍数仅称为Model-Implied假设，不代表Market Consensus或市场事实。")
        output.append(ExpectationPoint(
            series_type="model_implied", metric=metric, period=period, value=value,
            unit=unit, available=True, evidence_refs=refs, derivation=derivation,
            assumptions=list(dict.fromkeys(point_assumptions)),
        ))

    direct_metrics = {item.metric for item in output}
    if {"gross_margin", "ebit_margin"}.issubset(direct_metrics) or not base_projection:
        return output
    revenue = implied.get("implied_revenue")
    net_profit = implied.get("implied_net_profit")
    if revenue in {None, 0} or net_profit is None:
        return output
    tax_rate = float(base_projection.get("tax_rate") or 0)
    minority_rate = float(base_projection.get("minority_interest_ratio") or 0)
    after_tax_factor = (1 - tax_rate) * (1 - minority_rate)
    if net_profit >= 0:
        if after_tax_factor <= 0:
            return output
        pre_tax = net_profit / after_tax_factor
    else:
        pre_tax = net_profit
    financial_ratio = float(base_projection.get("financial_expense_ratio") or 0)
    other_income_ratio = float(base_projection.get("other_income_ratio") or 0)
    required_ebit = pre_tax + revenue * financial_ratio - revenue * other_income_ratio
    ebit_margin = required_ebit / revenue
    operating_cost_ratio = sum(float(base_projection.get(key) or 0) for key in (
        "rd_expense_ratio", "selling_expense_ratio", "admin_expense_ratio",
        "other_operating_expense_ratio",
    ))
    derived = {
        "ebit_margin": (
            ebit_margin,
            "Required EBIT=(Model-Implied Net Profit/税后归母因子)+财务费用-其他收益；Required EBIT Margin=Required EBIT/Model-Implied Revenue",
        ),
        "gross_margin": (
            ebit_margin + operating_cost_ratio,
            "Required Gross Margin=Required EBIT Margin+Base研发/销售/管理/其他经营费用率",
        ),
    }
    for metric, (value, derivation) in derived.items():
        if metric in direct_metrics:
            continue
        output.append(ExpectationPoint(
            series_type="model_implied", metric=metric, period=period, value=value,
            unit="ratio", available=True, evidence_refs=refs, derivation=derivation,
            assumptions=list(dict.fromkeys(assumptions + [
                f"Base费用率合计={operating_cost_ratio:.6f}", f"Base税率={tax_rate:.6f}",
                f"Base少数股东比例={minority_rate:.6f}",
                "这是Model-Implied under disclosed cost/tax assumptions，不是Market Consensus。",
            ])),
        ))
    return output


def _authoritative_points(
    expectation: ExpectationOutput,
    valuation: ValuationOutput,
    evidence: dict[str, dict[str, Any]],
    analysis_date: date,
) -> tuple[list[ExpectationPoint], bool, list[str]]:
    points: list[ExpectationPoint] = []
    issues: list[str] = []
    valid_consensus = False
    for item in expectation.consensus_estimates:
        gate = evaluate_evidence_gate(item.evidence_refs, evidence)
        valid = bool(
            item.available and item.source_count >= 2 and item.as_of
            and item.as_of <= analysis_date and gate.passed
        )
        if not valid:
            issues.append(
                f"{item.year} Market Consensus unavailable：需要当前有效机构预测、至少两个独立来源且通过Evidence Gate"
            )
            continue
        valid_consensus = True
        for metric in ("revenue", "net_profit"):
            value = getattr(item, metric)
            if value is not None:
                points.append(ExpectationPoint(
                    series_type="market_consensus", metric=metric, period=str(item.year),
                    value=value, unit="亿元", available=True,
                    evidence_refs=list(item.evidence_refs),
                    derivation=f"外部机构一致预期；source_count={item.source_count}；as_of={item.as_of}",
                ))

    path = valuation.reverse_valuation or expectation.price_implied_path or expectation.market_implied_path
    base = next((scenario for scenario in valuation.scenarios if scenario.name == "base"), None)
    base_by_year = {str(item.year): item.model_dump(mode="json") for item in (base.projections if base else [])}
    for item in path:
        raw = item.model_dump(mode="json")
        for metric, field, unit in (
            ("net_profit", "implied_net_profit", "亿元"),
            ("revenue", "implied_revenue", "亿元"),
        ):
            value = raw.get(field)
            if value is not None:
                points.append(ExpectationPoint(
                    series_type="model_implied", metric=metric, period=str(item.year),
                    value=value, unit=unit, available=True,
                    evidence_refs=list(item.evidence_refs), derivation=item.formula,
                    assumptions=list(item.assumptions),
                ))
        points.extend(_implied_operating_points(raw, base_by_year.get(str(item.year))))

    for scenario in valuation.scenarios:
        for projection in scenario.projections:
            ebit_margin = (
                projection.ebit / projection.revenue if projection.revenue not in {None, 0} else None
            )
            net_margin = (
                projection.net_profit / projection.revenue
                if projection.revenue not in {None, 0} else None
            )
            for metric, value, unit in (
                ("revenue", projection.revenue, "亿元"),
                ("gross_margin", projection.gross_margin, "ratio"),
                ("ebit_margin", ebit_margin, "ratio"),
                ("ebit", projection.ebit, "亿元"),
                ("ebitda", projection.ebitda, "亿元"),
                ("net_profit", projection.net_profit, "亿元"),
                ("net_margin", net_margin, "ratio"),
                ("valuation_multiple", projection.valuation_multiple or None, "x"),
            ):
                if value is not None:
                    points.append(ExpectationPoint(
                        series_type="agent_forecast", scenario=scenario.name,
                        metric=metric, period=str(projection.year), value=value, unit=unit,
                        available=True,
                        evidence_refs=list(dict.fromkeys(scenario.evidence_refs + projection.evidence_refs)),
                        derivation="Agent情景假设经Python财务桥计算；不属于Market Consensus或Model-Implied",
                    ))
        if scenario.target_price is not None:
            points.append(ExpectationPoint(
                series_type="agent_forecast", scenario=scenario.name,
                metric="target_price", period=str(scenario.target_year),
                value=scenario.target_price, unit="元/股", available=True,
                evidence_refs=list(scenario.evidence_refs),
                derivation="统一EV→Equity→Target Price估值桥",
            ))
    return points, valid_consensus, issues


def _build_gaps(points: list[ExpectationPoint]) -> list[MispricingGap]:
    benchmarks = {
        (point.series_type, point.metric, point.period): point
        for point in points if point.series_type in {"market_consensus", "model_implied"}
        and point.available and point.value is not None
    }
    gaps: list[MispricingGap] = []
    index = 0
    for agent in points:
        if agent.series_type != "agent_forecast" or agent.metric not in {
            "revenue", "gross_margin", "ebit_margin", "ebit", "ebitda",
            "net_profit", "net_margin", "valuation_multiple",
        } or agent.value is None or agent.scenario is None:
            continue
        for benchmark_type in ("market_consensus", "model_implied"):
            benchmark = benchmarks.get((benchmark_type, agent.metric, agent.period))
            if not benchmark or benchmark.value is None:
                continue
            index += 1
            absolute = agent.value - benchmark.value
            percentage = absolute / abs(benchmark.value) if benchmark.value != 0 else None
            if agent.metric in {"gross_margin", "ebit_margin", "net_margin"}:
                significant = abs(absolute) >= 0.02
            else:
                significant = percentage is not None and abs(percentage) >= 0.10
            if abs(absolute) <= 1e-10:
                direction = "neutral"
            else:
                direction = "market_expectation_too_low" if absolute > 0 else "market_expectation_too_high"
            gaps.append(MispricingGap(
                gap_id=f"G{index}", scenario=agent.scenario, metric=agent.metric,
                period=agent.period, benchmark_type=benchmark_type,
                benchmark_value=benchmark.value, agent_value=agent.value,
                absolute_gap=absolute, gap_percentage=percentage, direction=direction,
                significant=significant,
                formula=(
                    f"Agent {agent.value:.6f} - Benchmark {benchmark.value:.6f} = {absolute:.6f}; "
                    + (f"Gap%={percentage:.6%}" if percentage is not None else "Benchmark=0，Gap%不计算")
                ),
            ))
    return gaps


def _normalized_gap_magnitude(item: MispricingGap) -> float | None:
    if item.absolute_gap is None:
        return None
    if item.metric in {"gross_margin", "ebit_margin", "net_margin"}:
        return abs(item.absolute_gap) / 0.02
    if item.benchmark_value in {None, 0}:
        return None
    return abs(item.absolute_gap) / abs(item.benchmark_value)


def _build_variable_gap_contributions(
    gaps: list[MispricingGap], points: list[ExpectationPoint]
) -> list[VariableGapContribution]:
    point_index = {
        (item.series_type, item.scenario, item.metric, item.period): item for item in points
    }
    candidates = [item for item in gaps if item.scenario == "base"] or list(gaps)
    rows: list[tuple[VariableGapContribution, float]] = []
    for gap in candidates:
        magnitude = _normalized_gap_magnitude(gap)
        benchmark = point_index.get((gap.benchmark_type, None, gap.metric, gap.period))
        agent = point_index.get(("agent_forecast", gap.scenario, gap.metric, gap.period))
        refs = list(dict.fromkeys(
            list(benchmark.evidence_refs if benchmark else [])
            + list(agent.evidence_refs if agent else [])
        ))
        if gap.absolute_gap is None:
            direction = "unavailable"
        elif abs(gap.absolute_gap) <= 1e-10:
            direction = "neutral"
        else:
            direction = "positive" if gap.absolute_gap > 0 else "negative"
        denominator = "2个百分点" if gap.metric in {"gross_margin", "ebit_margin", "net_margin"} else "|Benchmark|"
        rows.append((VariableGapContribution(
            variable=gap.metric, period=gap.period, scenario=gap.scenario,
            benchmark_value=gap.benchmark_value, agent_value=gap.agent_value,
            absolute_gap=gap.absolute_gap, contribution=magnitude,
            direction=direction, available=magnitude is not None,
            formula=f"normalized amplitude=|Agent-Benchmark|/{denominator}",
            evidence_refs=refs,
        ), magnitude if magnitude is not None else -1.0))
    total = sum(magnitude for _, magnitude in rows if magnitude >= 0)
    output: list[VariableGapContribution] = []
    for item, magnitude in sorted(
        rows, key=lambda pair: (pair[1], pair[0].variable, pair[0].period), reverse=True
    ):
        item.contribution_percentage = magnitude / total if magnitude >= 0 and total > 0 else None
        output.append(item)
    return output


def _largest_base_gap(gaps: list[MispricingGap]) -> MispricingGap | None:
    candidates = [item for item in gaps if item.scenario == "base"] or gaps
    if not candidates:
        return None
    def magnitude(item: MispricingGap) -> float:
        if item.metric in {"gross_margin", "ebit_margin"}:
            return abs(item.absolute_gap or 0) / 0.02
        return abs(item.gap_percentage or 0)
    return max(candidates, key=lambda item: (item.significant, magnitude(item)))


def _build_drivers(
    draft: MispricingNarrativeDraft,
    evidence: dict[str, dict[str, Any]],
    catalysts: CatalystOutput,
) -> list[MispricingDriver]:
    known_catalysts = {item.catalyst_id: item for item in catalysts.catalysts}
    drivers: list[MispricingDriver] = []
    used: set[str] = set()
    for index, item in enumerate(draft.drivers, start=1):
        driver_id = item.driver_id.strip() or f"D{index}"
        if driver_id in used:
            driver_id = f"D{index}"
        used.add(driver_id)
        refs = list(dict.fromkeys(item.evidence_refs))
        gate = evaluate_evidence_gate(refs, evidence)
        catalyst_ids = list(dict.fromkeys(cid for cid in item.catalyst_ids if cid in known_catalysts))
        issues: list[str] = []
        if not gate.passed:
            issues.append("Gap Driver未通过Evidence Gate")
        if not catalyst_ids:
            issues.append("未链接现有Catalyst ID")
        for field, value in (
            ("claim", item.claim), ("mechanism", item.mechanism),
            ("verification_metric", item.verification_metric),
            ("time_window", item.time_window), ("rerating_mechanism", item.rerating_mechanism),
        ):
            if not value.strip():
                issues.append(f"缺少{field}")
        drivers.append(MispricingDriver(
            driver_id=driver_id, driver_type=item.driver_type, claim=item.claim,
            mechanism=item.mechanism, evidence_refs=refs,
            counter_evidence_refs=list(dict.fromkeys(item.counter_evidence_refs)),
            catalyst_ids=catalyst_ids, verification_metric=item.verification_metric,
            time_window=item.time_window, rerating_mechanism=item.rerating_mechanism,
            evidence_gate=gate, supported=gate.passed and not any(x.startswith("缺少") for x in issues),
            issues=issues,
        ))
    return drivers


def _build_falsifiers(
    draft: MispricingNarrativeDraft,
    evidence: dict[str, dict[str, Any]],
    drivers: list[MispricingDriver],
) -> list[FalsifiableCondition]:
    driver_ids = {item.driver_id for item in drivers}
    output: list[FalsifiableCondition] = []
    for index, item in enumerate(draft.falsifiable_conditions, start=1):
        condition_id = item.condition_id.strip() or f"F{index}"
        evidence_gate = evaluate_evidence_gate(item.evidence_refs, evidence)
        counter_gate = evaluate_evidence_gate(item.counter_evidence_refs, evidence)
        issues: list[str] = []
        if item.driver_id not in driver_ids:
            issues.append("未链接有效Gap Driver")
        required = {
            "current_judgment": item.current_judgment,
            "verification_metric": item.verification_metric,
            "confirms_if": item.confirms_if,
            "invalidates_if": item.invalidates_if,
            "latest_verification_time": item.latest_verification_time,
        }
        issues.extend(f"缺少{name}" for name, value in required.items() if not value.strip())
        if not item.key_variables:
            issues.append("缺少key_variables")
        complete = not issues
        output.append(FalsifiableCondition(
            condition_id=condition_id, driver_id=item.driver_id,
            current_judgment=item.current_judgment, key_variables=item.key_variables,
            verification_metric=item.verification_metric, confirms_if=item.confirms_if,
            invalidates_if=item.invalidates_if,
            latest_verification_time=item.latest_verification_time,
            current_status=item.current_status,
            evidence_refs=list(dict.fromkeys(item.evidence_refs)),
            counter_evidence_refs=list(dict.fromkeys(item.counter_evidence_refs)),
            evidence_gate=evidence_gate, counter_evidence_gate=counter_gate,
            currently_invalidated=item.current_status == "refuted" and counter_gate.passed,
            complete=complete, issues=issues,
        ))
    return output


def _core_gate_status(valuation: ValuationOutput, research: dict[str, Any]) -> tuple[bool, list[str]]:
    blockers: list[str] = []
    snapshot_report = ((research.get("market_snapshot") or {}).get("consistency_report") or {})
    checks = (
        (snapshot_report.get("valuation_allowed") is True, "Market Snapshot consistency/valuation_allowed未通过"),
        (valuation.pre_valuation_probability_audit.passed, "Pre-Valuation Probability Gate未通过"),
        (valuation.probability_audit.passed, "Probability Audit未通过"),
        (valuation.forecast_reasonableness_audit.passed and valuation.forecast_reasonableness_audit.valuation_allowed,
         "Forecast Reasonableness Gate未通过"),
        (valuation.forecast_reasonableness_audit.margin_expansion_evidence_passed,
         "Margin Expansion Evidence Gate未通过"),
        (valuation.comparable_valuation_evidence_audit.passed,
         "Comparable Valuation Evidence Gate未通过"),
        (valuation.valuation_audit.passed, "Valuation/Expected Return Audit未通过"),
        (valuation.expected_price_return is not None and valuation.annualized_expected_return is not None,
         "Expected Return不可用"),
    )
    blockers.extend(message for passed, message in checks if not passed)
    return not blockers, blockers


def _build_chains(
    maximum_gap: MispricingGap | None,
    drivers: list[MispricingDriver],
    catalysts: CatalystOutput,
) -> list[ExpectationGapChain]:
    catalyst_by_id = {item.catalyst_id: item for item in catalysts.catalysts}
    chains: list[ExpectationGapChain] = []
    if not maximum_gap:
        return chains
    for driver in drivers:
        valid_catalysts = [
            catalyst_by_id[cid] for cid in driver.catalyst_ids
            if cid in catalyst_by_id and catalyst_by_id[cid].evidence_gate.passed
        ]
        issues: list[str] = []
        if not driver.supported:
            issues.append("Driver Evidence不足")
        if not valid_catalysts:
            issues.append("缺少通过Evidence Gate的Catalyst")
        if not driver.verification_metric:
            issues.append("缺少Verification Metric")
        if not driver.time_window:
            issues.append("缺少Time Window")
        if not driver.rerating_mechanism:
            issues.append("缺少Potential Re-rating / De-rating机制")
        chains.append(ExpectationGapChain(
            driver_id=driver.driver_id,
            market_expectation=(
                f"{maximum_gap.benchmark_type} {maximum_gap.period} {maximum_gap.metric}="
                f"{maximum_gap.benchmark_value}"
            ),
            agent_expectation=(
                f"Agent {maximum_gap.scenario} {maximum_gap.period} {maximum_gap.metric}="
                f"{maximum_gap.agent_value}"
            ),
            gap_id=maximum_gap.gap_id, evidence_refs=driver.evidence_refs,
            catalyst_ids=[item.catalyst_id for item in valid_catalysts],
            verification_metric=driver.verification_metric,
            time_window=driver.time_window,
            repricing_mechanism=driver.rerating_mechanism,
            complete=not issues, issues=issues,
        ))
    return chains


def _edge_components(
    maximum_gap: MispricingGap | None,
    drivers: list[MispricingDriver],
    catalysts: CatalystOutput,
    valuation: ValuationOutput,
) -> tuple[list[InvestmentEdgeComponent], float]:
    earnings_gaps = [
        item for item in valuation.forecast_comparison
        if item.agent_forecast_profit is not None and item.model_implied_profit not in {None, 0}
    ]
    earnings_gap = max(
        (abs((item.agent_forecast_profit - item.model_implied_profit) / item.model_implied_profit)
         for item in earnings_gaps), default=None,
    )
    if maximum_gap and maximum_gap.metric == "net_profit" and maximum_gap.gap_percentage is not None:
        earnings_gap = max(earnings_gap or 0, abs(maximum_gap.gap_percentage))
    earnings = InvestmentEdgeComponent(
        name="market_vs_agent_earnings_gap", available=earnings_gap is not None,
        score=_clamp((earnings_gap or 0) * 200) if earnings_gap is not None else None,
        formula="min(100, |Agent Earnings-Benchmark Earnings|/|Benchmark Earnings|×200)",
        inputs={"maximum_absolute_gap_ratio": earnings_gap},
        issues=[] if earnings_gap is not None else ["同年盈利Benchmark或Agent Forecast不可用"],
    )

    supported = sum(1 for item in drivers if item.supported)
    evidence_score = supported / len(drivers) * 100 if drivers else None
    evidence_component = InvestmentEdgeComponent(
        name="evidence_strength", available=evidence_score is not None,
        score=evidence_score,
        formula="通过Evidence Gate且结构完整的Gap Drivers数量/全部Gap Drivers数量×100",
        inputs={"supported_drivers": supported, "all_drivers": len(drivers)},
        issues=[] if drivers else ["Mispricing Agent未提供Gap Driver"],
    )

    forecast = valuation.forecast_reasonableness_audit
    forecast_flags = {
        "historical_anchor": forecast.historical_anchor_passed,
        "margin_bridge": forecast.margin_bridge_passed,
        "scenario_separation": forecast.scenario_separation_passed,
        "margin_expansion_evidence": forecast.margin_expansion_evidence_passed,
    }
    forecast_score = sum(25 for passed in forecast_flags.values() if passed)
    forecast_component = InvestmentEdgeComponent(
        name="forecast_confidence", available=True, score=forecast_score,
        formula="Historical Anchor、Margin Bridge、Scenario Separation、Margin Expansion Evidence各25分",
        inputs=forecast_flags,
    )

    linked = {cid for driver in drivers for cid in driver.catalyst_ids}
    timeframe_scores = {"30d": 100.0, "quarter": 75.0, "half_year": 50.0, "one_year": 25.0}
    valid_catalysts = [
        item for item in catalysts.catalysts if item.catalyst_id in linked and item.evidence_gate.passed
    ]
    catalyst_score = max((timeframe_scores[item.timeframe] for item in valid_catalysts), default=None)
    catalyst_component = InvestmentEdgeComponent(
        name="catalyst_proximity", available=catalyst_score is not None,
        score=catalyst_score,
        formula="通过Evidence Gate且链接Gap Driver的最近Catalyst：30d=100、quarter=75、half_year=50、one_year=25",
        inputs={"valid_catalysts": [item.catalyst_id for item in valid_catalysts]},
        issues=[] if valid_catalysts else ["无通过Evidence Gate并链接Gap Driver的Catalyst"],
    )

    required = next((item.required_return for item in valuation.reverse_valuation), 0.10)
    annualized = valuation.annualized_expected_return
    valuation_available = valuation.valuation_audit.passed and annualized is not None
    valuation_score = _clamp(50 + (annualized - required) * 250) if valuation_available else None
    valuation_component = InvestmentEdgeComponent(
        name="valuation_gap", available=valuation_available, score=valuation_score,
        formula="clamp(0,100,50+(Annualized Expected Return-Required Return)×250)",
        inputs={"annualized_expected_return": annualized, "required_return": required},
        issues=[] if valuation_available else ["Valuation/Expected Return Gate未通过"],
    )

    bear = next((item for item in valuation.scenarios if item.name == "bear"), None)
    p_bear = bear.probability if bear else None
    negative_return = sum(
        item.probability for item in valuation.scenarios
        if item.target_price is not None and valuation.current_price is not None
        and item.target_price < valuation.current_price
    ) if valuation.scenarios else None
    downside_available = p_bear is not None and negative_return is not None
    downside_score = _clamp((1 - max(p_bear or 0, negative_return or 0)) * 100) if downside_available else None
    downside = InvestmentEdgeComponent(
        name="downside_risk", available=downside_available, score=downside_score,
        formula="(1-max(P(Bear), Probability of Negative Return))×100；不使用风险独立并集",
        inputs={"p_bear": p_bear, "probability_of_negative_return": negative_return},
        issues=[] if downside_available else ["P(Bear)或负回报概率不可用"],
    )
    return [earnings, evidence_component, forecast_component, catalyst_component, valuation_component, downside], negative_return or 0


def _classification(
    maximum_gap: MispricingGap | None,
    valuation: ValuationOutput,
    financial: FinancialOutput,
) -> MispricingClassification:
    base = next((item for item in valuation.scenarios if item.name == "base"), None)
    latest_profit = next(
        (item.net_profit for item in reversed(financial.historical) if item.net_profit is not None), None
    )
    improving = bool(base and base.net_profit is not None and latest_profit is not None and base.net_profit > latest_profit)
    cheap = bool(
        valuation.valuation_audit.passed and base and base.price_return is not None and base.price_return > 0
    )
    too_low = bool(maximum_gap and maximum_gap.significant and maximum_gap.direction == "market_expectation_too_low")
    too_high = bool(maximum_gap and maximum_gap.significant and maximum_gap.direction == "market_expectation_too_high")
    return MispricingClassification(
        price_cheap=cheap,
        market_expectation_too_high=too_high,
        market_expectation_too_low=too_low,
        fundamentals_improving=improving,
        valuation_rerating=False,
        basis={
            "price_cheap": "Valuation Audit通过且Base目标价高于当前价" if cheap else "Base目标价/估值Gate未证明价格便宜",
            "market_expectation_too_high": "最大Base Gap为Agent低于Benchmark" if too_high else "最大Base Gap未显示市场预期过高",
            "market_expectation_too_low": "最大Base Gap为Agent高于Benchmark" if too_low else "最大Base Gap未显示市场预期过低",
            "fundamentals_improving": (
                f"Base终值净利润{base.net_profit if base else None}高于历史基准{latest_profit}"
                if improving else "Base终值净利润未高于可用历史基准"
            ),
            "valuation_rerating": "缺少跨期同口径市场估值倍数证据，不把未来倍数假设误判为‘正在重估’",
        },
    )


def _unique_refs(*groups: list[str]) -> list[str]:
    return list(dict.fromkeys(ref for group in groups for ref in group if ref))


def _build_catalyst_verification_audit(
    catalysts: CatalystOutput,
    valuation: ValuationOutput,
    evidence: dict[str, dict[str, Any]],
) -> CatalystVerificationAudit:
    trigger_ids = {
        trigger.trigger_id
        for scenario in valuation.scenarios
        for trigger in scenario.business_triggers
        if trigger.trigger_id
    }
    logical_assumption_ids = {
        node.node_id for node in valuation.financial_tree.nodes if node.node_id
    }
    known_assumption_ids = trigger_ids | logical_assumption_ids
    valid_scenarios = {"bear", "base", "bull"}
    items: list[CatalystVerificationItem] = []
    audit_issues: list[str] = []
    unverified: list[str] = []
    for catalyst in catalysts.catalysts:
        issues: list[str] = []
        gate = evaluate_evidence_gate(catalyst.evidence_refs, evidence)
        observable = bool(catalyst.observable_metric.strip())
        time_bound = bool(
            _as_date(catalyst.expected_date)
            or catalyst.timeframe in {"30d", "quarter", "half_year", "one_year"}
        )
        assumption_ids = list(dict.fromkeys(catalyst.verifies_assumption_ids))
        catalyst_known_ids = known_assumption_ids | set(catalyst.linked_story_nodes)
        invalid_ids = [item for item in assumption_ids if item not in catalyst_known_ids]
        assumption_ids_valid = bool(assumption_ids) and not invalid_ids
        affected_variables = [
            item for item in dict.fromkeys(catalyst.affected_financial_variables) if item.strip()
        ]
        valid_effects = {
            key: value for key, value in catalyst.scenario_probability_effects.items()
            if key.lower() in valid_scenarios
            and value is not None
            and not isinstance(value, bool)
            and float("-inf") < float(value) < float("inf")
        }
        probability_effects_defined = bool(valid_effects)
        if not time_bound:
            issues.append("expected_date与timeframe均不可执行")
        if not observable:
            issues.append("缺少observable_metric")
        if not assumption_ids:
            issues.append("缺少verifies_assumption_ids")
        elif invalid_ids:
            issues.append("verifies_assumption_ids未匹配Business Trigger或Financial Logic Node：" + ", ".join(invalid_ids))
        if not affected_variables:
            issues.append("缺少affected_financial_variables")
        if not catalyst.favorable_result.strip():
            issues.append("缺少favorable_result")
        if not catalyst.adverse_result.strip():
            issues.append("缺少adverse_result")
        if not probability_effects_defined:
            issues.append("scenario_probability_effects缺少有效Bear/Base/Bull key/value")
        if not gate.passed:
            issues.append("Catalyst Evidence Gate未通过：" + gate.reason)
        passed = bool(
            time_bound and observable and assumption_ids_valid and affected_variables
            and catalyst.favorable_result.strip() and catalyst.adverse_result.strip()
            and probability_effects_defined and gate.passed
        )
        if not passed:
            unverified.append(catalyst.catalyst_id)
            audit_issues.append(f"{catalyst.catalyst_id or '未命名Catalyst'}：{'；'.join(issues)}")
        items.append(CatalystVerificationItem(
            catalyst_id=catalyst.catalyst_id,
            verifies_assumption_ids=assumption_ids,
            affected_financial_variables=affected_variables,
            favorable_result=catalyst.favorable_result,
            adverse_result=catalyst.adverse_result,
            scenario_probability_effects=dict(catalyst.scenario_probability_effects),
            observable=observable,
            time_bound=time_bound,
            probability_effects_defined=probability_effects_defined,
            passed=passed,
            status="verified" if passed else "blocked",
            issues=issues,
        ))
    passed = bool(items) and all(item.passed for item in items)
    if not items:
        audit_issues.append("没有可审计Catalyst")
    return CatalystVerificationAudit(
        passed=passed,
        status="available" if passed else ("unavailable" if not items else "blocked"),
        items=items,
        unverified_catalyst_ids=list(dict.fromkeys(unverified)),
        issues=list(dict.fromkeys(audit_issues)),
    )


def _logic_step(
    step_id: str,
    premise: str,
    conclusion: str,
    mechanism: str,
    refs: list[str],
    evidence: dict[str, dict[str, Any]],
    *,
    depends_on: list[str] | None = None,
    assumption_ids: list[str] | None = None,
    variables: list[str] | None = None,
    requirements_passed: bool = True,
    issues: list[str] | None = None,
) -> InvestmentLogicStep:
    refs = list(dict.fromkeys(refs))
    gate = evaluate_evidence_gate(refs, evidence)
    row_issues = list(issues or [])
    if not premise.strip() or not conclusion.strip() or not mechanism.strip():
        row_issues.append("步骤缺少结构化premise/conclusion/mechanism")
    if not refs:
        row_issues.append("步骤缺少真实Evidence Ref")
    elif not gate.passed:
        row_issues.append("Evidence Gate未通过：" + gate.reason)
    if not requirements_passed:
        row_issues.append("步骤前置确定性条件未通过")
    passed = bool(
        premise.strip() and conclusion.strip() and mechanism.strip()
        and refs and gate.passed and requirements_passed and not row_issues
    )
    return InvestmentLogicStep(
        step_id=step_id,
        premise=premise,
        conclusion=conclusion,
        mechanism=mechanism,
        depends_on=depends_on or [],
        assumption_ids=list(dict.fromkeys(assumption_ids or [])),
        affected_financial_variables=list(dict.fromkeys(variables or [])),
        evidence_refs=refs,
        evidence_gate=gate,
        passed=passed,
        status="verified" if passed else "blocked",
        issues=list(dict.fromkeys(row_issues)),
    )


def _build_investment_logic_chain(
    industry: IndustryOutput | None,
    company: CompanyOutput | None,
    valuation: ValuationOutput,
    bear: BearOutput,
    maximum_gap: MispricingGap | None,
    points: list[ExpectationPoint],
    catalysts: CatalystOutput,
    catalyst_audit: CatalystVerificationAudit,
    evidence: dict[str, dict[str, Any]],
    *,
    pre_verdict_state: bool,
) -> InvestmentLogicChainAudit:
    steps: list[InvestmentLogicStep] = []
    industry_changes = (
        industry.growth_drivers + industry.technology_trends + industry.policy_drivers
        + industry.nonlinear_profit_drivers + industry.industry_risks
        if industry else []
    )
    step = _logic_step(
        "L1-industry-change",
        industry.industry if industry else "",
        "；".join(industry_changes),
        "行业结构化变化形成公司经营环境变化",
        list(industry.evidence_refs) if industry else [],
        evidence,
        requirements_passed=bool(industry and industry_changes),
        issues=[] if industry else ["缺少IndustryOutput上下文"],
    )
    steps.append(step)

    valid_mapping = None
    if company:
        for mapping in company.opportunity_mapping:
            mapping_gate = evaluate_evidence_gate(mapping.evidence_refs, evidence)
            if (
                mapping.validation_status == "validated"
                and mapping.industry_change.strip()
                and mapping.company_capability.strip()
                and mapping.capture_mechanism.strip()
                and mapping_gate.passed
            ):
                valid_mapping = mapping
                break
    mapping_refs = list(valid_mapping.evidence_refs) if valid_mapping else []
    step = _logic_step(
        "L2-company-impact",
        valid_mapping.industry_change if valid_mapping else "",
        valid_mapping.company_capability if valid_mapping else "",
        valid_mapping.capture_mechanism if valid_mapping else "",
        mapping_refs,
        evidence,
        depends_on=[steps[-1].step_id],
        assumption_ids=[valid_mapping.mapping_id] if valid_mapping else [],
        requirements_passed=steps[-1].passed and valid_mapping is not None,
        issues=[] if company else ["缺少CompanyOutput上下文"],
    )
    steps.append(step)

    triggers = [
        trigger for scenario in valuation.scenarios for trigger in scenario.business_triggers
        if trigger.trigger_id and trigger.business_change.strip()
        and trigger.financial_variable.strip() and trigger.mechanism.strip()
    ]
    trigger_refs = _unique_refs(*(list(item.evidence_refs) for item in triggers)) if triggers else []
    trigger_variables = list(dict.fromkeys(item.financial_variable for item in triggers))
    step = _logic_step(
        "L3-financial-variables",
        "；".join(item.business_change for item in triggers),
        "；".join(trigger_variables),
        "；".join(item.mechanism for item in triggers),
        trigger_refs,
        evidence,
        depends_on=[steps[-1].step_id],
        assumption_ids=[item.trigger_id for item in triggers],
        variables=trigger_variables,
        requirements_passed=steps[-1].passed and valuation.scenario_trigger_audit.passed and bool(triggers),
    )
    steps.append(step)

    profit_nodes = [
        node for node in valuation.financial_tree.nodes
        if any(token in (node.metric + " " + node.label).lower()
               for token in ("profit", "ebit", "利润", "净利"))
    ]
    projection_refs = _unique_refs(*(
        list(item.evidence_refs)
        + [
            ref for refs in item.assumption_evidence_refs.values()
            for ref in refs
        ]
        for scenario in valuation.scenarios for item in scenario.projections
    )) if valuation.scenarios else []
    profit_refs = _unique_refs(trigger_refs, projection_refs)
    step = _logic_step(
        "L4-profit",
        "；".join(item.label for item in profit_nodes),
        "；".join(
            f"{item.metric or item.label}={item.value if item.value is not None else '未量化'}"
            for item in profit_nodes
        ),
        valuation.financial_tree.summary,
        profit_refs,
        evidence,
        depends_on=[steps[-1].step_id],
        assumption_ids=[item.node_id for item in profit_nodes],
        variables=[item.metric for item in profit_nodes if item.metric],
        requirements_passed=steps[-1].passed and bool(profit_nodes),
    )
    steps.append(step)

    gap_refs: list[str] = []
    if maximum_gap:
        gap_refs = _unique_refs(*(
            list(item.evidence_refs) for item in points
            if item.metric == maximum_gap.metric and item.period == maximum_gap.period
            and (
                (item.series_type == maximum_gap.benchmark_type and item.scenario is None)
                or (item.series_type == "agent_forecast" and item.scenario == maximum_gap.scenario)
            )
        ))
    gap_premise = (
        f"{maximum_gap.benchmark_type} {maximum_gap.period} {maximum_gap.metric}={maximum_gap.benchmark_value}"
        if maximum_gap else ""
    )
    gap_conclusion = (
        f"Agent {maximum_gap.scenario}={maximum_gap.agent_value}; gap={maximum_gap.absolute_gap}"
        if maximum_gap else ""
    )
    step = _logic_step(
        "L5-expectation-gap", gap_premise, gap_conclusion,
        maximum_gap.formula if maximum_gap else "", gap_refs, evidence,
        depends_on=[steps[-1].step_id],
        variables=[maximum_gap.metric] if maximum_gap else [],
        requirements_passed=steps[-1].passed and bool(maximum_gap and maximum_gap.significant),
    )
    steps.append(step)

    verified_ids = {item.catalyst_id for item in catalyst_audit.items if item.passed}
    verified_catalysts = [item for item in catalysts.catalysts if item.catalyst_id in verified_ids]
    catalyst_refs = _unique_refs(*(list(item.evidence_refs) for item in verified_catalysts)) if verified_catalysts else []
    step = _logic_step(
        "L6-catalyst",
        "；".join(item.event for item in verified_catalysts),
        "；".join(item.favorable_result + " / " + item.adverse_result for item in verified_catalysts),
        "；".join(item.repricing_mechanism for item in verified_catalysts),
        catalyst_refs,
        evidence,
        depends_on=[steps[-1].step_id],
        assumption_ids=[aid for item in verified_catalysts for aid in item.verifies_assumption_ids],
        variables=[var for item in verified_catalysts for var in item.affected_financial_variables],
        requirements_passed=steps[-1].passed and catalyst_audit.passed and bool(verified_catalysts),
    )
    steps.append(step)

    selected_valuation_refs: list[str] = []
    for scenario in valuation.scenarios:
        if not scenario.projections:
            continue
        selected = next(
            (candidate for candidate in scenario.projections[-1].valuation_candidates if candidate.selected),
            None,
        )
        if selected:
            selected_valuation_refs.extend(selected.evidence_refs)
            selected_valuation_refs.extend(
                item.evidence_ref for item in selected.comparable_evidence
                if item.valid and item.evidence_ref
            )
    price_refs = _unique_refs(*(
        list(item.evidence_refs) for item in valuation.reverse_valuation
    )) if valuation.reverse_valuation else []
    valuation_refs = _unique_refs(selected_valuation_refs, price_refs)
    selected_methods = [item.valuation_method for item in valuation.scenarios if item.valuation_method]
    step = _logic_step(
        "L7-valuation",
        "；".join(selected_methods),
        f"probability_weighted_target_price={valuation.probability_weighted_target_price}",
        valuation.financial_bridge_method,
        valuation_refs,
        evidence,
        depends_on=[steps[-1].step_id],
        requirements_passed=steps[-1].passed and valuation.valuation_audit.passed
        and valuation.double_counting_audit.passed,
    )
    steps.append(step)

    target_text = "；".join(
        f"{item.name} target={item.target_price}" for item in valuation.scenarios
        if item.target_price is not None
    )
    step = _logic_step(
        "L8-share-price",
        f"current_price={valuation.current_price}" if valuation.current_price is not None else "",
        target_text,
        "目标价与当前价经Risk/Reward Audit转换为可比回报",
        valuation_refs,
        evidence,
        depends_on=[steps[-1].step_id],
        requirements_passed=steps[-1].passed and valuation.current_price is not None
        and valuation.risk_reward_audit.passed,
    )
    steps.append(step)

    risk_refs = _unique_refs(*(list(item.evidence_refs) for item in bear.risks)) if bear.risks else []
    step = _logic_step(
        "L9-risk",
        "；".join(item.risk for item in bear.risks),
        f"downside_risk_probability={bear.downside_risk_probability}",
        "Bear结构化风险与Risk/Reward Audit约束投资结论",
        risk_refs,
        evidence,
        depends_on=[steps[-1].step_id],
        requirements_passed=steps[-1].passed and bool(bear.risks)
        and valuation.risk_reward_audit.passed,
    )
    steps.append(step)

    conclusion_refs = _unique_refs(*(item.evidence_refs for item in steps))
    step = _logic_step(
        "L10-investment-conclusion",
        "确定性Gate、Valuation Audit、Expectation Gap与Risk/Reward前置状态",
        "前置状态允许形成投资结论" if pre_verdict_state else "前置状态阻断投资结论",
        "不读取尚未构造的Mispricing Verdict；仅聚合前九步及确定性前置Gate",
        conclusion_refs,
        evidence,
        depends_on=[steps[-1].step_id],
        requirements_passed=steps[-1].passed and pre_verdict_state,
    )
    steps.append(step)

    failed = [item.step_id for item in steps if not item.passed]
    passed = len(steps) == 10 and not failed
    return InvestmentLogicChainAudit(
        passed=passed,
        status="available" if passed else "blocked",
        steps=steps,
        complete=len(steps) == 10 and all(item.premise and item.conclusion and item.mechanism for item in steps),
        causal_links_verified=passed and all(
            index == 0 or item.depends_on == [steps[index - 1].step_id]
            for index, item in enumerate(steps)
        ),
        failed_step_ids=failed,
        issues=[f"{item.step_id}：{'；'.join(item.issues)}" for item in steps if item.issues],
    )


def _build_tracking_outputs(
    falsifiers: list[FalsifiableCondition],
    catalysts: CatalystOutput,
    catalyst_audit: CatalystVerificationAudit,
    valuation: ValuationOutput,
) -> tuple[list[TrackableInvestmentCondition], FutureValidationChecklist]:
    conditions: list[TrackableInvestmentCondition] = []
    checklist: list[FutureValidationItem] = []
    for item in falsifiers:
        structured = bool(
            item.verification_metric.strip() and item.latest_verification_time.strip()
            and item.confirms_if.strip() and item.invalidates_if.strip()
        )
        status = (
            "invalidated" if structured and item.currently_invalidated else
            "supporting" if structured and item.current_status == "supporting" else
            "unverified" if structured else "unavailable"
        )
        conditions.append(TrackableInvestmentCondition(
            condition_id=item.condition_id,
            description=item.current_judgment,
            metric=item.verification_metric,
            deadline=item.latest_verification_time,
            confirms_if=item.confirms_if,
            invalidates_if=item.invalidates_if,
            evidence_refs=list(item.evidence_refs),
            status=status,
        ))
        checklist.append(FutureValidationItem(
            validation_id=f"validation-{item.condition_id}",
            condition_id=item.condition_id,
            question=item.current_judgment,
            metric=item.verification_metric,
            data_source=", ".join(item.evidence_refs),
            expected_date=item.latest_verification_time,
            favorable_result=item.confirms_if,
            adverse_result=item.invalidates_if,
            evidence_refs=list(item.evidence_refs),
            status="pending" if structured else "unavailable",
        ))

    audit_by_id = {item.catalyst_id: item for item in catalyst_audit.items}
    for item in catalysts.catalysts:
        audit = audit_by_id.get(item.catalyst_id)
        deadline = item.expected_date.strip() or item.timeframe
        structured = bool(
            deadline and item.observable_metric.strip() and item.favorable_result.strip()
            and item.adverse_result.strip() and audit and audit.passed
        )
        conditions.append(TrackableInvestmentCondition(
            condition_id=item.catalyst_id,
            description=item.event,
            metric=item.observable_metric,
            deadline=deadline,
            confirms_if=item.favorable_result,
            invalidates_if=item.adverse_result,
            evidence_refs=list(item.evidence_refs),
            status="unverified" if structured else "unavailable",
        ))
        checklist.append(FutureValidationItem(
            validation_id=f"validation-{item.catalyst_id}",
            condition_id=item.catalyst_id,
            question=(
                item.event + "；验证假设：" + ", ".join(item.verifies_assumption_ids)
                if item.verifies_assumption_ids else item.event
            ),
            metric=item.observable_metric,
            data_source=", ".join(item.evidence_refs),
            expected_date=deadline,
            favorable_result=item.favorable_result,
            adverse_result=item.adverse_result,
            evidence_refs=list(item.evidence_refs),
            status="pending" if structured else "unavailable",
        ))

    for scenario in valuation.scenarios:
        for item in scenario.business_triggers:
            condition_id = item.trigger_id
            conditions.append(TrackableInvestmentCondition(
                condition_id=condition_id,
                description=item.business_change,
                metric=item.financial_variable,
                evidence_refs=list(item.evidence_refs),
                status="unavailable",
            ))
            checklist.append(FutureValidationItem(
                validation_id=f"validation-{condition_id}",
                condition_id=condition_id,
                question=item.business_change,
                metric=item.financial_variable,
                data_source=", ".join(item.evidence_refs),
                evidence_refs=list(item.evidence_refs),
                status="unavailable",
            ))

    available_dates = [item.expected_date for item in checklist if item.status == "pending" and item.expected_date]
    complete = bool(checklist) and all(item.status not in {"unavailable", "blocked"} for item in checklist)
    issues = [] if complete else ["部分Falsifier/Catalyst/Business Trigger缺少结构化时间、指标或利好/利空结果，保持unavailable"]
    return conditions, FutureValidationChecklist(
        status="available" if complete else ("unavailable" if not checklist else "blocked"),
        items=checklist,
        complete=complete,
        next_validation_date=min(available_dates) if available_dates else "",
        issues=issues,
    )


def calculate_mispricing_output(
    narrative: MispricingNarrativeDraft | dict[str, Any] | None,
    expectation: ExpectationOutput | dict[str, Any],
    valuation: ValuationOutput | dict[str, Any],
    catalyst: CatalystOutput | dict[str, Any],
    bear: BearOutput | dict[str, Any],
    research: dict[str, Any],
    financial: FinancialOutput | dict[str, Any],
    industry: IndustryOutput | dict[str, Any] | None = None,
    company: CompanyOutput | dict[str, Any] | None = None,
    *,
    agent_output_available: bool = True,
) -> MispricingOutput:
    draft = MispricingNarrativeDraft.model_validate(_dict(narrative))
    expectation_model = ExpectationOutput.model_validate(_dict(expectation))
    valuation_model = ValuationOutput.model_validate(_dict(valuation))
    catalyst_model = CatalystOutput.model_validate(_dict(catalyst))
    bear_model = BearOutput.model_validate(_dict(bear))
    financial_model = FinancialOutput.model_validate(_dict(financial))
    industry_model = IndustryOutput.model_validate(_dict(industry)) if industry is not None else None
    company_model = CompanyOutput.model_validate(_dict(company)) if company is not None else None
    research_data = _dict(research)
    evidence = {
        item.get("id", ""): item for item in research_data.get("evidence", []) if item.get("id")
    }
    analysis_date = _as_date(
        next((item.get("analysis_date") for item in evidence.values() if item.get("analysis_date")), None)
    ) or date.today()

    points, consensus_available, consensus_issues = _authoritative_points(
        expectation_model, valuation_model, evidence, analysis_date
    )
    if (
        not consensus_available and draft.market_bet
        and not draft.market_bet.startswith("可靠Market Consensus unavailable")
    ):
        draft.market_bet = (
            "可靠Market Consensus unavailable；以下仅为Model-Implied条件叙事，不代表市场共识或已Price In："
            + draft.market_bet
        )
    gaps = _build_gaps(points)
    maximum_gap = _largest_base_gap(gaps)
    drivers = _build_drivers(draft, evidence, catalyst_model) if agent_output_available else []
    falsifiers = _build_falsifiers(draft, evidence, drivers) if agent_output_available else []
    chains = _build_chains(maximum_gap, drivers, catalyst_model)
    core_passed, core_blockers = _core_gate_status(valuation_model, research_data)
    components, negative_return = _edge_components(maximum_gap, drivers, catalyst_model, valuation_model)

    supported_drivers = [item for item in drivers if item.supported]
    catalyst_passed = any(chain.catalyst_ids for chain in chains)
    falsifier_defined = bool(supported_drivers) and all(
        any(item.driver_id == driver.driver_id and item.complete for item in falsifiers)
        for driver in supported_drivers
    )
    complete_chain = any(item.complete for item in chains)
    significant_gap = bool(maximum_gap and maximum_gap.significant)
    benchmark_available = any(
        item.series_type == "model_implied" and item.available for item in points
    ) or consensus_available
    authoritative_separated = bool(
        any(item.series_type == "agent_forecast" for item in points)
        and any(item.series_type == "model_implied" for item in points)
        and all(item.series_type != "market_consensus" or consensus_available for item in points)
    )
    variable_gap_contributions = _build_variable_gap_contributions(gaps, points)
    largest_gap_variable = next(
        (
            item.variable for item in variable_gap_contributions
            if item.available and item.contribution is not None
        ),
        "",
    )
    catalyst_verification_audit = _build_catalyst_verification_audit(
        catalyst_model, valuation_model, evidence
    )
    trackable_conditions, future_validation_checklist = _build_tracking_outputs(
        falsifiers, catalyst_model, catalyst_verification_audit, valuation_model
    )
    pre_verdict_state = bool(
        agent_output_available and core_passed and authoritative_separated
        and benchmark_available and significant_gap and supported_drivers
        and catalyst_passed and falsifier_defined and complete_chain
        and valuation_model.scenario_trigger_audit.passed
        and catalyst_verification_audit.passed
        and valuation_model.double_counting_audit.passed
        and valuation_model.valuation_audit.passed
        and valuation_model.risk_reward_audit.passed
    )
    investment_logic_chain = _build_investment_logic_chain(
        industry_model, company_model, valuation_model, bear_model, maximum_gap, points,
        catalyst_model, catalyst_verification_audit, evidence,
        pre_verdict_state=pre_verdict_state,
    )
    evidence_status_layer = build_evidence_status_layer(
        valuation_model, catalyst_model, drivers, investment_logic_chain, evidence,
        falsifiers=falsifiers,
    )
    evidence_acquisition_plan = build_evidence_acquisition_plan(evidence_status_layer)

    gate_issues = list(core_blockers)
    if not agent_output_available:
        gate_issues.append("历史分析没有Mispricing Agent结构化输出；禁止回填或伪造Investment Edge")
    if not benchmark_available:
        gate_issues.append("Market Consensus与Model-Implied benchmark均不可用")
    if not significant_gap:
        gate_issues.append("未发现达到阈值的Base Expectation Gap")
    if not supported_drivers:
        gate_issues.append("没有通过Evidence Gate的Gap Driver")
    if not catalyst_passed:
        gate_issues.append("没有通过Evidence Gate且链接Gap Driver的Catalyst")
    if not falsifier_defined:
        gate_issues.append("核心Gap Driver缺少完整可证伪条件")
    if not complete_chain:
        gate_issues.append("Gap→Driver→Evidence→Catalyst→Verification→Time→Re-rating链不完整")
    if not valuation_model.scenario_trigger_audit.passed:
        gate_issues.append("Scenario Trigger Audit未通过")
        gate_issues.extend(valuation_model.scenario_trigger_audit.issues)
    if not catalyst_verification_audit.passed:
        gate_issues.append("Catalyst Verification Audit未通过")
        gate_issues.extend(catalyst_verification_audit.issues)
    if not investment_logic_chain.passed:
        gate_issues.append("Investment Logic Chain Audit未通过")
        gate_issues.extend(investment_logic_chain.issues)
    if not valuation_model.double_counting_audit.passed:
        gate_issues.append("Valuation Double Counting Audit未通过")
        gate_issues.extend(valuation_model.double_counting_audit.issues)
    if not evidence_status_layer.actionable_edge_allowed:
        gate_issues.append(evidence_status_layer.one_sentence_conclusion)
        if evidence_status_layer.has_refuted_assumption:
            gate_issues.append("统一证据状态：关键假设存在有效反证")
        elif evidence_status_layer.has_conflicting_evidence:
            gate_issues.append("统一证据状态：关键假设的高质量证据互相矛盾")
        elif evidence_status_layer.has_logic_error:
            gate_issues.append("统一证据状态：存在投资逻辑或量化传导错误")
        else:
            gate_issues.append("统一证据状态：关键假设证据不足，暂不能判断，不等于已证伪")

    all_components = all(item.available and item.score is not None for item in components)
    edge_available = bool(
        agent_output_available and core_passed and authoritative_separated
        and benchmark_available and significant_gap and supported_drivers
        and catalyst_passed and falsifier_defined and complete_chain and all_components
        and valuation_model.scenario_trigger_audit.passed
        and catalyst_verification_audit.passed
        and investment_logic_chain.passed
        and valuation_model.double_counting_audit.passed
        and evidence_status_layer.actionable_edge_allowed
    )
    total = sum(item.score or 0 for item in components) / len(components) if edge_available else None
    edge = InvestmentEdgeScore(
        available=edge_available, total_score=total, components=components,
        blockers=[] if edge_available else list(dict.fromkeys(gate_issues)),
    )
    gate = MispricingGate(
        passed=edge_available, agent_output_available=agent_output_available,
        authoritative_series_separated=authoritative_separated,
        consensus_gate_passed=consensus_available,
        benchmark_available=benchmark_available,
        existing_gates_passed=core_passed,
        significant_gap_found=significant_gap,
        driver_evidence_passed=bool(supported_drivers),
        catalyst_gate_passed=catalyst_passed,
        falsifier_defined=falsifier_defined, complete_chain=complete_chain,
        scenario_trigger_gate_passed=valuation_model.scenario_trigger_audit.passed,
        catalyst_verification_passed=catalyst_verification_audit.passed,
        investment_logic_chain_passed=investment_logic_chain.passed,
        double_counting_passed=valuation_model.double_counting_audit.passed,
        unified_evidence_status_passed=evidence_status_layer.actionable_edge_allowed,
        issues=list(dict.fromkeys(gate_issues + consensus_issues)),
    )

    invalidated = next((item for item in falsifiers if item.currently_invalidated), None)
    primary_benchmark = (
        "market_consensus" if consensus_available else
        ("model_implied" if benchmark_available else "unavailable")
    )
    primary_chain = next((item for item in chains if item.complete), chains[0] if chains else None)
    primary_falsifier = next(
        (item for item in falsifiers if maximum_gap and item.driver_id in {chain.driver_id for chain in chains}),
        falsifiers[0] if falsifiers else None,
    )
    if invalidated:
        status, investable, blocked = "INVALIDATED", False, True
        summary = f"结构化反证已触发：{invalidated.invalidates_if}"
    elif not core_passed or not benchmark_available or not significant_gap:
        status, investable, blocked = "NO EDGE", False, True
        summary = "关键数据/概率/预测/估值Gate失败，或未发现显著预期差；Investment Edge保持null。"
    elif not consensus_available:
        status, investable, blocked = "WATCH", False, not edge_available
        summary = "存在Model-Implied与Agent Forecast差异，但可靠Market Consensus unavailable，Verdict最高为WATCH。"
    elif not edge_available:
        status, investable, blocked = "WATCH", False, True
        summary = "预期差存在，但Driver证据、Catalyst、可证伪条件或完整重定价链尚未全部通过。"
    else:
        score_map = {item.name: item.score or 0 for item in components}
        actionable = bool(
            total is not None and total >= 75
            and score_map["market_vs_agent_earnings_gap"] >= 20
            and score_map["evidence_strength"] >= 75
            and score_map["catalyst_proximity"] >= 50
            and score_map["valuation_gap"] >= 50
            and score_map["downside_risk"] >= 50
            and negative_return <= 0.5
        )
        emerging = bool(
            total is not None and total >= 55
            and score_map["evidence_strength"] >= 50
        )
        if actionable:
            status, investable, blocked = "ACTIONABLE EDGE", True, False
            summary = "显著预期差、证据、催化剂、估值与风险收益均通过确定性阈值。"
        elif emerging:
            status, investable, blocked = "EMERGING EDGE", False, False
            summary = "预期差和早期验证链已形成，但尚未达到严格可投资阈值。"
        else:
            status, investable, blocked = "WATCH", False, False
            summary = "完整链已建立，但结构化Edge总分或关键分项尚未达到Emerging阈值。"

    maximum_text = ""
    if maximum_gap:
        maximum_text = (
            f"{maximum_gap.period} {maximum_gap.metric}: Agent {maximum_gap.agent_value} vs "
            f"{maximum_gap.benchmark_type} {maximum_gap.benchmark_value}; "
            f"Gap {maximum_gap.absolute_gap:+.4f}"
            + (f" ({maximum_gap.gap_percentage:+.2%})" if maximum_gap.gap_percentage is not None else "")
        )
    verdict = MispricingVerdict(
        status=status, label=status, summary=summary, investable=investable, blocked=blocked,
        primary_benchmark=primary_benchmark, maximum_gap=maximum_text,
        repricing_event=(
            ", ".join(primary_chain.catalyst_ids) + " → " + primary_chain.repricing_mechanism
            if primary_chain else ""
        ),
        verification_deadline=primary_chain.time_window if primary_chain else "",
        failure_point=primary_falsifier.invalidates_if if primary_falsifier else "",
        blockers=[] if status in {"EMERGING EDGE", "ACTIONABLE EDGE"} else list(dict.fromkeys(gate_issues)),
    )
    evidence_based_verdict = build_evidence_based_verdict(
        evidence_status_layer, evidence_acquisition_plan, verdict, valuation_model,
        significant_gap=significant_gap, edge_available=edge_available,
    )

    consensus_lines = [
        f"{item.period} {item.metric}={item.value} {item.unit}"
        for item in points if item.series_type == "market_consensus"
    ]
    implied_lines = [
        f"{item.period} {item.metric}={item.value:.4f} {item.unit}"
        for item in points if item.series_type == "model_implied" and item.value is not None
    ]
    agent_lines = [
        f"{item.scenario} {item.period} {item.metric}={item.value:.4f} {item.unit}"
        for item in points if item.series_type == "agent_forecast"
        and item.metric in {"revenue", "net_profit"} and item.value is not None
    ]
    return MispricingOutput(
        narrative_draft=draft,
        market_expectation_summary="；".join(consensus_lines) if consensus_lines else "Market Consensus unavailable",
        model_implied_summary="；".join(implied_lines) if implied_lines else "Model-Implied Expectation unavailable",
        agent_forecast_summary="；".join(agent_lines) if agent_lines else "Agent Forecast unavailable",
        expectation_points=points, gaps=gaps, maximum_gap=maximum_gap,
        drivers=drivers, falsifiable_conditions=falsifiers, chains=chains,
        variable_gap_contributions=variable_gap_contributions,
        largest_gap_variable=largest_gap_variable,
        catalyst_verification_audit=catalyst_verification_audit,
        investment_logic_chain=investment_logic_chain,
        trackable_conditions=trackable_conditions,
        future_validation_checklist=future_validation_checklist,
        evidence_status_layer=evidence_status_layer,
        evidence_acquisition_plan=evidence_acquisition_plan,
        evidence_based_verdict=evidence_based_verdict,
        edge=edge, classification=_classification(maximum_gap, valuation_model, financial_model),
        gate=gate, verdict=verdict,
        limitations=list(dict.fromkeys(draft.limitations + consensus_issues + gate_issues)),
    )


def replay_mispricing_output(
    stored: MispricingOutput | dict[str, Any] | None,
    expectation: ExpectationOutput | dict[str, Any],
    valuation: ValuationOutput | dict[str, Any],
    catalyst: CatalystOutput | dict[str, Any],
    bear: BearOutput | dict[str, Any],
    research: dict[str, Any],
    financial: FinancialOutput | dict[str, Any],
    industry: IndustryOutput | dict[str, Any] | None = None,
    company: CompanyOutput | dict[str, Any] | None = None,
) -> MispricingOutput:
    stored_data = _dict(stored)
    narrative = stored_data.get("narrative_draft") if stored_data else None
    return calculate_mispricing_output(
        narrative, expectation, valuation, catalyst, bear, research, financial,
        industry=industry, company=company,
        agent_output_available=bool(stored_data),
    )
