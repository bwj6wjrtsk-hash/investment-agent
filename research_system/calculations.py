from __future__ import annotations

import json
import re
from math import isfinite, prod
from statistics import median
from typing import Any
from urllib.parse import urlparse

from .schemas import (
    ComparableValuationEvidenceAudit,
    ComparableValuationEvidenceItem,
    DataConsistencyReport,
    DecisionProbabilitySummary,
    DoubleCountingAudit,
    DoubleCountingItem,
    EvidenceGateResult,
    ExpectationOutput,
    FinancialBridge,
    FinancialOutput,
    ForecastReasonablenessAudit,
    ForecastReasonablenessEntry,
    HistoricalAnchorAuditItem,
    InvestmentVerdict,
    LogicTree,
    LogicTreeNode,
    MarginBridgeAuditItem,
    MarginExpansionEvidenceAudit,
    MarginExpansionEvidenceAuditItem,
    MarketSnapshot,
    NearestScenarioDistance,
    NearestScenarioMatch,
    PreValuationProbabilityAudit,
    ProbabilityAdjustmentStep,
    ProbabilityAuditEntry,
    ProbabilityAuditReport,
    ProbabilityBasis,
    RiskRewardAudit,
    ScenarioProbabilityOutput,
    ScenarioRiskReward,
    ScenarioTriggerAudit,
    ScenarioTriggerAuditItem,
    RequiredEarningsGapAudit,
    RequiredEarningsGapEntry,
    ScenarioSeparationAudit,
    ScenarioSeparationDriver,
    ScenarioValuationAudit,
    SensitivityAuditItem,
    ValidationIssue,
    ValuationAuditReport,
    ValuationComparabilityAudit,
    ValuationGapAudit,
    ValuationGapEntry,
    ValuationMethodApplicabilityAudit,
    ValuationMethodApplicabilityItem,
    ValuationMethodResult,
    ValuationOutput,
)


def _source_key(item: dict[str, Any]) -> str:
    domain = urlparse(str(item.get("source_url") or "")).netloc.lower()
    if domain:
        parts = domain.removeprefix("www.").split(".")
        return ".".join(parts[-2:]) if len(parts) >= 2 else domain
    return re.sub(r"\s+", "", str(item.get("source") or "").lower())


def evaluate_evidence_gate(
    refs: list[str], evidence: dict[str, dict[str, Any]]
) -> EvidenceGateResult:
    accepted: list[str] = []
    rejected: list[str] = []
    a_refs: list[str] = []
    b_sources: set[str] = set()
    for ref in dict.fromkeys(refs):
        item = evidence.get(ref)
        if (
            not item
            or item.get("supports_current_claim") is not True
            or item.get("temporal_scope") == "future"
        ):
            rejected.append(ref)
            continue
        grade = item.get("source_grade")
        if grade == "A":
            a_refs.append(ref)
            accepted.append(ref)
        elif grade == "B":
            source = _source_key(item)
            if source:
                b_sources.add(source)
            accepted.append(ref)
        else:
            rejected.append(ref)
    passed = bool(a_refs or len(b_sources) >= 2)
    reason = (
        f"通过：当前有效A级{len(a_refs)}条，独立B级来源{len(b_sources)}个"
        if passed else f"未通过：需要A级1条或独立B级来源2个；当前A级{len(a_refs)}条、B级来源{len(b_sources)}个"
    )
    return EvidenceGateResult(
        passed=passed, current_a_count=len(a_refs), current_b_source_count=len(b_sources),
        accepted_evidence_refs=accepted, rejected_evidence_refs=rejected, reason=reason,
    )


def build_pre_valuation_probability_audit(
    probability: ScenarioProbabilityOutput,
    evidence: dict[str, dict[str, Any]],
) -> PreValuationProbabilityAudit:
    """Validate one mutually-exclusive/exhaustive probability space before forecasts or valuation."""
    required = ("bear", "base", "bull")
    counts = {name: sum(item.name == name for item in probability.scenarios) for name in required}
    exact = len(probability.scenarios) == 3 and all(counts[name] == 1 for name in required)
    probability_sum = sum(item.base_probability for item in probability.scenarios)
    horizons = {item.horizon.strip() for item in probability.scenarios if item.horizon.strip()}
    common_horizon = exact and len(horizons) == 1
    definitions = [item.event_definition.strip().lower() for item in probability.scenarios]
    boundaries_complete = exact and all(item.boundary_conditions for item in probability.scenarios)
    mutually_exclusive = bool(
        exact and boundaries_complete and len(set(definitions)) == 3
        and probability.mutually_exclusive_reasoning.strip()
    )
    exhaustive = bool(
        exact and boundaries_complete and probability.common_state_variable.strip()
        and probability.exhaustive_reasoning.strip()
    )
    coverage: dict[str, bool] = {}
    for item in probability.scenarios:
        probability_refs = list(dict.fromkeys(
            list(item.probability_basis_refs) + list(item.evidence_refs)
        ))
        valid = [
            ref for ref in probability_refs
            if ref in evidence
            and evidence[ref].get("source_grade") in {"A", "B"}
            and evidence[ref].get("supports_current_claim") is True
            and evidence[ref].get("temporal_scope") != "future"
        ]
        coverage[item.name] = bool(valid)

    issues: list[str] = []
    needed: list[str] = []
    if not exact:
        issues.append("Probability Agent必须且只能生成Bear/Base/Bull各一个情景")
    if abs(probability_sum - 1) > 1e-9:
        issues.append(f"Probability Agent原始三情景概率必须精确合计100%，当前为{probability_sum:.6%}")
    if not probability.common_state_variable.strip():
        issues.append("缺少三情景共同状态变量")
    if not common_horizon:
        issues.append("Bear/Base/Bull必须使用相同且非空的概率预测期")
    if not mutually_exclusive:
        issues.append("三情景缺少可审查的互斥边界或互斥性说明")
    if not exhaustive:
        issues.append("三情景缺少穷尽性说明，无法证明覆盖共同状态变量全部结果")
    for item in probability.scenarios:
        if not item.reasoning.strip():
            issues.append(f"{item.name.title()}缺少概率reasoning")
        if not coverage.get(item.name, False):
            issues.append(f"{item.name.title()}概率缺少当前有效、非未来A/B级Evidence Ref")
            needed.append(f"补充{item.name.title()}情景概率的当前有效A/B级证据")
    passed = bool(
        exact and abs(probability_sum - 1) <= 1e-9
        and probability.common_state_variable.strip() and common_horizon
        and mutually_exclusive and exhaustive
        and all(item.reasoning.strip() for item in probability.scenarios)
        and all(coverage.get(name, False) for name in required)
    )
    return PreValuationProbabilityAudit(
        passed=passed,
        exact_scenarios=exact,
        probability_sum=probability_sum,
        mutually_exclusive=mutually_exclusive,
        exhaustive=exhaustive,
        mutually_exclusive_reasoning=probability.mutually_exclusive_reasoning,
        exhaustive_reasoning=probability.exhaustive_reasoning,
        common_state_variable=probability.common_state_variable,
        common_horizon=common_horizon,
        evidence_coverage=coverage,
        scenarios=probability.scenarios,
        issues=list(dict.fromkeys(issues)),
        evidence_needed=list(dict.fromkeys(needed)),
    )


def revalidate_pre_valuation_probability_audit(
    output: ValuationOutput,
    evidence: dict[str, dict[str, Any]],
) -> ValuationOutput:
    audit = output.pre_valuation_probability_audit
    if not audit.scenarios:
        output.pre_valuation_probability_audit = PreValuationProbabilityAudit(
            issues=["历史结果缺少独立Pre-Valuation Probability Gate；旧概率不得继续用于估值"],
            evidence_needed=["重新运行Probability Agent并生成互斥、穷尽且合计100%的三情景概率"],
        )
        return output
    probability = ScenarioProbabilityOutput(
        common_state_variable=audit.common_state_variable,
        mutually_exclusive_reasoning=audit.mutually_exclusive_reasoning,
        exhaustive_reasoning=audit.exhaustive_reasoning,
        scenarios=audit.scenarios,
    )
    output.pre_valuation_probability_audit = build_pre_valuation_probability_audit(probability, evidence)
    return output


def _financial_baseline(financial: FinancialOutput) -> tuple[float | None, float | None, str]:
    candidates = [item for item in financial.historical if item.revenue is not None]
    if not candidates:
        return None, None, ""
    def key(item: Any) -> tuple[int, int]:
        text = str(item.period)
        year_match = re.search(r"20\d{2}", text)
        year = int(year_match.group()) if year_match else 0
        annual = int(bool(
            re.fullmatch(r"20\d{2}", text.strip())
            or "12-31" in text or "年报" in text or "FY" in text.upper()
        ))
        return year, annual
    annuals = [item for item in candidates if key(item)[1] == 1]
    selected = max(annuals or candidates, key=key)
    return selected.revenue, selected.net_profit, str(selected.period)


def _latest_reliable_annual_margin_anchors(
    financial: FinancialOutput,
) -> dict[str, tuple[float, str, list[str]]]:
    """Return directly evidenced margins from the latest usable annual structure."""
    field_sources = {
        "gross_margin": ("gross_margin", "gross_profit", "revenue"),
        "ebit_margin": ("ebit_margin", "ebit", "revenue"),
        "ebitda_margin": ("ebitda_margin", "ebitda", "revenue"),
    }
    candidates: list[tuple[int, Any, dict[str, list[str]]]] = []
    for period in financial.historical:
        text = str(period.period)
        match = re.search(r"20\d{2}", text)
        annual = bool(
            match and (
                re.fullmatch(r"20\d{2}", text.strip())
                or "12-31" in text or "年报" in text or "FY" in text.upper()
            )
        )
        if not annual or period.revenue is None or period.revenue <= 0:
            continue
        refs_by_field: dict[str, list[str]] = {}
        for field, source_fields in field_sources.items():
            value = getattr(period, field)
            if value is None or not isfinite(float(value)):
                continue
            direct_refs = list(dict.fromkeys(
                list(period.evidence_refs)
                + [
                    ref
                    for source_field in source_fields
                    for ref in (period.field_evidence_refs or {}).get(source_field, [])
                ]
            ))
            if direct_refs:
                refs_by_field[field] = direct_refs
        if refs_by_field:
            candidates.append((int(match.group()), period, refs_by_field))
    if not candidates:
        return {}
    _, selected, refs_by_field = max(candidates, key=lambda row: row[0])
    return {
        field: (float(getattr(selected, field)), str(selected.period), refs)
        for field, refs in refs_by_field.items()
    }


def calculate_price_implied_path(
    output: ExpectationOutput, research: dict[str, Any], financial: FinancialOutput
) -> ExpectationOutput:
    snapshot = research.get("market_snapshot") or {}
    price = snapshot.get("price")
    shares = snapshot.get("shares_outstanding")
    market_cap = snapshot.get("market_cap")
    if market_cap is None and price is not None and shares is not None:
        market_cap = price * shares
    base_revenue, base_profit, base_period = _financial_baseline(financial)
    anchors = _latest_reliable_annual_margin_anchors(financial)
    anchor_refs = list(dict.fromkeys(
        ref for _, _, refs in anchors.values() for ref in refs
    ))
    path = output.price_implied_path or output.market_implied_path
    previous_profit, previous_revenue = base_profit, base_revenue
    for index, item in enumerate(path, start=1):
        item.current_price = price
        item.shares_outstanding = shares
        item.current_market_cap = market_cap
        item.years_from_now = index
        rate = item.required_return / 100 if abs(item.required_return) > 1 else item.required_return
        margin = (
            item.assumed_net_margin / 100
            if item.assumed_net_margin is not None and abs(item.assumed_net_margin) > 1
            else item.assumed_net_margin
        )
        item.required_return = rate
        item.assumed_net_margin = margin
        item.implied_net_margin = margin
        item.base_net_profit = base_profit
        item.base_revenue = base_revenue
        item.target_market_cap = (
            market_cap * ((1 + rate) ** index)
            if market_cap is not None and market_cap >= 0 and 1 + rate >= 0 else None
        )
        valid_pe = item.assumed_pe is not None and item.assumed_pe > 0
        # The selected reverse-valuation coordinate is explicitly PE even when its
        # inputs are incomplete; solution_status and null outputs carry availability.
        item.implied_valuation_method = "pe"
        item.implied_valuation_multiple = item.assumed_pe
        item.implied_net_profit = (
            item.target_market_cap / item.assumed_pe
            if item.target_market_cap is not None and valid_pe else None
        )
        item.implied_revenue = (
            item.implied_net_profit / margin
            if item.implied_net_profit is not None and margin is not None and margin > 0 else None
        )
        item.implied_gross_margin = anchors.get("gross_margin", (None, "", []))[0]
        item.implied_ebit_margin = anchors.get("ebit_margin", (None, "", []))[0]
        item.implied_ebitda_margin = anchors.get("ebitda_margin", (None, "", []))[0]
        item.implied_ebit = (
            item.implied_revenue * item.implied_ebit_margin
            if item.implied_revenue is not None and item.implied_ebit_margin is not None else None
        )
        item.implied_ebitda = (
            item.implied_revenue * item.implied_ebitda_margin
            if item.implied_revenue is not None and item.implied_ebitda_margin is not None else None
        )
        item.implied_net_profit_growth = (
            item.implied_net_profit / previous_profit - 1
            if item.implied_net_profit is not None and previous_profit is not None and previous_profit > 0 else None
        )
        item.implied_revenue_growth = (
            item.implied_revenue / previous_revenue - 1
            if item.implied_revenue is not None and previous_revenue is not None and previous_revenue > 0 else None
        )
        item.implied_growth = item.implied_net_profit_growth
        if item.implied_net_profit is not None:
            previous_profit = item.implied_net_profit
        if item.implied_revenue is not None:
            previous_revenue = item.implied_revenue

        core_solved = item.implied_net_profit is not None and item.implied_revenue is not None
        anchor_complete = all(value is not None for value in (
            item.implied_gross_margin, item.implied_ebit_margin, item.implied_ebitda_margin,
            item.implied_ebit, item.implied_ebitda,
        ))
        if core_solved and anchor_complete:
            item.solution_status = "solved"
        elif item.implied_net_profit is not None or item.target_market_cap is not None:
            item.solution_status = "partial"
        else:
            item.solution_status = "unavailable"
        item.evidence_refs = list(dict.fromkeys(item.evidence_refs + anchor_refs))
        anchor_details = "、".join(
            f"{field}={value:.2%}（{period}年度）"
            for field, (value, period, _) in anchors.items()
        )
        anchor_disclosure = (
            f"成本结构锚：{anchor_details}；全部利润率来自同一最近可靠年度历史财务结构，仅用于给定成本/利润率假设下的Model-Implied非唯一解"
            if anchors
            else "成本结构锚不可用：缺少具备Evidence Ref、正收入且利润率可用的可靠年度历史财务结构；相关Model-Implied结构字段保持null，solution_status至多partial"
        )
        item.assumptions = list(dict.fromkeys(item.assumptions + [
            "口径声明：以下结果是给定成本/利润率和退出PE假设下的Model-Implied，不是Market Consensus，也不表示已被Price In",
            f"要求回报率 {rate:.1%}",
            f"退出PE {item.assumed_pe:g}倍" if valid_pe else "退出PE不可用",
            f"净利率假设 {margin:.1%}" if margin is not None else "净利率假设不可用",
            f"财务基期 {base_period or '不可用'}",
            anchor_disclosure,
        ]))
    output.price_implied_path = path
    output.market_implied_path = path
    return output


def _check_close(check: str, actual: float, expected: float, tolerance: float = 1e-8) -> ValidationIssue:
    scale = max(1.0, abs(actual), abs(expected))
    passed = isfinite(actual) and isfinite(expected) and abs(actual - expected) <= tolerance * scale
    return ValidationIssue(
        check=check, severity="info" if passed else "error", passed=passed,
        expected=f"{expected:.8f}", actual=f"{actual:.8f}",
        message="通过" if passed else "程序化财务恒等式不一致，估值结果已禁用",
    )


def _build_valuation_comparability_audit(
    output: ValuationOutput,
    by_name: dict[str, Any],
    complete: bool,
    complete_targets: bool,
    ordered_targets: bool,
) -> ValuationComparabilityAudit:
    """Audit economic comparability without changing scenario values or their ordering."""
    required_names = ("bear", "base", "bull")
    terminals = {
        name: by_name[name].projections[-1]
        for name in required_names
        if name in by_name and by_name[name].projections
    }
    methods = {
        name: terminals[name].valuation_method
        for name in required_names if name in terminals
    }
    method_values = list(methods.values())
    same_primary_method = bool(
        complete and len(method_values) == 3
        and "unavailable" not in method_values and len(set(method_values)) == 1
    )
    cross_method_switch = len(set(method_values)) > 1
    common_primary_method = method_values[0] if same_primary_method else ""

    same_valuation_system = bool(
        complete and len(terminals) == 3 and output.shares_outstanding not in {None, 0}
        and all(
            terminal.enterprise_value is not None
            and terminal.equity_value is not None
            and terminal.valuation_method != "unavailable"
            and by_name[name].target_price is not None
            for name, terminal in terminals.items()
        )
    )
    all_identity_checks_passed = bool(
        complete and len(terminals) == 3
        and all(
            projection.calculation_valid
            and projection.calculation_checks
            and all(check.passed for check in projection.calculation_checks)
            for name in required_names if name in by_name
            for projection in by_name[name].projections
        )
    )
    unsupported_multiple_scenarios: list[str] = []
    for name, terminal in terminals.items():
        selected = next((candidate for candidate in terminal.valuation_candidates if candidate.selected), None)
        if selected is not None and not selected.evidence_supported:
            unsupported_multiple_scenarios.append(name)

    cross_method_sorting_reversal = bool(
        cross_method_switch and complete_targets and not ordered_targets
    )
    issues: list[str] = []
    if not same_valuation_system:
        issues.append("三情景未全部映射到统一 EV→Equity Value→Target Price 估值桥")
    if not all_identity_checks_passed:
        issues.append("至少一个预测期的财务或 EV/Equity/Target Price 恒等式未通过")
    if cross_method_sorting_reversal:
        issues.append("跨估值方法切换导致目标价经济排序反转；禁止人为修正")
    if unsupported_multiple_scenarios:
        issues.append(
            "以下情景的终值倍数假设缺少直接 Evidence Ref："
            + ", ".join(unsupported_multiple_scenarios)
        )

    passed = bool(
        same_valuation_system and all_identity_checks_passed
        and not cross_method_sorting_reversal and not unsupported_multiple_scenarios
    )
    return ValuationComparabilityAudit(
        passed=passed,
        same_valuation_system=same_valuation_system,
        same_primary_method=same_primary_method,
        common_primary_method=common_primary_method,
        methods_by_scenario=methods,
        cross_method_switch=cross_method_switch,
        cross_method_sorting_reversal=cross_method_sorting_reversal,
        all_identity_checks_passed=all_identity_checks_passed,
        unsupported_multiple_scenarios=unsupported_multiple_scenarios,
        issues=issues,
    )


def _canonical_trigger_variables(value: str) -> set[str]:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    aliases = {
        "revenue_growth": ("revenue_growth", "revenue", "sales_growth", "收入增长", "营收增长"),
        "gross_margin": ("gross_margin", "gross_profit_margin", "毛利率"),
        "rd_expense_ratio": ("rd_expense_ratio", "r_d_expense_ratio", "研发费用率"),
        "selling_expense_ratio": ("selling_expense_ratio", "sales_expense_ratio", "销售费用率"),
        "admin_expense_ratio": ("admin_expense_ratio", "administrative_expense_ratio", "管理费用率"),
        "other_operating_expense_ratio": ("other_operating_expense_ratio", "other_opex_ratio", "其他经营费用率"),
        "financial_expense_ratio": ("financial_expense_ratio", "财务费用率"),
        "other_income_ratio": ("other_income_ratio", "其他收益率"),
        "tax_rate": ("tax_rate", "税率"),
        "minority_interest_ratio": ("minority_interest_ratio", "少数股东损益率"),
    }
    compact_original = value.lower().replace(" ", "")
    covered: set[str] = set()
    for canonical, names in aliases.items():
        if any(
            alias in normalized or alias.replace("_", "") in normalized.replace("_", "")
            or alias in compact_original
            for alias in names
        ):
            covered.add(canonical)
    return covered


def build_scenario_trigger_audit(
    output: ValuationOutput, evidence: dict[str, dict[str, Any]],
) -> ScenarioTriggerAudit:
    """Require evidenced business mechanisms for Base and every material Base delta."""

    required = ("bear", "base", "bull")
    by_name = {scenario.name: scenario for scenario in output.scenarios}
    items: list[Any] = []
    verified_by_scenario: dict[str, set[str]] = {name: set() for name in required}
    verified_directions: dict[str, dict[str, set[str]]] = {
        name: {} for name in required
    }
    issues: list[str] = []
    needed: list[str] = []
    missing_scenarios: list[str] = []
    for name in required:
        scenario = by_name.get(name)
        triggers = scenario.business_triggers if scenario else []
        if not triggers:
            missing_scenarios.append(name)
            issues.append(f"{name.title()}缺少业务触发项")
            needed.append(f"补充{name.title()}情景business_change→financial_variable→mechanism及Evidence Ref")
            continue
        for index, trigger in enumerate(triggers, start=1):
            gate = evaluate_evidence_gate(trigger.evidence_refs, evidence)
            trigger.evidence_gate = gate
            row_issues: list[str] = []
            if not trigger.business_change.strip():
                row_issues.append("business_change为空")
            if not trigger.financial_variable.strip():
                row_issues.append("financial_variable为空")
            if not trigger.mechanism.strip():
                row_issues.append("mechanism为空")
            if trigger.direction == "unknown":
                row_issues.append("direction必须明确为increase/decrease/mixed")
            if not gate.passed:
                row_issues.append(f"Evidence Gate未通过：{gate.reason}")
            if gate.rejected_evidence_refs:
                row_issues.append(
                    "存在非当前有效、future或非A/B级Evidence Ref："
                    + ", ".join(gate.rejected_evidence_refs)
                )
            passed = not row_issues
            covered = _canonical_trigger_variables(trigger.financial_variable) if passed else set()
            verified_by_scenario[name].update(covered)
            for variable in covered:
                verified_directions[name].setdefault(variable, set()).add(trigger.direction)
            trigger_id = trigger.trigger_id or f"{name}-trigger-{index}"
            items.append(ScenarioTriggerAuditItem(
                scenario=name,
                trigger_id=trigger_id,
                business_change=trigger.business_change,
                financial_variable=trigger.financial_variable,
                direction=trigger.direction,
                mechanism=trigger.mechanism,
                evidence_refs=list(dict.fromkeys(trigger.evidence_refs)),
                evidence_gate=gate,
                passed=passed,
                status="verified" if passed else "blocked",
                issues=row_issues,
            ))
            if row_issues:
                issues.append(f"{name.title()} {trigger_id}: {'；'.join(row_issues)}")

    base = by_name.get("base")
    material_variables = (
        "revenue_growth", "gross_margin", "rd_expense_ratio", "selling_expense_ratio",
        "admin_expense_ratio", "other_operating_expense_ratio",
    )
    if base:
        base_path = {str(item.year): item for item in base.projections}
        for name in ("bear", "bull"):
            scenario = by_name.get(name)
            if not scenario:
                continue
            scenario_path = {str(item.year): item for item in scenario.projections}
            for year in sorted(set(base_path) & set(scenario_path)):
                base_item, scenario_item = base_path[year], scenario_path[year]
                for variable in material_variables:
                    delta = float(getattr(scenario_item, variable)) - float(getattr(base_item, variable))
                    if abs(delta) <= 1e-9:
                        continue
                    expected_direction = "increase" if delta > 0 else "decrease"
                    directions = verified_directions[name].get(variable, set())
                    if variable not in verified_by_scenario[name] or not ({expected_direction, "mixed"} & directions):
                        message = (
                            f"{name.title()} {year}相对Base变化的{variable}"
                            f"没有对应、direction={expected_direction}且通过Evidence Gate的financial_variable trigger"
                        )
                        issues.append(message)
                        needed.append(f"补充{name.title()} {year} {variable}变化的业务触发机制、明确方向和独立可靠证据")
    else:
        issues.append("缺少Base情景，无法审计Bear/Bull相对Base的财务变量触发覆盖")

    passed = bool(
        set(by_name) >= set(required) and not missing_scenarios and items
        and all(item.passed for item in items) and not issues
    )
    return ScenarioTriggerAudit(
        passed=passed,
        status="available" if passed else ("unavailable" if not items else "blocked"),
        items=items,
        missing_scenarios=missing_scenarios,
        issues=list(dict.fromkeys(issues)),
        evidence_needed=list(dict.fromkeys(needed)),
    )


def _model_implied_values(item: Any) -> dict[str, float | None]:
    return {
        "revenue": item.implied_revenue,
        "net_profit": item.implied_net_profit,
        "gross_margin": item.implied_gross_margin,
        "ebit_margin": item.implied_ebit_margin,
        "ebit": item.implied_ebit,
        "ebitda": item.implied_ebitda,
        "net_margin": item.implied_net_margin,
        "valuation_multiple": item.implied_valuation_multiple,
    }


def _scenario_projection_values(item: FinancialBridge) -> dict[str, float | None]:
    return {
        "revenue": item.revenue,
        "net_profit": item.net_profit,
        "gross_margin": item.gross_margin,
        "ebit_margin": item.ebit / item.revenue if item.revenue else None,
        "ebit": item.ebit,
        "ebitda": item.ebitda,
        "net_margin": item.net_profit / item.revenue if item.revenue else None,
        "valuation_multiple": item.valuation_multiple if item.valuation_multiple > 0 else None,
    }


def calculate_nearest_scenario_match(output: ValuationOutput, epsilon: float = 1e-9) -> NearestScenarioMatch:
    """Match the first Model-Implied year using available specified variables only."""

    required = ("bear", "base", "bull")
    by_name = {scenario.name: scenario for scenario in output.scenarios}
    common_years = set.intersection(*(
        {str(item.year) for item in by_name[name].projections}
        for name in required if name in by_name
    )) if set(by_name) >= set(required) else set()
    implied = next((
        item for item in output.reverse_valuation
        if item.solution_status in {"solved", "partial"}
        and any(value is not None for value in _model_implied_values(item).values())
    ), None)
    if implied is None:
        reason = "缺少首个可用Model-Implied记录，无法计算最近情景"
        return NearestScenarioMatch(status="unavailable", reasoning=reason, blocked_reason=reason)
    if str(implied.year) not in common_years:
        reason = f"首个Model-Implied年份{implied.year}不是Bear/Base/Bull共同预测年份，匹配不可用"
        return NearestScenarioMatch(status="unavailable", reasoning=reason, blocked_reason=reason)

    implied_values = _model_implied_values(implied)
    distances: list[Any] = []
    comparable: list[Any] = []
    for name in required:
        projection = next(
            (item for item in by_name[name].projections if str(item.year) == str(implied.year)), None
        )
        variable_distances: dict[str, float | None] = {}
        row_issues: list[str] = []
        if projection is None:
            row_issues.append(f"{name.title()}缺少{implied.year}同年预测")
        scenario_values = _scenario_projection_values(projection) if projection else {}
        for variable, implied_value in implied_values.items():
            agent_value = scenario_values.get(variable)
            if implied_value is None or agent_value is None:
                variable_distances[variable] = None
                missing_side = "Model-Implied" if implied_value is None else name.title()
                row_issues.append(f"{variable}缺少{missing_side}关键变量")
                continue
            variable_distances[variable] = abs(agent_value - implied_value) / max(abs(implied_value), epsilon)
        available_values = [value for value in variable_distances.values() if value is not None]
        available = len(available_values) >= 2
        normalized = sum(available_values) / len(available_values) if available else None
        row = NearestScenarioDistance(
            scenario=name,
            total_distance=sum(available_values) if available else None,
            normalized_distance=normalized,
            variable_distances=variable_distances,
            available=available,
            formula=(
                f"mean(abs(agent-implied)/max(abs(implied),{epsilon:g})); n={len(available_values)}"
                if available else f"仅{len(available_values)}项可比；至少需要2项"
            ),
            issues=list(dict.fromkeys(row_issues + ([] if available else ["可比关键变量少于2项"]))),
        )
        distances.append(row)
        if available:
            comparable.append(row)
    if len(comparable) != 3:
        reason = "Bear/Base/Bull并非都至少有2项同年关键变量可比，最近情景匹配不可用"
        return NearestScenarioMatch(
            status="unavailable", distances=distances, reasoning=reason, blocked_reason=reason
        )
    nearest = min(comparable, key=lambda row: row.normalized_distance)
    missing_count = sum(
        value is None for row in distances for value in row.variable_distances.values()
    )
    return NearestScenarioMatch(
        status="matched",
        nearest_scenario=nearest.scenario,
        distance=nearest.normalized_distance,
        distances=distances,
        reasoning=(
            f"Model-Implied {implied.year}与{nearest.scenario.title()}平均标准化距离最小；"
            f"按8个指定变量中可用项计算，缺失变量共{missing_count}项并已逐项列入issues"
        ),
    )


def build_valuation_method_applicability_audit(
    output: ValuationOutput, evidence: dict[str, dict[str, Any]],
) -> ValuationMethodApplicabilityAudit:
    """Validate each selected method, metric, inputs, and independent comparable support."""

    required = ("bear", "base", "bull")
    items: list[Any] = []
    issues: list[str] = []
    for name in required:
        scenario = next((item for item in output.scenarios if item.name == name), None)
        terminal = scenario.projections[-1] if scenario and scenario.projections else None
        if terminal is None:
            issues.append(f"{name.title()}缺少终值预测，无法检查估值方法适用性")
            items.append(ValuationMethodApplicabilityItem(scenario=name, status="unavailable"))
            continue
        selected = next((candidate for candidate in terminal.valuation_candidates if candidate.selected), None)
        method = selected.method if selected else terminal.valuation_method
        required_inputs: list[str]
        missing: list[str] = []
        metric_name = ""
        metric_value: float | None = None
        if method == "pe":
            required_inputs = ["net_profit>0", "pe_multiple>0", "selected_candidate", "evidence_supported"]
            metric_name, metric_value = "net_profit", terminal.net_profit
            if terminal.net_profit <= 0:
                missing.append("net_profit>0（净利润<=0时PE禁用）")
            if terminal.pe_multiple <= 0:
                missing.append("pe_multiple>0")
        elif method == "ev_ebitda":
            required_inputs = ["ebitda>0", "ev_ebitda_multiple>0", "net_debt", "selected_candidate", "evidence_supported"]
            metric_name, metric_value = "ebitda", terminal.ebitda
            if terminal.ebitda <= 0:
                missing.append("ebitda>0")
            if terminal.ev_ebitda_multiple <= 0:
                missing.append("ev_ebitda_multiple>0")
        elif method == "ev_sales":
            required_inputs = ["revenue>0", "ev_sales_multiple>0", "net_debt", "selected_candidate", "evidence_supported"]
            metric_name, metric_value = "revenue", terminal.revenue
            if terminal.revenue <= 0:
                missing.append("revenue>0")
            if terminal.ev_sales_multiple <= 0:
                missing.append("ev_sales_multiple>0")
        elif method == "pb":
            required_inputs = ["book_value>0", "pb_multiple>0", "selected_candidate", "evidence_supported"]
            metric_name, metric_value = "book_value", terminal.book_value
            if terminal.book_value <= 0:
                missing.append("book_value>0")
            if terminal.pb_multiple <= 0:
                missing.append("pb_multiple>0")
        else:
            required_inputs = ["selected_supported_valuation_method"]
            method = "unavailable"
            missing.append("PE/EV-EBITDA/EV-Sales/PB均未选出")
        valid_comparable_refs = list(dict.fromkeys(
            record.evidence_ref
            for record in (selected.comparable_evidence if selected else [])
            if record.valid and record.evidence_ref
        ))
        comparable_gate = evaluate_evidence_gate(valid_comparable_refs, evidence)
        if terminal.valuation_multiple <= 0:
            missing.append("valuation_multiple>0")
        if selected is None:
            missing.append("selected_candidate")
        elif not selected.evidence_supported:
            missing.append("selected candidate evidence_supported")
        if selected is not None and not selected.available:
            missing.append("selected candidate available")
        if not valid_comparable_refs:
            missing.append("可靠且方法匹配的independent comparable evidence")
        elif not comparable_gate.passed:
            missing.append(f"independent comparable Evidence Gate未通过：{comparable_gate.reason}")
        applicable = not missing
        if not applicable:
            issues.append(f"{name.title()} {method}不适用或输入不足：{'、'.join(missing)}")
        items.append(ValuationMethodApplicabilityItem(
            scenario=name,
            method=method,
            applicable=applicable,
            status="applicable" if applicable else "blocked",
            financial_metric=metric_name,
            metric_value=metric_value,
            required_inputs=required_inputs,
            missing_inputs=list(dict.fromkeys(missing)),
            evidence_refs=valid_comparable_refs,
            reasoning=(
                f"{method}财务适用性、必需输入及独立可靠Comparable Evidence Gate均通过"
                if applicable else "；".join(missing)
            ),
        ))
    passed = len(items) == 3 and all(item.applicable for item in items)
    return ValuationMethodApplicabilityAudit(
        passed=passed,
        status="available" if passed else ("unavailable" if not items else "blocked"),
        primary_method=output.primary_valuation_method,
        items=items,
        issues=list(dict.fromkeys(issues)),
    )


def _mechanism_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", str(value or "").lower())


def _evidence_mechanism_keys(
    refs: list[str], evidence: dict[str, dict[str, Any]],
) -> set[str]:
    keys: set[str] = set()
    for ref in refs:
        item = evidence.get(ref) or {}
        excerpt = item.get("excerpt")
        try:
            metadata = json.loads(str(excerpt or ""))
        except (TypeError, json.JSONDecodeError):
            metadata = {}
        if isinstance(metadata, dict):
            for field in ("mechanism", "business_change", "applicability", "rationale"):
                key = _mechanism_key(metadata.get(field))
                if key:
                    keys.add(key)
    return keys


def build_double_counting_audit(
    output: ValuationOutput, evidence: dict[str, dict[str, Any]],
) -> DoubleCountingAudit:
    """Block a scenario when one evidence/mechanism supports both profit and re-rating."""

    base = next((scenario for scenario in output.scenarios if scenario.name == "base"), None)
    if base is None or not base.projections:
        return DoubleCountingAudit(
            passed=False, status="unavailable", issues=["缺少Base终值，无法执行Double Counting Audit"]
        )
    if not output.valuation_method_applicability_audit.passed:
        return DoubleCountingAudit(
            passed=False,
            status="unavailable",
            issues=[
                "估值方法适用性或独立倍数证据未通过，盈利增长与估值提升是否重复计价不可判定"
            ],
        )
    base_terminal = base.projections[-1]
    operating_variables = (
        "revenue_growth", "gross_margin", "rd_expense_ratio", "selling_expense_ratio",
        "admin_expense_ratio", "other_operating_expense_ratio", "financial_expense_ratio",
        "other_income_ratio", "tax_rate", "minority_interest_ratio",
    )
    items: list[Any] = []
    unresolved: list[str] = []
    issues: list[str] = []
    for scenario in output.scenarios:
        if scenario.name == "base" or not scenario.projections:
            continue
        terminal = scenario.projections[-1]
        operating_profit_improved = (
            terminal.operating_profit > base_terminal.operating_profit + 1e-9
        )
        net_profit_improved = terminal.net_profit > base_terminal.net_profit + 1e-9
        profit_improved = operating_profit_improved or net_profit_improved
        multiple_improved = terminal.valuation_multiple > base_terminal.valuation_multiple + 1e-9
        if not (profit_improved and multiple_improved):
            continue
        changed = [
            variable for variable in operating_variables
            if abs(float(getattr(terminal, variable)) - float(getattr(base_terminal, variable))) > 1e-9
        ]
        operating_refs = list(dict.fromkeys(
            ref for variable in changed for ref in terminal.assumption_evidence_refs.get(variable, [])
        ))
        operating_mechanisms: set[str] = set()
        for trigger in scenario.business_triggers:
            if _canonical_trigger_variables(trigger.financial_variable) & set(changed):
                operating_refs.extend(trigger.evidence_refs)
                for value in (trigger.business_change, trigger.mechanism):
                    key = _mechanism_key(value)
                    if key:
                        operating_mechanisms.add(key)
        operating_refs = list(dict.fromkeys(operating_refs))
        operating_mechanisms.update(_evidence_mechanism_keys(operating_refs, evidence))
        selected = next((candidate for candidate in terminal.valuation_candidates if candidate.selected), None)
        fallback_multiple_refs = terminal.assumption_evidence_refs.get(
            {"pe": "pe_multiple", "ev_ebitda": "ev_ebitda_multiple", "ev_sales": "ev_sales_multiple", "pb": "pb_multiple"}.get(terminal.valuation_method, ""), []
        )
        multiple_refs = list(dict.fromkeys(
            (list(selected.evidence_refs) if selected else list(fallback_multiple_refs))
            + (
                [
                    record.evidence_ref for record in selected.comparable_evidence
                    if record.valid and record.evidence_ref
                ]
                if selected else []
            )
        ))
        overlap = sorted(set(operating_refs) & set(multiple_refs))
        multiple_mechanisms = _evidence_mechanism_keys(multiple_refs, evidence)
        if selected:
            for record in selected.comparable_evidence:
                key = _mechanism_key(record.applicability)
                if key:
                    multiple_mechanisms.add(key)
        mechanism_overlap = sorted(operating_mechanisms & multiple_mechanisms)
        operating_gate = evaluate_evidence_gate(operating_refs, evidence)
        multiple_gate = evaluate_evidence_gate(multiple_refs, evidence)
        independent = bool(
            operating_refs and multiple_refs and not overlap and not mechanism_overlap
            and operating_gate.passed and multiple_gate.passed
        )
        item_id = f"{scenario.name}-profit-and-multiple"
        if not independent:
            unresolved.append(item_id)
            detail = []
            if not operating_refs or not operating_gate.passed:
                detail.append("经营变量证据缺失或未通过Evidence Gate")
            if not multiple_refs or not multiple_gate.passed:
                detail.append("估值倍数证据缺失或未通过Evidence Gate")
            if overlap:
                detail.append(f"经营与倍数证据重叠：{', '.join(overlap)}")
            if mechanism_overlap:
                detail.append(f"经营与re-rating复用同一mechanism：{', '.join(mechanism_overlap)}")
            issues.append(f"{scenario.name.title()}同时盈利改善和倍数提高：{'；'.join(detail)}")
        items.append(DoubleCountingItem(
            item_id=item_id,
            scenario=scenario.name,
            first_component=(
                "经营改善："
                f"operating_profit {base_terminal.operating_profit:.6f}→{terminal.operating_profit:.6f}；"
                f"net_profit {base_terminal.net_profit:.6f}→{terminal.net_profit:.6f}"
            ),
            second_component=f"倍数提高：{base_terminal.valuation_multiple:.6f}→{terminal.valuation_multiple:.6f}",
            overlap_description=(
                "；".join(filter(None, [
                    f"重叠Evidence Refs: {', '.join(overlap)}" if overlap else "",
                    f"重叠Mechanisms: {', '.join(mechanism_overlap)}" if mechanism_overlap else "",
                ])) or "经营变量与倍数的Evidence Refs及mechanism均独立不重叠"
            ),
            financial_impact=terminal.net_profit - base_terminal.net_profit,
            detected=not independent,
            resolved=independent,
            resolution=(
                "经营改善与估值倍数提高分别由通过Evidence Gate的独立非重叠证据支持"
                if independent else "证据缺失、无效或重叠；不得把同一事实同时用于利润和倍数上调"
            ),
            evidence_refs=list(dict.fromkeys(operating_refs + multiple_refs)),
            status="clear" if independent else "blocked",
        ))
    passed = not unresolved
    return DoubleCountingAudit(
        passed=passed,
        status="clear" if passed else "blocked",
        items=items,
        unresolved_item_ids=unresolved,
        issues=list(dict.fromkeys(issues)),
    )


def build_risk_reward_audit(output: ValuationOutput) -> RiskRewardAudit:
    """Compute risk/reward only from a passed valuation audit and three complete targets."""

    target_complete = bool(
        len(output.scenarios) == 3
        and {scenario.name for scenario in output.scenarios} == {"bear", "base", "bull"}
        and all(scenario.target_price is not None for scenario in output.scenarios)
    )
    extended_gates_passed = bool(
        output.scenario_trigger_audit.passed
        and output.valuation_method_applicability_audit.passed
        and output.double_counting_audit.passed
    )
    if not output.valuation_audit.passed or not extended_gates_passed or not target_complete:
        if not output.valuation_audit.passed:
            gate_reason = "最终Valuation Audit未全部通过"
        elif not target_complete:
            gate_reason = "Bear/Base/Bull三个目标价不完整"
        else:
            gate_reason = "最终Valuation Audit所需扩展Gate未全部通过"
        return RiskRewardAudit(
            passed=False,
            status="blocked",
            scenarios=[
                ScenarioRiskReward(
                    scenario=name, status="blocked",
                    reasoning=f"{gate_reason}，风险收益数值禁用",
                )
                for name in ("bear", "base", "bull")
            ],
            blocked_reason=f"{gate_reason}；所有关键风险收益值保持None",
            issues=list(dict.fromkeys(
                output.valuation_audit.issues
                + output.scenario_trigger_audit.issues
                + output.valuation_method_applicability_audit.issues
                + output.double_counting_audit.issues
            )),
        )
    by_name = {scenario.name: scenario for scenario in output.scenarios}
    rows: list[Any] = []
    issues: list[str] = []
    for name in ("bear", "base", "bull"):
        scenario = by_name.get(name)
        if scenario is None or output.current_price in {None, 0} or scenario.target_price is None:
            issues.append(f"{name.title()}缺少当前价或目标价")
            rows.append(ScenarioRiskReward(scenario=name, status="blocked"))
            continue
        price_return = scenario.target_price / output.current_price - 1
        rows.append(ScenarioRiskReward(
            scenario=name,
            probability=scenario.probability,
            current_price=output.current_price,
            target_price=scenario.target_price,
            price_return=price_return,
            annualized_return=scenario.annualized_price_return,
            weighted_return=scenario.probability * price_return,
            downside=max(-price_return, 0.0),
            upside=max(price_return, 0.0),
            status="available",
            reasoning=(
                "涨跌幅/安全边际=目标价÷当前价-1；"
                "upside=max(安全边际,0)，downside=max(-安全边际,0)；"
                "概率加权值=对应幅度×最终归一化情景概率"
            ),
        ))
    if issues or len(rows) != 3:
        return RiskRewardAudit(
            passed=False, status="blocked", scenarios=rows, issues=issues,
            blocked_reason="风险收益必需输入缺失；所有汇总保持None",
        )
    expected_upside = sum((row.probability or 0) * (row.upside or 0) for row in rows)
    expected_downside = sum((row.probability or 0) * (row.downside or 0) for row in rows)
    expected_return = sum(row.weighted_return or 0 for row in rows)
    probability_of_loss = sum((row.probability or 0) for row in rows if (row.price_return or 0) < 0)
    ratio = expected_upside / expected_downside if expected_downside > 0 else None
    ratio_issues = [] if ratio is not None else ["Expected downside为0，upside/downside ratio数学上不可定义，保持None"]
    return RiskRewardAudit(
        passed=True,
        status="available",
        scenarios=rows,
        expected_return=expected_return,
        annualized_expected_return=output.annualized_expected_return,
        probability_of_loss=probability_of_loss,
        expected_upside=expected_upside,
        expected_downside=expected_downside,
        upside_downside_ratio=ratio,
        issues=ratio_issues,
    )


def calculate_extended_valuation_audits(
    output: ValuationOutput,
    evidence: dict[str, dict[str, Any]],
    financial: FinancialOutput | None = None,
    research: dict[str, Any] | None = None,
) -> ValuationOutput:
    """Run hard audits in ValuationAgent order; repeated pre/post-finalize calls are safe."""
    output.scenario_trigger_audit = build_scenario_trigger_audit(output, evidence)
    output.valuation_method_applicability_audit = build_valuation_method_applicability_audit(
        output, evidence
    )
    output.double_counting_audit = build_double_counting_audit(output, evidence)
    output.nearest_scenario_match = calculate_nearest_scenario_match(output)
    output.risk_reward_audit = build_risk_reward_audit(output)
    output = finalize_deterministic_decision_summary(output, financial, research or {})
    return output


def finalize_valuation_probabilities(output: ValuationOutput) -> ValuationOutput:
    """Normalize the exclusive scenario state and fail closed on any valuation-chain defect."""

    required_names = ("bear", "base", "bull")
    counts = {name: sum(item.name == name for item in output.scenarios) for name in required_names}
    complete = len(output.scenarios) == 3 and all(counts[name] == 1 for name in required_names)
    by_name = {item.name: item for item in output.scenarios}
    base_probability_sum = sum(item.base_probability for item in output.scenarios)
    raw_total = sum(item.base_probability * item.evidence_factor for item in output.scenarios)

    year_paths = [tuple(str(p.year) for p in item.projections) for item in output.scenarios]
    common_horizon = bool(
        complete and year_paths and all(year_paths[0] == years for years in year_paths)
        and len(year_paths[0]) > 0
    )
    terminal_profits = {
        name: by_name[name].projections[-1].net_profit
        for name in required_names if name in by_name and by_name[name].projections
    }
    profit_ordered = len(terminal_profits) == 3 and (
        terminal_profits["bear"] <= terminal_profits["base"] <= terminal_profits["bull"]
    )
    if len(terminal_profits) == 3:
        lower = (terminal_profits["bear"] + terminal_profits["base"]) / 2
        upper = (terminal_profits["base"] + terminal_profits["bull"]) / 2
        definitions = {
            "bear": (
                f"终值净利润≤{lower:.2f}亿元的重大基本面恶化区间",
                [f"terminal net profit <= {lower:.2f}", "与 Base/Bull 不重叠"],
            ),
            "base": (
                f"终值净利润>{lower:.2f}且≤{upper:.2f}亿元的中枢兑现区间",
                [f"{lower:.2f} < terminal net profit <= {upper:.2f}", "与 Bear/Bull 不重叠"],
            ),
            "bull": (
                f"终值净利润>{upper:.2f}亿元的上行超预期区间",
                [f"terminal net profit > {upper:.2f}", "与 Bear/Base 不重叠"],
            ),
        }
    else:
        definitions = {
            "bear": ("终值经营结果落入下行情景区间", ["下行区间边界待有效预测补齐"]),
            "base": ("终值经营结果落入中枢情景区间", ["中枢区间边界待有效预测补齐"]),
            "bull": ("终值经营结果落入上行情景区间", ["上行区间边界待有效预测补齐"]),
        }

    entries: list[ProbabilityAuditEntry] = []
    for item in output.scenarios:
        raw = item.base_probability * item.evidence_factor
        normalized = raw / raw_total if raw_total > 0 else 0
        item.probability = normalized
        if not item.event_definition:
            item.event_definition = definitions[item.name][0]
        if not item.boundary_conditions:
            item.boundary_conditions = definitions[item.name][1]
        years = [str(p.year) for p in item.projections]
        item.horizon = item.horizon or (f"{years[0]}-{years[-1]}" if years else "")
        item.probability_assessment.raw_adjusted_probability = raw
        item.probability_assessment.adjusted_probability = raw
        item.probability_assessment.normalization_denominator = raw_total
        item.probability_assessment.normalized_probability = normalized
        pre_entry = next(
            (entry for entry in output.pre_valuation_probability_audit.scenarios if entry.name == item.name),
            None,
        )
        pre_reasoning = pre_entry.reasoning.strip() if pre_entry else ""
        pre_refs = list(dict.fromkeys(
            (list(pre_entry.probability_basis_refs) + list(pre_entry.evidence_refs))
            if pre_entry else []
        ))
        evidence_ready = bool(
            pre_reasoning and pre_refs
            and output.pre_valuation_probability_audit.evidence_coverage.get(item.name, False)
        )
        raw_step_value = raw if raw <= 1 else None
        raw_step_status = "available" if raw <= 1 and evidence_ready else "blocked"
        entries.append(ProbabilityAuditEntry(
            scenario=item.name,
            event_definition=item.event_definition,
            boundary_conditions=item.boundary_conditions,
            horizon=item.horizon,
            base_probability=item.base_probability,
            evidence_factor=item.evidence_factor,
            raw_adjusted_probability=raw,
            normalization_denominator=raw_total,
            normalized_probability=normalized,
            formula=f"({item.base_probability:.6f}×{item.evidence_factor:.6f})÷{raw_total:.6f}={normalized:.6f}" if raw_total else "分母为0",
            evidence_refs=pre_refs,
            reasoning=pre_reasoning or "Pre-Valuation Probability Audit缺少该情景reasoning",
            adjustment_steps=[
                ProbabilityAdjustmentStep(
                    step_id=f"{item.name}-evidence-adjustment",
                    adjustment_type="evidence_factor",
                    prior_probability=item.base_probability,
                    adjustment_factor=item.evidence_factor,
                    adjusted_probability=raw_step_value,
                    reasoning=(
                        f"沿用Pre-Valuation reasoning：{pre_reasoning}"
                        if pre_reasoning else "Pre-Valuation reasoning不可用"
                    ),
                    formula=f"{item.base_probability:.6f}×{item.evidence_factor:.6f}={raw:.6f}",
                    evidence_refs=pre_refs,
                    status=raw_step_status,
                ),
                ProbabilityAdjustmentStep(
                    step_id=f"{item.name}-normalization",
                    adjustment_type="normalization",
                    prior_probability=raw_step_value,
                    adjustment_factor=(1 / raw_total if raw_total > 0 else None),
                    adjusted_probability=normalized if raw_total > 0 else None,
                    reasoning="raw probability除以三情景raw probability总和；引用同一Pre-Valuation依据",
                    formula=f"{raw:.6f}÷{raw_total:.6f}={normalized:.6f}" if raw_total else "normalization denominator=0",
                    evidence_refs=pre_refs,
                    status=(
                        "available"
                        if raw_total > 0 and raw <= 1 and evidence_ready else "blocked"
                    ),
                ),
            ],
        ))

    probability_sum = sum(item.probability for item in output.scenarios)
    probability_sum_valid = complete and raw_total > 0 and abs(probability_sum - 1) <= 1e-9
    probability_issues: list[str] = list(output.pre_valuation_probability_audit.issues)
    if not output.pre_valuation_probability_audit.passed:
        probability_issues.append("Pre-Valuation Probability Gate未通过")
    if not complete:
        probability_issues.append("必须恰好包含 Bear/Base/Bull 各一个情景")
    if raw_total <= 0:
        probability_issues.append("三情景证据调整权重合计必须大于0")
    if any(item.base_probability * item.evidence_factor > 1 for item in output.scenarios):
        probability_issues.append("至少一个base probability×evidence factor超过100%，两步调整审计无法以概率字段表达")
    incomplete_ledgers = [
        entry.scenario for entry in entries
        if not entry.reasoning.strip() or not entry.evidence_refs
        or len(entry.adjustment_steps) < 2
        or any(
            not step.reasoning.strip() or not step.formula.strip()
            or not step.evidence_refs or step.status != "available"
            for step in entry.adjustment_steps
        )
    ]
    if incomplete_ledgers:
        probability_issues.append(
            "以下情景的base→evidence adjusted→normalized adjustment ledger不完整或被阻断："
            + ", ".join(incomplete_ledgers)
        )
    if complete and abs(base_probability_sum - 1) > 1e-9:
        probability_issues.append(f"三情景基础概率必须合计100%，当前为{base_probability_sum:.6%}")
    if not common_horizon:
        probability_issues.append("Bear/Base/Bull 必须使用完全相同的预测年份")
    if not profit_ordered:
        probability_issues.append("Bear/Base/Bull 终值净利润未按悲观到乐观排序")
    output.probability_audit = ProbabilityAuditReport(
        passed=bool(output.pre_valuation_probability_audit.passed and not probability_issues and probability_sum_valid),
        mutually_exclusive=bool(output.pre_valuation_probability_audit.mutually_exclusive),
        exhaustive=bool(output.pre_valuation_probability_audit.exhaustive),
        base_probability_sum=base_probability_sum,
        probability_sum=probability_sum,
        entries=entries,
        issues=probability_issues,
    )

    targets = {name: by_name[name].target_price for name in required_names if name in by_name}
    complete_targets = complete and all(targets.get(name) is not None for name in required_names)
    ordered_targets = bool(
        complete_targets and targets["bear"] <= targets["base"] <= targets["bull"]
    )
    issues = list(probability_issues)
    if not output.forecast_reasonableness_audit.passed:
        issues.append("Forecast Reasonableness Gate未通过，预测合理性层不成立，禁止进入估值")
    if not output.comparable_valuation_evidence_audit.passed:
        issues.append("Comparable Valuation Evidence Gate未通过，至少一个终值情景没有可靠倍数证据")
        issues.extend(output.comparable_valuation_evidence_audit.issues)
    if not output.scenario_trigger_audit.passed:
        issues.append("Scenario Trigger Audit未通过，业务变化未完整映射到有证据的财务变量触发项")
        issues.extend(output.scenario_trigger_audit.issues)
    if not output.valuation_method_applicability_audit.passed:
        issues.append("Valuation Method Applicability Audit未通过")
        issues.extend(output.valuation_method_applicability_audit.issues)
    if not output.double_counting_audit.passed:
        issues.append("Double Counting Audit未通过，盈利改善与倍数提高存在未解决的证据复用风险")
        issues.extend(output.double_counting_audit.issues)
    if not complete_targets:
        issues.append("三个情景必须都有有效目标价；禁止对子集重新归一化")
    elif not ordered_targets:
        issues.append(
            f"目标价经济排序失败：Bear {targets['bear']:.4f}、Base {targets['base']:.4f}、Bull {targets['bull']:.4f}"
        )
    if output.current_price in {None, 0}:
        issues.append("当前价格缺失或为0，无法计算 Expected Return")

    comparability = _build_valuation_comparability_audit(
        output, by_name, complete, complete_targets, ordered_targets
    )
    issues.extend(comparability.issues)

    first_implied = next((item for item in output.reverse_valuation if item.implied_net_profit is not None), None)
    rows: list[ScenarioValuationAudit] = []
    for name in required_names:
        scenario = by_name.get(name)
        if not scenario:
            continue
        terminal = scenario.projections[-1] if scenario.projections else None
        comparison_projection = next(
            (p for p in scenario.projections if first_implied and str(p.year) == str(first_implied.year)),
            None,
        )
        contribution = (
            scenario.probability * scenario.target_price
            if scenario.target_price is not None else None
        )
        valuation_chain = (
            f"{terminal.valuation_formula}；EV={terminal.enterprise_value}；"
            f"Net Debt={terminal.net_debt}；Equity Value={terminal.equity_value}；"
            f"Target Price={scenario.target_price}"
            if terminal else "终值预测不可用"
        )
        rows.append(ScenarioValuationAudit(
            scenario=name,
            target_year=scenario.target_year,
            revenue=scenario.revenue,
            ebit=terminal.ebit if terminal else None,
            ebitda=terminal.ebitda if terminal else None,
            free_cash_flow=terminal.free_cash_flow if terminal else None,
            net_profit=scenario.net_profit,
            eps=terminal.eps if terminal else None,
            valuation_method=scenario.valuation_method,
            method_selection_reason=terminal.method_selection_reason if terminal else "",
            valuation_multiple=scenario.valuation_multiple,
            enterprise_value=scenario.enterprise_value,
            net_debt=scenario.net_debt,
            equity_value=scenario.equity_value,
            implied_market_cap=scenario.implied_market_cap,
            target_price=scenario.target_price,
            probability=scenario.probability,
            weighted_contribution=contribution,
            model_implied_profit=first_implied.implied_net_profit if first_implied else None,
            profit_gap_vs_model_implied=(
                comparison_projection.net_profit - first_implied.implied_net_profit
                if comparison_projection and first_implied and first_implied.implied_net_profit is not None else None
            ),
            valuation_chain=valuation_chain,
        ))

    passed = bool(
        output.pre_valuation_probability_audit.passed
        and output.probability_audit.passed and common_horizon and complete_targets
        and ordered_targets and output.current_price not in {None, 0}
        and output.forecast_reasonableness_audit.passed
        and output.comparable_valuation_evidence_audit.passed
        and output.scenario_trigger_audit.passed
        and output.valuation_method_applicability_audit.passed
        and output.double_counting_audit.passed
        and comparability.passed and comparability.all_identity_checks_passed
    )
    if passed:
        weighted_target = sum(item.probability * item.target_price for item in output.scenarios)  # type: ignore[operator]
        weighted_cap = sum(item.probability * item.implied_market_cap for item in output.scenarios) if all(
            item.implied_market_cap is not None for item in output.scenarios
        ) else None
        expected_return = weighted_target / output.current_price - 1  # type: ignore[operator]
        horizon = len(year_paths[0])
        annualized = (1 + expected_return) ** (1 / max(1, horizon)) - 1 if expected_return > -1 else expected_return
        output.probability_weighted_market_cap = weighted_cap
        output.probability_weighted_target_price = weighted_target
        output.expected_price_return = expected_return
        output.annualized_expected_return = annualized
    else:
        weighted_target = expected_return = annualized = None
        output.probability_weighted_market_cap = None
        output.probability_weighted_target_price = None
        output.expected_price_return = None
        output.annualized_expected_return = None
        for issue in issues:
            warning = f"模型审计阻断：{issue}"
            if warning not in output.limitations:
                output.limitations.append(warning)

    weighted_formula = " + ".join(
        f"{row.probability:.6f}×{row.target_price:.6f}"
        for row in rows if row.target_price is not None
    )
    if weighted_target is not None:
        weighted_formula += f" = {weighted_target:.6f}"
    output.valuation_audit = ValuationAuditReport(
        passed=passed,
        complete_scenarios=complete,
        common_horizon=common_horizon,
        complete_targets=complete_targets,
        ordered_targets=ordered_targets,
        probability_sum_valid=probability_sum_valid,
        valuation_methods_comparable=comparability.passed,
        all_identity_checks_passed=comparability.all_identity_checks_passed,
        comparability_audit=comparability,
        scenario_rows=rows,
        probability_weighted_target_price=weighted_target,
        expected_price_return=expected_return,
        annualized_expected_return=annualized,
        weighted_target_formula=weighted_formula,
        expected_return_formula=(
            f"{weighted_target:.6f}÷{output.current_price:.6f}-1={expected_return:.6%}"
            if weighted_target is not None and expected_return is not None else "审计未通过，Expected Return 已禁用"
        ),
        issues=list(dict.fromkeys(issues)),
    )
    # Risk/reward and deterministic decision are deliberately recomputed only
    # after the final valuation audit is settled; both functions are idempotent.
    output.risk_reward_audit = build_risk_reward_audit(output)
    output = finalize_deterministic_decision_summary(output)
    return build_gap_audits(output)


def calculate_valuation_engine(
    output: ValuationOutput, research: dict[str, Any], financial: FinancialOutput
) -> ValuationOutput:
    base_revenue, _, base_period = _financial_baseline(financial)
    snapshot = research.get("market_snapshot") or {}
    market_cap, pb = snapshot.get("market_cap"), snapshot.get("pb")
    current_price, shares = snapshot.get("price"), snapshot.get("shares_outstanding")
    report = snapshot.get("consistency_report") or {}
    valuation_allowed = bool(report and report.get("valuation_allowed") is True)
    if base_revenue is None or base_revenue <= 0:
        valuation_allowed = False
    output.current_price, output.shares_outstanding, output.current_market_cap = current_price, shares, market_cap
    derived_book_value = market_cap / pb if market_cap is not None and pb not in {None, 0} else 0
    scenario_scales = {"bear": 0.70, "base": 1.00, "bull": 1.20}
    for scenario in output.scenarios:
        previous_revenue = base_revenue
        fallback_scale = scenario_scales.get(scenario.name, 1.0)
        scenario.projections.sort(key=lambda item: int(item.year) if str(item.year).isdigit() else 9999)
        for item in scenario.projections:
            item.revenue = previous_revenue * (1 + item.revenue_growth) if previous_revenue is not None else 0
            item.gross_profit = item.revenue * item.gross_margin
            item.rd_expense = item.revenue * item.rd_expense_ratio
            item.selling_expense = item.revenue * item.selling_expense_ratio
            item.admin_expense = item.revenue * item.admin_expense_ratio
            item.other_operating_expense = item.revenue * item.other_operating_expense_ratio
            item.operating_profit = item.gross_profit - item.rd_expense - item.selling_expense - item.admin_expense - item.other_operating_expense
            item.financial_expense = item.revenue * item.financial_expense_ratio
            item.other_income = item.revenue * item.other_income_ratio
            item.pre_tax_profit = item.operating_profit - item.financial_expense + item.other_income
            item.tax = max(item.pre_tax_profit, 0) * item.tax_rate
            post_tax = item.pre_tax_profit - item.tax
            item.minority_interest = max(post_tax, 0) * item.minority_interest_ratio
            item.net_profit = post_tax - item.minority_interest
            depreciation = item.revenue * item.depreciation_amortization_ratio
            item.ebitda = item.operating_profit + depreciation
            item.eps = item.net_profit / shares if shares not in {None, 0} else None
            item.net_debt = item.debt - item.cash
            item.book_value = item.book_value or derived_book_value
            item.enterprise_value, item.implied_market_cap = None, None
            item.valuation_method, item.valuation_multiple = "unavailable", 0
            item.method_warning = ""
            if valuation_allowed and item.net_profit > 0 and item.pe_multiple > 0:
                item.valuation_method, item.valuation_multiple = "pe", item.pe_multiple
                item.implied_market_cap = item.net_profit * item.pe_multiple
                item.valuation_formula = "股权价值=程序计算净利润×PE假设"
            elif valuation_allowed:
                if item.net_profit <= 0:
                    item.method_warning = "净利润≤0，程序已禁止PE并自动切换非PE估值。"
                for method in dict.fromkeys([item.preferred_loss_method, "ev_ebitda", "ev_sales", "pb"]):
                    if method == "ev_ebitda" and item.ebitda > 0 and item.ev_ebitda_multiple > 0:
                        item.valuation_method, item.valuation_multiple = method, item.ev_ebitda_multiple
                        item.enterprise_value = item.ebitda * item.ev_ebitda_multiple
                        item.implied_market_cap = item.enterprise_value - item.net_debt
                        item.valuation_formula = "EV=EBITDA×EV/EBITDA；股权价值=EV-净债务"
                        break
                    if method == "ev_sales" and item.revenue > 0:
                        observed_ev_sales = (
                            (market_cap + item.net_debt) / base_revenue
                            if market_cap is not None and base_revenue not in {None, 0} else 0
                        )
                        multiple = item.ev_sales_multiple or max(0.1, observed_ev_sales * fallback_scale)
                        if multiple > 0:
                            item.valuation_method, item.valuation_multiple = method, multiple
                            item.enterprise_value = item.revenue * multiple
                            item.implied_market_cap = item.enterprise_value - item.net_debt
                            item.valuation_formula = "EV=收入×EV/Sales；股权价值=EV-净债务"
                            if item.ev_sales_multiple <= 0:
                                item.method_warning += f" 未提供非PE倍数，程序以当前EV/Sales×{fallback_scale:.2f}作为{scenario.name}情景兜底。"
                            break
                    if method == "pb" and item.book_value > 0:
                        multiple = item.pb_multiple or ((pb or 0) * fallback_scale)
                        if multiple > 0:
                            item.valuation_method, item.valuation_multiple = method, multiple
                            item.implied_market_cap = item.book_value * multiple
                            item.valuation_formula = "股权价值=账面净资产×PB"
                            if item.pb_multiple <= 0:
                                item.method_warning += f" 未提供PB倍数，程序以当前PB×{fallback_scale:.2f}作为{scenario.name}情景兜底。"
                            break
            else:
                item.method_warning = "市场数据一致性检查未通过，程序禁止进入估值计算。"
            if item.implied_market_cap is not None and item.implied_market_cap < 0:
                item.method_warning += " 股权价值为负，结果不可用。"
                item.implied_market_cap, item.valuation_method = None, "unavailable"
            checks = [
                _check_close("收入×毛利率=毛利", item.gross_profit, item.revenue * item.gross_margin),
                _check_close("毛利-期间费用=营业利润", item.operating_profit, item.gross_profit - item.rd_expense - item.selling_expense - item.admin_expense - item.other_operating_expense),
                _check_close("营业利润+折旧摊销=EBITDA", item.ebitda, item.operating_profit + depreciation),
                _check_close("税前利润-税-少数股东=净利润", item.net_profit, item.pre_tax_profit - item.tax - item.minority_interest),
            ]
            if item.valuation_method in {"ev_ebitda", "ev_sales"} and item.enterprise_value is not None and item.implied_market_cap is not None:
                checks.append(_check_close("EV-净债务=股权价值", item.implied_market_cap, item.enterprise_value - item.net_debt))
            if item.implied_market_cap is not None and shares not in {None, 0}:
                target_price = item.implied_market_cap / shares
                checks.append(_check_close("股权价值=目标价×总股本", item.implied_market_cap, target_price * shares))
            item.calculation_checks = checks
            item.calculation_valid = all(issue.passed for issue in checks)
            if not item.calculation_valid:
                item.implied_market_cap, item.valuation_method = None, "unavailable"
            item.assumptions = list(dict.fromkeys(item.assumptions + [
                f"收入增长率 {item.revenue_growth:.1%}", f"毛利率 {item.gross_margin:.1%}",
                f"研发/销售/管理费用率 {item.rd_expense_ratio:.1%}/{item.selling_expense_ratio:.1%}/{item.admin_expense_ratio:.1%}",
                f"财务基期 {base_period or '不可用'}",
            ]))
            previous_revenue = item.revenue
        if scenario.projections:
            target = scenario.projections[-1]
            scenario.target_year, scenario.revenue, scenario.net_profit = target.year, target.revenue, target.net_profit
            scenario.valuation_method, scenario.valuation_multiple = target.valuation_method, target.valuation_multiple
            scenario.implied_market_cap = target.implied_market_cap
            scenario.target_price = target.implied_market_cap / shares if target.implied_market_cap is not None and shares not in {None, 0} else None
            scenario.price_return = scenario.target_price / current_price - 1 if scenario.target_price is not None and current_price not in {None, 0} else None
            horizon = max(1, len(scenario.projections))
            scenario.annualized_price_return = (
                (1 + scenario.price_return) ** (1 / horizon) - 1
                if scenario.price_return is not None and scenario.price_return > -1 else scenario.price_return
            )
    base = next((item for item in output.scenarios if item.name == "base" and item.projections), None)
    if base:
        target = base.projections[-1]
        output.financial_tree = LogicTree(
            tree_type="financial", root_id="F6",
            nodes=[
                LogicTreeNode(node_id="F1", label="收入", metric="Revenue", value=target.revenue, unit="亿元", period=target.year),
                LogicTreeNode(node_id="F2", label="毛利", metric="Gross Profit", value=target.gross_profit, unit="亿元", period=target.year, depends_on=["F1"]),
                LogicTreeNode(node_id="F3", label="营业利润", metric="Operating Profit", value=target.operating_profit, unit="亿元", period=target.year, depends_on=["F2"]),
                LogicTreeNode(node_id="F4", label="EBITDA", metric="EBITDA", value=target.ebitda, unit="亿元", period=target.year, depends_on=["F3"]),
                LogicTreeNode(node_id="F5", label="净利润", metric="Net Profit", value=target.net_profit, unit="亿元", period=target.year, depends_on=["F3"]),
                LogicTreeNode(node_id="F6", label="EPS", metric="EPS", value=target.eps, unit="元/股", period=target.year, depends_on=["F5"]),
            ],
            summary="Base情景程序化财务兑现树：收入→毛利→营业利润/EBITDA→净利润→EPS",
        )
    if base_revenue is None:
        output.limitations.append("缺少可靠年度财务基期收入，程序化收入路径无法计算。")
    if not valuation_allowed:
        output.limitations.append("市场数据一致性检查未通过，所有估值与目标价已禁用。")
    return output


def _annual_historical_periods(financial: FinancialOutput) -> list[Any]:
    annuals = []
    for item in financial.historical:
        text = str(item.period)
        match = re.search(r"20\d{2}", text)
        if not match:
            continue
        if re.fullmatch(r"20\d{2}", text.strip()) or "12-31" in text or "年报" in text or "FY" in text.upper():
            annuals.append((int(match.group()), item))
    return [item for _, item in sorted(annuals, key=lambda pair: pair[0])[-5:]]


def _valid_forecast_evidence_refs(
    refs: list[str], evidence: dict[str, dict[str, Any]]
) -> list[str]:
    valid: list[str] = []
    for ref in dict.fromkeys(refs):
        item = evidence.get(ref)
        if not item or item.get("source_grade") not in {"A", "B"}:
            continue
        if item.get("temporal_scope") == "future" or item.get("source_type") == "collection_error":
            continue
        valid.append(ref)
    return valid


def _projection_variable_refs(item: FinancialBridge, variable: str) -> list[str]:
    mapping = item.assumption_evidence_refs or {}
    dependencies = {
        "revenue_growth": ["revenue_growth"],
        "gross_margin": ["gross_margin"],
        "ebit_margin": [
            "ebit_margin", "gross_margin", "rd_expense_ratio", "selling_expense_ratio",
            "admin_expense_ratio", "other_operating_expense_ratio",
        ],
        "ebitda_margin": [
            "ebitda_margin", "gross_margin", "rd_expense_ratio", "selling_expense_ratio",
            "admin_expense_ratio", "other_operating_expense_ratio", "depreciation_amortization_ratio",
        ],
        "net_margin": [
            "net_margin", "gross_margin", "rd_expense_ratio", "selling_expense_ratio",
            "admin_expense_ratio", "other_operating_expense_ratio", "financial_expense_ratio",
            "other_income_ratio", "tax_rate", "minority_interest_ratio",
        ],
        "depreciation_amortization_ratio": ["depreciation_amortization_ratio"],
    }
    return list(dict.fromkeys(
        ref for key in dependencies.get(variable, [variable]) for ref in mapping.get(key, [])
    ))


def _projection_variable_value(item: FinancialBridge, variable: str) -> float | None:
    if variable == "revenue_growth":
        return item.revenue_growth
    if variable == "gross_margin":
        return item.gross_margin
    if variable == "ebit_margin":
        return item.ebit / item.revenue if item.revenue else None
    if variable == "ebitda_margin":
        return item.ebitda / item.revenue if item.revenue else None
    if variable == "net_margin":
        return item.net_profit / item.revenue if item.revenue else None
    if variable == "depreciation_amortization_ratio":
        return item.depreciation_amortization_ratio
    value = getattr(item, variable, None)
    return float(value) if value is not None else None


def calculate_forecast_reasonableness_audit(
    output: ValuationOutput,
    financial: FinancialOutput,
    evidence: dict[str, dict[str, Any]],
) -> ValuationOutput:
    """Gate forecasts on historical anchors, explainable P&L bridges and economic scenario separation."""
    annuals = _annual_historical_periods(financial)
    labels = {
        "revenue_growth": "Revenue Growth",
        "gross_margin": "Gross Margin",
        "ebit_margin": "EBIT Margin",
        "ebitda_margin": "EBITDA Margin",
        "net_margin": "Net Margin",
        "depreciation_amortization_ratio": "D&A / Revenue",
    }
    historical: dict[str, list[tuple[str, float, list[str]]]] = {key: [] for key in labels}
    for index, period in enumerate(annuals):
        period_text = str(period.period)
        period_refs = list(period.evidence_refs)
        field_refs = period.field_evidence_refs or {}
        direct_values = {
            "gross_margin": period.gross_margin,
            "ebit_margin": period.ebit_margin,
            "ebitda_margin": period.ebitda_margin,
            "net_margin": period.net_margin,
            "depreciation_amortization_ratio": period.depreciation_amortization_ratio,
        }
        for variable, value in direct_values.items():
            if value is not None:
                refs = list(dict.fromkeys(field_refs.get(variable, []) + period_refs))
                historical[variable].append((period_text, float(value), refs))
        if index > 0:
            prior = annuals[index - 1]
            if period.revenue is not None and prior.revenue not in {None, 0}:
                growth = period.revenue / prior.revenue - 1
                refs = list(dict.fromkeys(
                    field_refs.get("revenue", []) + prior.field_evidence_refs.get("revenue", [])
                    + period_refs + list(prior.evidence_refs)
                ))
                historical["revenue_growth"].append((period_text, growth, refs))

    anchors: list[HistoricalAnchorAuditItem] = []
    anchor_by_variable: dict[str, HistoricalAnchorAuditItem] = {}
    for variable, label in labels.items():
        observations = historical[variable]
        values = [value for _, value, _ in observations]
        refs = list(dict.fromkeys(ref for _, _, item_refs in observations for ref in item_refs))
        anchor = HistoricalAnchorAuditItem(
            variable=variable,
            label=label,
            periods=[period for period, _, _ in observations],
            historical_values=values,
            historical_min=min(values) if values else None,
            historical_max=max(values) if values else None,
            historical_median=median(values) if values else None,
            available=bool(values),
            basis=(
                "最近最多5个年度实际值；历史EBIT使用营业利润proxy，EBITDA=EBIT proxy+D&A"
                if variable in {"ebit_margin", "ebitda_margin"}
                else "最近最多5个年度实际值"
            ),
            evidence_refs=refs,
            issue="" if values else f"缺少{label}年度历史实际数据，不能验证预测锚点",
        )
        anchors.append(anchor)
        anchor_by_variable[variable] = anchor

    forecast_entries: list[ForecastReasonablenessEntry] = []
    pad_floor = {
        "revenue_growth": 0.05,
        "gross_margin": 0.02,
        "ebit_margin": 0.02,
        "ebitda_margin": 0.02,
        "net_margin": 0.02,
        "depreciation_amortization_ratio": 0.01,
    }
    for scenario in output.scenarios:
        for projection in scenario.projections:
            for variable, label in labels.items():
                value = _projection_variable_value(projection, variable)
                anchor = anchor_by_variable[variable]
                refs = _projection_variable_refs(projection, variable)
                valid_refs = _valid_forecast_evidence_refs(refs, evidence)
                lower = upper = None
                outside = True
                if anchor.available and anchor.historical_min is not None and anchor.historical_max is not None:
                    span = anchor.historical_max - anchor.historical_min
                    pad = max(pad_floor[variable], span * 0.5)
                    lower, upper = anchor.historical_min - pad, anchor.historical_max + pad
                    outside = value is None or value < lower or value > upper
                if not outside:
                    status = "within_history"
                    reason = "位于扩展历史区间内；该区间仅用于异常识别，不是预测上限/下限"
                elif valid_refs:
                    status = "supported_structural_change"
                    reason = "突破或缺少历史区间，但存在非未来A/B级变量级证据，作为结构性变化保留"
                else:
                    status = "unsupported_forecast"
                    reason = (
                        "明显突破扩展历史区间且缺少变量级Evidence Ref"
                        if anchor.available else "历史锚点不可用且缺少变量级Evidence Ref"
                    )
                forecast_entries.append(ForecastReasonablenessEntry(
                    scenario=scenario.name,
                    year=str(projection.year),
                    variable=variable,
                    label=label,
                    forecast_value=value,
                    historical_min=anchor.historical_min,
                    historical_max=anchor.historical_max,
                    reasonableness_lower=lower,
                    reasonableness_upper=upper,
                    evidence_refs=refs,
                    valid_evidence_refs=valid_refs,
                    status=status,
                    reason=reason,
                ))

    margin_bridges: list[MarginBridgeAuditItem] = []
    for scenario in output.scenarios:
        for item in scenario.projections:
            depreciation = item.ebitda - item.ebit
            formula_checks = [
                isclose_value(item.gross_profit, item.revenue * item.gross_margin),
                isclose_value(item.ebit, item.gross_profit - item.rd_expense - item.selling_expense - item.admin_expense - item.other_operating_expense),
                isclose_value(item.ebitda, item.ebit + depreciation),
                isclose_value(item.pre_tax_profit, item.ebit - item.financial_expense + item.other_income),
                isclose_value(item.net_profit, item.pre_tax_profit - item.tax - item.minority_interest),
            ]
            issues: list[str] = []
            if not all(formula_checks):
                issues.append("Revenue→Gross Profit→EBIT→EBITDA→EBT→Net Profit 数学桥不一致")
            if not item.assumptions:
                issues.append("未解释利润率变化来源")
            da_entry = next((
                entry for entry in forecast_entries
                if entry.scenario == scenario.name and entry.year == str(item.year)
                and entry.variable == "depreciation_amortization_ratio"
            ), None)
            if da_entry and da_entry.status == "unsupported_forecast":
                issues.append("EBITDA-EBIT对应的D&A/Revenue缺少历史锚点或变量级证据")
            margin_bridges.append(MarginBridgeAuditItem(
                scenario=scenario.name,
                year=str(item.year),
                revenue=item.revenue,
                gross_profit=item.gross_profit,
                ebit=item.ebit,
                depreciation_amortization=depreciation,
                ebitda=item.ebitda,
                ebt=item.pre_tax_profit,
                net_profit=item.net_profit,
                gross_margin=item.gross_margin,
                ebit_margin=item.ebit / item.revenue if item.revenue else 0,
                ebitda_margin=item.ebitda / item.revenue if item.revenue else 0,
                net_margin=item.net_profit / item.revenue if item.revenue else 0,
                depreciation_amortization_ratio=depreciation / item.revenue if item.revenue else 0,
                formula_passed=all(formula_checks),
                explanation="；".join(item.assumptions) or "未提供解释",
                evidence_refs=list(dict.fromkeys(
                    ref for refs in item.assumption_evidence_refs.values() for ref in refs
                )),
                issues=issues,
            ))

    separation = _build_scenario_separation_audit(output, evidence)
    margin_expansion = _build_margin_expansion_evidence_audit(output, evidence)
    unsupported = [
        f"{entry.scenario} {entry.year} {entry.label}: {entry.reason}"
        for entry in forecast_entries if entry.status == "unsupported_forecast"
    ]
    historical_anchor_passed = all(anchor.available for anchor in anchors)
    margin_bridge_passed = bool(margin_bridges and all(
        item.formula_passed and not item.issues for item in margin_bridges
    ))
    issues = [anchor.issue for anchor in anchors if anchor.issue]
    issues.extend(unsupported)
    issues.extend(separation.issues)
    issues.extend(margin_expansion.issues)
    if not margin_bridge_passed:
        issues.append("Margin Bridge Audit未通过")
    passed = bool(
        historical_anchor_passed and margin_bridge_passed
        and separation.passed and margin_expansion.passed and not unsupported
    )
    evidence_needed = list(dict.fromkeys(
        [f"补充{anchor.label}年度实际值及对应财报Evidence Ref" for anchor in anchors if not anchor.available]
        + [f"补充{entry.scenario} {entry.year} {entry.label}结构性变化的变量级A/B证据" for entry in forecast_entries if entry.status == "unsupported_forecast"]
        + (["补充Bull相对Base经营变量差异及其Evidence Ref"] if not separation.passed else [])
        + margin_expansion.evidence_needed
    ))
    output.forecast_reasonableness_audit = ForecastReasonablenessAudit(
        passed=passed,
        valuation_allowed=passed,
        historical_anchor_passed=historical_anchor_passed,
        margin_bridge_passed=margin_bridge_passed,
        scenario_separation_passed=separation.passed,
        margin_expansion_evidence_passed=margin_expansion.passed,
        historical_anchors=anchors,
        forecast_entries=forecast_entries,
        margin_bridges=margin_bridges,
        margin_expansion_evidence=margin_expansion,
        scenario_separation=separation,
        unsupported_forecasts=unsupported,
        key_driver=separation.key_driver,
        evidence_needed=evidence_needed,
        failed_layer="" if passed else "forecast_reasonableness",
        issues=list(dict.fromkeys(issues)),
    )
    return output


def isclose_value(actual: float, expected: float, tolerance: float = 1e-8) -> bool:
    scale = max(1.0, abs(actual), abs(expected))
    return isfinite(actual) and isfinite(expected) and abs(actual - expected) <= tolerance * scale


def _build_scenario_separation_audit(
    output: ValuationOutput, evidence: dict[str, dict[str, Any]]
) -> ScenarioSeparationAudit:
    by_name = {scenario.name: scenario for scenario in output.scenarios}
    if not all(name in by_name for name in ("bear", "base", "bull")):
        return ScenarioSeparationAudit(issues=["缺少Bear/Base/Bull，无法审计情景分离"])
    variables = {
        "revenue_growth": "Revenue Growth",
        "gross_margin": "Gross Margin",
        "rd_expense_ratio": "R&D / Revenue",
        "selling_expense_ratio": "Selling Expense / Revenue",
        "admin_expense_ratio": "Admin Expense / Revenue",
        "other_operating_expense_ratio": "Other Opex / Revenue",
        "depreciation_amortization_ratio": "D&A / Revenue",
        "financial_expense_ratio": "Financial Expense / Revenue",
        "other_income_ratio": "Other Income / Revenue",
        "tax_rate": "Tax Rate",
    }
    paths = {name: {str(item.year): item for item in by_name[name].projections} for name in by_name}
    years = sorted(set(paths["bear"]) & set(paths["base"]) & set(paths["bull"]))
    drivers: list[ScenarioSeparationDriver] = []
    differentiated: list[str] = []
    for year in years:
        items = {name: paths[name][year] for name in ("bear", "base", "bull")}
        for variable, label in variables.items():
            values = {name: float(getattr(item, variable)) for name, item in items.items()}
            if max(values.values()) - min(values.values()) <= 1e-9:
                continue
            differentiated.append(variable)
            refs = list(dict.fromkeys(
                ref for item in items.values()
                for ref in item.assumption_evidence_refs.get(variable, [])
            ))
            supported = all(
                _valid_forecast_evidence_refs(item.assumption_evidence_refs.get(variable, []), evidence)
                for item in items.values()
            )
            bull = items["bull"]
            base = items["base"]
            after_tax = max(0.0, 1 - bull.tax_rate)
            if variable == "revenue_growth":
                base_margin = base.net_profit / base.revenue if base.revenue else 0
                impact = (bull.revenue - base.revenue) * abs(base_margin)
            elif variable == "gross_margin":
                impact = (bull.gross_margin - base.gross_margin) * bull.revenue * after_tax
            elif variable in {"rd_expense_ratio", "selling_expense_ratio", "admin_expense_ratio", "other_operating_expense_ratio", "financial_expense_ratio"}:
                impact = -(getattr(bull, variable) - getattr(base, variable)) * bull.revenue * after_tax
            elif variable == "other_income_ratio":
                impact = (bull.other_income_ratio - base.other_income_ratio) * bull.revenue * after_tax
            elif variable == "tax_rate":
                impact = -bull.pre_tax_profit * (bull.tax_rate - base.tax_rate)
            else:
                impact = 0.0
            drivers.append(ScenarioSeparationDriver(
                year=year,
                variable=variable,
                label=label,
                bear_value=values["bear"],
                base_value=values["base"],
                bull_value=values["bull"],
                bear_vs_base=values["bear"] - values["base"],
                bull_vs_base=values["bull"] - values["base"],
                estimated_bull_vs_base_profit_impact=impact,
                evidence_refs=refs,
                supported=supported,
                explanation=(
                    "三情景变量差异具有逐变量Evidence Ref"
                    if supported else "变量差异缺少Bear/Base/Bull逐情景Evidence Ref"
                ),
            ))
    operational = {
        "revenue_growth", "gross_margin", "rd_expense_ratio", "selling_expense_ratio",
        "admin_expense_ratio", "other_operating_expense_ratio", "financial_expense_ratio", "other_income_ratio",
    }
    terminal_base = by_name["base"].projections[-1] if by_name["base"].projections else None
    terminal_bull = by_name["bull"].projections[-1] if by_name["bull"].projections else None
    profit_improvement = (
        terminal_bull.net_profit - terminal_base.net_profit
        if terminal_base and terminal_bull else None
    )
    key = max(drivers, key=lambda item: abs(item.estimated_bull_vs_base_profit_impact or 0), default=None)
    issues: list[str] = []
    if not set(differentiated) & operational:
        issues.append("Bear/Base/Bull缺少可追溯的经营变量差异，不能只依赖最终利润或估值差异")
    unsupported_drivers = [driver for driver in drivers if not driver.supported]
    if unsupported_drivers:
        issues.append("部分情景分离变量缺少逐变量Evidence Ref")
    if profit_improvement is not None and profit_improvement > 0 and not any(
        driver.supported and (driver.estimated_bull_vs_base_profit_impact or 0) > 0 for driver in drivers
    ):
        issues.append("Bull相对Base的利润改善无法追溯到有证据的经营假设")
    passed = bool(drivers and set(differentiated) & operational and not issues)
    return ScenarioSeparationAudit(
        passed=passed,
        differentiated_variables=list(dict.fromkeys(differentiated)),
        drivers=drivers,
        bull_vs_base_profit_improvement=profit_improvement,
        key_driver=(
            f"{key.year} {key.label}，估算Bull相对Base利润影响{key.estimated_bull_vs_base_profit_impact:+.2f}亿元"
            if key else "无法识别关键财务驱动"
        ),
        issues=issues,
    )


def _build_margin_expansion_evidence_audit(
    output: ValuationOutput,
    evidence: dict[str, dict[str, Any]],
) -> MarginExpansionEvidenceAudit:
    """Trace Bull-vs-Base gross-margin expansion through profit and valuation."""
    by_name = {scenario.name: scenario for scenario in output.scenarios}
    if not all(name in by_name for name in ("base", "bull")):
        return MarginExpansionEvidenceAudit(
            issues=["缺少Base或Bull情景，无法执行Margin Expansion Evidence Audit"],
            evidence_needed=["补齐Base/Bull同年预测与毛利率经营驱动"],
        )
    base_path = {str(item.year): item for item in by_name["base"].projections}
    bull_path = {str(item.year): item for item in by_name["bull"].projections}
    years = sorted(set(base_path) & set(bull_path))
    terminal_improvement = None
    if years:
        terminal_improvement = bull_path[years[-1]].net_profit - base_path[years[-1]].net_profit
    items: list[MarginExpansionEvidenceAuditItem] = []
    issues: list[str] = []
    needed: list[str] = []
    any_primary = False
    for year in years:
        base = base_path[year]
        bull = bull_path[year]
        delta = bull.gross_margin - base.gross_margin
        if delta <= 1e-9:
            continue
        gross_impact = delta * bull.revenue
        after_tax = max(0.0, 1 - bull.tax_rate) * max(0.0, 1 - bull.minority_interest_ratio)
        net_impact = gross_impact * after_tax
        year_profit_improvement = bull.net_profit - base.net_profit
        primary = bool(year_profit_improvement > 0 and net_impact >= 0.5 * year_profit_improvement)
        material = bool(delta >= 0.01 or primary)
        any_primary = any_primary or primary
        drivers = list(bull.margin_expansion_drivers)
        driver_sum = sum(driver.gross_margin_impact for driver in drivers)
        valid_refs: list[str] = []
        driver_evidence_ok = bool(drivers)
        for driver in drivers:
            refs = _valid_forecast_evidence_refs(driver.evidence_refs, evidence)
            valid_refs.extend(refs)
            if not driver.description.strip() or not refs:
                driver_evidence_ok = False
        tolerance = max(0.002, abs(delta) * 0.20)
        impact_explained = bool(drivers and abs(driver_sum - delta) <= tolerance)
        row_issues: list[str] = []
        if material and not drivers:
            row_issues.append("重大Bull毛利率扩张缺少结构化经营驱动")
        if material and drivers and not impact_explained:
            row_issues.append(
                f"经营驱动毛利率影响合计{driver_sum:.2%}不能解释Bull-Base毛利率差{delta:.2%}"
            )
        if material and not driver_evidence_ok:
            row_issues.append("至少一个毛利率经营驱动缺少变量级当前有效A/B级Evidence Ref")
        supported = bool(not material or (drivers and impact_explained and driver_evidence_ok))
        if not supported:
            issues.extend(f"{year}: {message}" for message in row_issues)
            needed.append(
                f"补充Bull {year}产品结构/业务占比/价格/成本/规模效应/市场份额驱动、量化毛利率影响及A/B级Evidence Ref"
            )
        valuation_impact = (
            by_name["bull"].target_price - by_name["base"].target_price
            if None not in {by_name["bull"].target_price, by_name["base"].target_price} else None
        )
        items.append(MarginExpansionEvidenceAuditItem(
            year=year,
            base_gross_margin=base.gross_margin,
            bull_gross_margin=bull.gross_margin,
            gross_margin_delta=delta,
            material=material,
            primary_profit_driver=primary,
            driver_impact_sum=driver_sum,
            gross_profit_impact=gross_impact,
            ebit_impact=gross_impact,
            net_profit_impact=net_impact,
            valuation_impact=valuation_impact,
            drivers=drivers,
            valid_evidence_refs=list(dict.fromkeys(valid_refs)),
            supported=supported,
            causal_chain=(
                f"Business Driver ({driver_sum:.2%}) → Gross Margin Δ {delta:.2%} → "
                f"Gross Profit/EBIT Δ {gross_impact:.2f}亿元 → Net Profit Δ {net_impact:.2f}亿元 → Valuation"
            ),
            issues=row_issues,
        ))
    passed = not issues
    return MarginExpansionEvidenceAudit(
        passed=passed,
        items=items,
        bull_profit_improvement=terminal_improvement,
        gross_margin_is_primary_driver=any_primary,
        issues=list(dict.fromkeys(issues)),
        evidence_needed=list(dict.fromkeys(needed)),
    )


def _comparable_evidence_item(
    ref: str,
    method: str,
    evidence: dict[str, dict[str, Any]],
) -> ComparableValuationEvidenceItem:
    item = evidence.get(ref) or {}
    metadata: dict[str, Any] = {}
    try:
        parsed = json.loads(str(item.get("excerpt") or ""))
        metadata = parsed if isinstance(parsed, dict) else {}
    except (TypeError, json.JSONDecodeError):
        metadata = {}
    metric = str(metadata.get("metric") or metadata.get("method") or "").lower().replace("/", "_")
    metric = {"ev_ebitda": "ev_ebitda", "ev_sales": "ev_sales", "p_b": "pb", "p_e": "pe"}.get(metric, metric)
    try:
        value = float(metadata["value"]) if metadata.get("value") is not None else None
        low = float(metadata["range_low"]) if metadata.get("range_low") is not None else value
        high = float(metadata["range_high"]) if metadata.get("range_high") is not None else value
    except (TypeError, ValueError):
        value = low = high = None
    as_of = str(
        metadata.get("as_of") or item.get("available_at") or item.get("publication_date")
        or item.get("information_date") or ""
    )
    period = str(metadata.get("period") or "")
    applicability = str(metadata.get("applicability") or "")
    accounting_basis = str(metadata.get("accounting_basis") or "")
    relationship = str(metadata.get("relationship") or "other")
    if relationship not in {"peer", "company_historical", "other"}:
        relationship = "other"
    reasons: list[str] = []
    if not item:
        reasons.append("Evidence Ref不存在")
    if item.get("source_type") != "comparable_valuation":
        reasons.append("source_type不是comparable_valuation")
    if item.get("source_grade") not in {"A", "B"} or item.get("supports_current_claim") is not True:
        reasons.append("证据不是当前有效、非未来A/B级来源")
    if metric != method:
        reasons.append(f"metric不匹配：需要{method}，实际{metric or 'missing'}")
    if relationship not in {"peer", "company_historical"}:
        reasons.append("证据未证明为同行可比或公司历史估值区间")
    if not as_of or not period:
        reasons.append("缺少as_of或估值期间/TTM口径")
    if not applicability:
        reasons.append("缺少适用性说明")
    if not accounting_basis:
        reasons.append("缺少会计/预测口径")
    if value is None and (low is None or high is None):
        reasons.append("缺少可审查倍数值或区间")
    return ComparableValuationEvidenceItem(
        subject=str(metadata.get("subject") or ""),
        symbol=str(metadata.get("symbol") or ""),
        relationship=relationship,
        method=method,
        metric=metric,
        value=value,
        range_low=low,
        range_high=high,
        as_of=as_of,
        period=period,
        source=str(item.get("source") or ""),
        evidence_ref=ref,
        applicability=applicability,
        accounting_basis=accounting_basis,
        valid=not reasons,
        reason="；".join(reasons) or "方法、日期、口径、适用性和Evidence Ref均匹配",
    )


def _valuation_method_candidates(
    item: FinancialBridge,
    shares: float | None,
    valuation_allowed: bool,
    valuation_block_reason: str = "",
    evidence: dict[str, dict[str, Any]] | None = None,
) -> list[ValuationMethodResult]:
    evidence = evidence or {}
    results: list[ValuationMethodResult] = []
    ref_fields = {
        "pe": "pe_multiple",
        "ev_ebitda": "ev_ebitda_multiple",
        "ev_sales": "ev_sales_multiple",
        "pb": "pb_multiple",
    }

    def add(
        method: str, available: bool, multiple: float, enterprise_value: float | None,
        equity_value: float | None, formula: str, reason: str,
    ) -> None:
        refs = list(dict.fromkeys(item.assumption_evidence_refs.get(ref_fields[method], [])))
        comparable = [_comparable_evidence_item(ref, method, evidence) for ref in refs]
        valid = [entry for entry in comparable if entry.valid]
        lows = [entry.range_low for entry in valid if entry.range_low is not None]
        highs = [entry.range_high for entry in valid if entry.range_high is not None]
        range_supported = bool(
            multiple > 0 and lows and highs
            and min(lows) * 0.95 <= multiple <= max(highs) * 1.05
        )
        validation_issues = [entry.reason for entry in comparable if not entry.valid]
        if not refs:
            validation_issues.append(f"缺少assumption_evidence_refs['{ref_fields[method]}']")
        elif valid and not range_supported:
            validation_issues.append(
                f"假设倍数{multiple:g}不在有效可比证据区间{min(lows):g}~{max(highs):g}内"
            )
        supported = bool(valid and range_supported)
        target = equity_value / shares if equity_value is not None and shares not in {None, 0} else None
        economic_available = bool(available and equity_value is not None and equity_value >= 0)
        final_reason = reason
        if economic_available and not supported:
            final_reason = "倍数虽有数值但缺少匹配metric/date/period/applicability的可靠Comparable Evidence"
        if not valuation_allowed:
            final_reason = valuation_block_reason or "前置Gate未通过，禁止估值"
        results.append(ValuationMethodResult(
            method=method,
            available=bool(valuation_allowed and economic_available and supported),
            multiple=multiple if multiple > 0 else None,
            multiple_source="comparable_evidence" if supported else ("unsupported_assumption" if multiple > 0 else "missing"),
            evidence_supported=supported,
            enterprise_value=enterprise_value,
            equity_value=equity_value,
            target_price=target,
            formula=formula,
            reason=final_reason,
            evidence_refs=refs,
            evidence_validation_issues=list(dict.fromkeys(validation_issues)),
            comparable_evidence=comparable,
        ))

    pe_equity = item.net_profit * item.pe_multiple if item.net_profit > 0 and item.pe_multiple > 0 else None
    add(
        "pe", pe_equity is not None, item.pe_multiple,
        pe_equity + item.net_debt if pe_equity is not None else None, pe_equity,
        "Equity=Net Profit×PE；EV=Equity+Net Debt",
        "可用" if pe_equity is not None else ("净利润≤0，PE禁止使用" if item.net_profit <= 0 else "缺少有效PE倍数"),
    )
    ebitda_ev = item.ebitda * item.ev_ebitda_multiple if item.ebitda > 0 and item.ev_ebitda_multiple > 0 else None
    add(
        "ev_ebitda", ebitda_ev is not None, item.ev_ebitda_multiple,
        ebitda_ev, ebitda_ev - item.net_debt if ebitda_ev is not None else None,
        "EV=EBITDA×EV/EBITDA；Equity=EV-Net Debt",
        "可用" if ebitda_ev is not None else ("EBITDA≤0，EV/EBITDA不适用" if item.ebitda <= 0 else "缺少有效EV/EBITDA倍数"),
    )
    sales_ev = item.revenue * item.ev_sales_multiple if item.revenue > 0 and item.ev_sales_multiple > 0 else None
    add(
        "ev_sales", sales_ev is not None, item.ev_sales_multiple,
        sales_ev, sales_ev - item.net_debt if sales_ev is not None else None,
        "EV=Revenue×EV/Sales；Equity=EV-Net Debt",
        "可用" if sales_ev is not None else ("收入≤0，EV/Sales不适用" if item.revenue <= 0 else "缺少有效EV/Sales倍数；不使用情景排序缩放兜底"),
    )
    pb_equity = item.book_value * item.pb_multiple if item.book_value > 0 and item.pb_multiple > 0 else None
    add(
        "pb", pb_equity is not None, item.pb_multiple,
        pb_equity + item.net_debt if pb_equity is not None else None, pb_equity,
        "Equity=Book Value×PB；EV=Equity+Net Debt",
        "可用" if pb_equity is not None else ("账面净资产≤0，PB不适用" if item.book_value <= 0 else "缺少有效PB倍数"),
    )
    return results


def _build_comparable_valuation_evidence_audit(
    output: ValuationOutput,
) -> ComparableValuationEvidenceAudit:
    records: dict[tuple[str, str], ComparableValuationEvidenceItem] = {}
    methods_available = {method: False for method in ("pe", "ev_ebitda", "ev_sales", "pb")}
    selected_methods: dict[str, str] = {}
    issues: list[str] = []
    needed: list[str] = []
    for scenario in output.scenarios:
        if not scenario.projections:
            issues.append(f"{scenario.name.title()}缺少终值预测")
            continue
        terminal = scenario.projections[-1]
        for candidate in terminal.valuation_candidates:
            methods_available[candidate.method] = methods_available[candidate.method] or candidate.available
            for record in candidate.comparable_evidence:
                records[(record.evidence_ref, record.method)] = record
        selected = next((candidate for candidate in terminal.valuation_candidates if candidate.selected), None)
        selected_methods[scenario.name] = selected.method if selected else "unavailable"
        if selected is None or not selected.available or not selected.evidence_supported:
            issues.append(f"{scenario.name.title()}终值没有由可靠Comparable Evidence支持的可用估值方法")
            needed.append(
                f"补充{scenario.name.title()}终值PE/EV-EBITDA/EV-Sales/PB中至少一种适用倍数的来源、日期、期间、会计口径及Evidence Ref"
            )
    supported = bool(
        len(output.scenarios) == 3
        and set(selected_methods) == {"bear", "base", "bull"}
        and all(method != "unavailable" for method in selected_methods.values())
        and not issues
    )
    return ComparableValuationEvidenceAudit(
        passed=supported,
        selected_methods_supported=supported,
        evidence_items=list(records.values()),
        methods_available=methods_available,
        selected_methods=selected_methods,
        issues=list(dict.fromkeys(issues)),
        evidence_needed=list(dict.fromkeys(needed)),
    )


def calculate_unified_valuation_engine(
    output: ValuationOutput,
    research: dict[str, Any],
    financial: FinancialOutput,
    evidence: dict[str, dict[str, Any]] | None = None,
) -> ValuationOutput:
    """Validate forecasts first, then build scenarios on the unified EV→Equity→Target coordinate."""
    base_revenue, _, base_period = _financial_baseline(financial)
    snapshot = research.get("market_snapshot") or {}
    market_cap, pb = snapshot.get("market_cap"), snapshot.get("pb")
    current_price, shares = snapshot.get("price"), snapshot.get("shares_outstanding")
    report = snapshot.get("consistency_report") or {}
    market_valuation_allowed = bool(report and report.get("valuation_allowed") is True and base_revenue and base_revenue > 0)
    output.current_price, output.shares_outstanding, output.current_market_cap = current_price, shares, market_cap
    derived_book_value = market_cap / pb if market_cap is not None and pb not in {None, 0} else 0

    for scenario in output.scenarios:
        previous_revenue = base_revenue
        scenario.projections.sort(key=lambda item: int(item.year) if str(item.year).isdigit() else 9999)
        for item in scenario.projections:
            item.revenue = previous_revenue * (1 + item.revenue_growth) if previous_revenue is not None else 0
            item.gross_profit = item.revenue * item.gross_margin
            item.rd_expense = item.revenue * item.rd_expense_ratio
            item.selling_expense = item.revenue * item.selling_expense_ratio
            item.admin_expense = item.revenue * item.admin_expense_ratio
            item.other_operating_expense = item.revenue * item.other_operating_expense_ratio
            item.operating_profit = item.gross_profit - item.rd_expense - item.selling_expense - item.admin_expense - item.other_operating_expense
            item.ebit = item.operating_profit
            item.financial_expense = item.revenue * item.financial_expense_ratio
            item.other_income = item.revenue * item.other_income_ratio
            item.pre_tax_profit = item.operating_profit - item.financial_expense + item.other_income
            item.tax = max(item.pre_tax_profit, 0) * item.tax_rate
            post_tax = item.pre_tax_profit - item.tax
            item.minority_interest = max(post_tax, 0) * item.minority_interest_ratio
            item.net_profit = post_tax - item.minority_interest
            depreciation = item.revenue * item.depreciation_amortization_ratio
            item.ebitda = item.operating_profit + depreciation
            item.free_cash_flow = None
            item.fcf_unavailable_reason = "缺少资本开支和营运资本变动的同口径预测，FCF未计算且不参与估值"
            item.eps = item.net_profit / shares if shares not in {None, 0} else None
            item.net_debt = item.debt - item.cash
            item.book_value = item.book_value or derived_book_value
            item.valuation_candidates = []
            previous_revenue = item.revenue

    output = calculate_forecast_reasonableness_audit(output, financial, evidence or {})
    probability_allowed = output.pre_valuation_probability_audit.passed
    valuation_allowed = bool(
        probability_allowed and market_valuation_allowed
        and output.forecast_reasonableness_audit.valuation_allowed
    )
    if not probability_allowed:
        valuation_block_reason = "Pre-Valuation Probability Gate未通过，禁止生成估值候选"
    elif market_valuation_allowed and not output.forecast_reasonableness_audit.valuation_allowed:
        valuation_block_reason = "Forecast Reasonableness Gate未通过，异常预测不得进入估值"
    else:
        valuation_block_reason = "市场数据一致性检查未通过，禁止估值"
    for scenario in output.scenarios:
        for item in scenario.projections:
            item.valuation_candidates = _valuation_method_candidates(
                item, shares, valuation_allowed, valuation_block_reason, evidence or {}
            )

    terminal_items = [scenario.projections[-1] for scenario in output.scenarios if scenario.projections]
    any_terminal_loss = any(item.net_profit <= 0 for item in terminal_items)
    priority = (
        ["ev_ebitda", "ev_sales", "pb"]
        if any_terminal_loss else ["ev_ebitda", "pe", "ev_sales", "pb"]
    )
    common_method = next((
        method for method in priority
        if len(terminal_items) == 3 and all(
            next((candidate.available for candidate in item.valuation_candidates if candidate.method == method), False)
            for item in terminal_items
        )
    ), "")
    output.primary_valuation_method = common_method or "mixed"
    output.financial_bridge_method = (
        "统一估值桥：经营预测→EBIT/EBITDA（FCF缺数据则明确不可用）→EV→减净债务→Equity Value→Target Price；"
        + (f"三情景共同使用 {common_method}" if common_method else "不存在三情景共同可用方法，逐情景选择后交由可比性审计决定是否阻断")
    )

    for scenario in output.scenarios:
        for item in scenario.projections:
            candidates = {candidate.method: candidate for candidate in item.valuation_candidates}
            local_priority = (
                ["ev_ebitda", "ev_sales", "pb"]
                if item.net_profit <= 0 else ["ev_ebitda", "pe", "ev_sales", "pb"]
            )
            selected_method = common_method if common_method and candidates[common_method].available else next(
                (method for method in local_priority if candidates[method].available), ""
            )
            for candidate in item.valuation_candidates:
                candidate.selected = candidate.method == selected_method
            selected = candidates.get(selected_method)
            item.unavailable_methods = [
                f"{candidate.method}: {candidate.reason}"
                for candidate in item.valuation_candidates if not candidate.available
            ]
            item.enterprise_value = selected.enterprise_value if selected else None
            item.equity_value = selected.equity_value if selected else None
            item.implied_market_cap = item.equity_value
            item.valuation_method = selected_method or "unavailable"
            item.valuation_multiple = selected.multiple or 0 if selected else 0
            item.valuation_formula = selected.formula if selected else "无可用估值方法"
            item.method_selection_reason = (
                f"{selected_method}是Bear/Base/Bull终值共同可用的最高优先级方法，统一用于跨情景比较"
                if selected and selected_method == common_method
                else (
                    f"无共同方法；按{'亏损' if item.net_profit <= 0 else '盈利'}情景优先级选择{selected_method}，跨方法结果必须通过可比性审计"
                    if selected else "PE、EV/EBITDA、EV/Sales、PB均不可用"
                )
            )
            item.method_warning = "；".join(item.unavailable_methods)
            checks = [
                _check_close("收入×毛利率=毛利", item.gross_profit, item.revenue * item.gross_margin),
                _check_close("毛利-期间费用=EBIT", item.ebit, item.gross_profit - item.rd_expense - item.selling_expense - item.admin_expense - item.other_operating_expense),
                _check_close("EBIT+折旧摊销=EBITDA", item.ebitda, item.ebit + item.revenue * item.depreciation_amortization_ratio),
                _check_close("税前利润-税-少数股东=净利润", item.net_profit, item.pre_tax_profit - item.tax - item.minority_interest),
            ]
            if item.enterprise_value is not None and item.equity_value is not None:
                checks.append(_check_close("EV-净债务=Equity Value", item.equity_value, item.enterprise_value - item.net_debt))
            if item.equity_value is not None and shares not in {None, 0}:
                target_price = item.equity_value / shares
                checks.append(_check_close("Equity Value=目标价×总股本", item.equity_value, target_price * shares))
            item.calculation_checks = checks
            item.calculation_valid = bool(selected and all(issue.passed for issue in checks))
            if not item.calculation_valid:
                item.enterprise_value = item.equity_value = item.implied_market_cap = None
                item.valuation_method = "unavailable"
            item.assumptions = list(dict.fromkeys(item.assumptions + [
                f"收入增长率 {item.revenue_growth:.1%}", f"毛利率 {item.gross_margin:.1%}",
                f"研发/销售/管理费用率 {item.rd_expense_ratio:.1%}/{item.selling_expense_ratio:.1%}/{item.admin_expense_ratio:.1%}",
                f"财务基期 {base_period or '不可用'}",
            ]))

        if scenario.projections:
            target = scenario.projections[-1]
            scenario.target_year, scenario.revenue, scenario.net_profit = target.year, target.revenue, target.net_profit
            scenario.valuation_method, scenario.valuation_multiple = target.valuation_method, target.valuation_multiple
            scenario.enterprise_value, scenario.net_debt = target.enterprise_value, target.net_debt
            scenario.equity_value = target.equity_value
            scenario.implied_market_cap = target.equity_value
            scenario.target_price = target.equity_value / shares if target.equity_value is not None and shares not in {None, 0} else None
            scenario.price_return = scenario.target_price / current_price - 1 if scenario.target_price is not None and current_price not in {None, 0} else None
            horizon = max(1, len(scenario.projections))
            scenario.annualized_price_return = (
                (1 + scenario.price_return) ** (1 / horizon) - 1
                if scenario.price_return is not None and scenario.price_return > -1 else scenario.price_return
            )

    output.comparable_valuation_evidence_audit = _build_comparable_valuation_evidence_audit(output)
    base = next((item for item in output.scenarios if item.name == "base" and item.projections), None)
    current_net_debt = base.projections[0].net_debt if base else None
    output.current_valuation_multiples = {
        "pe": snapshot.get("pe_ttm"),
        "pb": snapshot.get("pb"),
        "ev_sales": (
            (market_cap + current_net_debt) / base_revenue
            if None not in {market_cap, current_net_debt} and base_revenue not in {None, 0} else None
        ),
        "ev_ebitda": None,
    }
    if base:
        target = base.projections[-1]
        output.financial_tree = LogicTree(
            tree_type="financial", root_id="F10",
            nodes=[
                LogicTreeNode(node_id="F1", label="收入", metric="Revenue", value=target.revenue, unit="亿元", period=target.year),
                LogicTreeNode(node_id="F2", label="毛利", metric="Gross Profit", value=target.gross_profit, unit="亿元", period=target.year, depends_on=["F1"]),
                LogicTreeNode(node_id="F3", label="EBIT", metric="EBIT", value=target.ebit, unit="亿元", period=target.year, depends_on=["F2"]),
                LogicTreeNode(node_id="F4", label="EBITDA", metric="EBITDA", value=target.ebitda, unit="亿元", period=target.year, depends_on=["F3"]),
                LogicTreeNode(node_id="F5", label="FCF（数据不足）", metric="FCF", value=target.free_cash_flow, unit="亿元", period=target.year, depends_on=["F4"]),
                LogicTreeNode(node_id="F6", label="净利润", metric="Net Profit", value=target.net_profit, unit="亿元", period=target.year, depends_on=["F3"]),
                LogicTreeNode(node_id="F7", label="Enterprise Value", metric="EV", value=target.enterprise_value, unit="亿元", period=target.year, depends_on=["F4", "F6"]),
                LogicTreeNode(node_id="F8", label="净债务", metric="Net Debt", value=target.net_debt, unit="亿元", period=target.year),
                LogicTreeNode(node_id="F9", label="Equity Value", metric="Equity Value", value=target.equity_value, unit="亿元", period=target.year, depends_on=["F7", "F8"]),
                LogicTreeNode(node_id="F10", label="目标价", metric="Target Price", value=base.target_price, unit="元/股", period=target.year, depends_on=["F9"]),
            ],
            summary="统一财务估值桥：收入→EBIT/EBITDA（FCF缺少CapEx/营运资本预测时明确不可用）→EV→减净债务→Equity Value→Target Price",
        )
    if base_revenue is None:
        output.limitations.append("缺少可靠年度财务基期收入，程序化收入路径无法计算。")
    if not output.pre_valuation_probability_audit.passed:
        output.limitations.append("Pre-Valuation Probability Gate未通过；不得生成财务预测估值候选、目标价或Expected Return。")
    elif not market_valuation_allowed:
        output.limitations.append("市场数据一致性检查未通过，所有估值与目标价已禁用。")
    elif not output.forecast_reasonableness_audit.passed:
        output.limitations.append("Forecast Reasonableness Gate未通过，预测仅展示但不得进入估值；目标价与Expected Return已禁用。")
    if not output.comparable_valuation_evidence_audit.passed:
        output.limitations.append("Comparable Valuation Evidence不足；无可靠倍数证据的方法保持unavailable/null。")
    output = calculate_extended_valuation_audits(
        output, evidence or {}, financial, research
    )
    return build_gap_audits(output)


def build_gap_audits(output: ValuationOutput) -> ValuationOutput:
    implied = next((item for item in output.reverse_valuation if item.implied_net_profit is not None), None)
    earnings_entries: list[RequiredEarningsGapEntry] = []
    if implied:
        for scenario in output.scenarios:
            projection = next((item for item in scenario.projections if str(item.year) == str(implied.year)), None)
            forecast = projection.net_profit if projection else None
            gap = implied.implied_net_profit - forecast if forecast is not None else None
            gap_pct = gap / abs(implied.implied_net_profit) if gap is not None and implied.implied_net_profit not in {None, 0} else None
            status = "unavailable" if gap is None else ("shortfall" if gap > 0 else ("exceeds" if gap < 0 else "meets"))
            earnings_entries.append(RequiredEarningsGapEntry(
                scenario=scenario.name, year=str(implied.year),
                model_implied_earnings=implied.implied_net_profit,
                agent_forecast_earnings=forecast,
                required_earnings_gap=gap,
                gap_percentage=gap_pct,
                status=status,
                formula=(
                    f"{implied.implied_net_profit:.6f}-{forecast:.6f}={gap:.6f}；{gap:.6f}/{implied.implied_net_profit:.6f}={gap_pct:.6%}"
                    if None not in {forecast, gap, gap_pct} else "同年Agent Forecast不可用"
                ),
            ))
    all_below = bool(earnings_entries and all(item.status == "shortfall" for item in earnings_entries))
    bull_entry = next((item for item in earnings_entries if item.scenario == "bull"), None)
    above_bull = bool(bull_entry and bull_entry.status == "shortfall")
    bull_gap = bull_entry.required_earnings_gap if bull_entry else None
    bull_gap_pct = bull_entry.gap_percentage if bull_entry else None
    output.required_earnings_gap = RequiredEarningsGapAudit(
        year=str(implied.year) if implied else "",
        model_implied_earnings=implied.implied_net_profit if implied else None,
        entries=earnings_entries,
        all_scenarios_below=all_below,
        required_earnings_above_bull=above_bull,
        model_implied_vs_bull_gap=bull_gap,
        model_implied_vs_bull_gap_percentage=bull_gap_pct,
        model_implied_vs_bull_conclusion=(
            "Required Earnings > Bull Forecast"
            if above_bull else ("Bull Forecast达到或超过Model-Implied Earnings" if bull_entry else "Bull Forecast不可用")
        ),
        conclusion=(
            "所有情景均无法满足当前价格隐含盈利要求"
            if all_below else ("至少一个情景达到或超过Model-Implied Earnings" if earnings_entries else "Model-Implied Earnings不可用")
        ),
    )

    gap_entries: list[ValuationGapEntry] = []
    for scenario in output.scenarios:
        current_multiple = output.current_valuation_multiples.get(scenario.valuation_method)
        multiple_change = (
            scenario.valuation_multiple - current_multiple
            if scenario.valuation_multiple is not None and current_multiple is not None else None
        )
        gap_entries.append(ValuationGapEntry(
            scenario=scenario.name,
            current_model_implied_price=output.current_price,
            current_model_implied_market_cap=output.current_market_cap,
            target_price=scenario.target_price,
            price_gap=(scenario.target_price - output.current_price if None not in {scenario.target_price, output.current_price} else None),
            price_return=scenario.price_return,
            scenario_method=scenario.valuation_method,
            scenario_multiple=scenario.valuation_multiple,
            current_comparable_multiple=current_multiple,
            multiple_change=multiple_change,
            multiple_change_percentage=(multiple_change / abs(current_multiple) if multiple_change is not None and current_multiple not in {None, 0} else None),
            multiple_comparable=current_multiple is not None,
            note=(
                "同口径倍数可比" if current_multiple is not None
                else "当前同口径倍数不可用；不得用其他方法倍数替代"
            ),
        ))
    output.valuation_gap = ValuationGapAudit(
        current_model_implied_price=output.current_price,
        current_model_implied_market_cap=output.current_market_cap,
        current_valuation_multiples=output.current_valuation_multiples,
        entries=gap_entries,
        conclusion="当前价格称为Model-Implied Valuation，仅用于与情景目标价比较，不代表市场一致预期或已经Price In。",
    )
    return output


def _sensitivity_target_price(
    item: FinancialBridge,
    shares: float | None,
    *,
    revenue: float | None = None,
    gross_margin: float | None = None,
    multiple: float | None = None,
) -> float | None:
    if shares in {None, 0}:
        return None
    revenue_value = item.revenue if revenue is None else revenue
    margin = item.gross_margin if gross_margin is None else gross_margin
    valuation_multiple = item.valuation_multiple if multiple is None else multiple
    operating_ratio = (
        margin - item.rd_expense_ratio - item.selling_expense_ratio
        - item.admin_expense_ratio - item.other_operating_expense_ratio
    )
    operating_profit = revenue_value * operating_ratio
    pre_tax = operating_profit - revenue_value * item.financial_expense_ratio + revenue_value * item.other_income_ratio
    tax = max(pre_tax, 0) * item.tax_rate
    post_tax = pre_tax - tax
    net_profit = post_tax - max(post_tax, 0) * item.minority_interest_ratio
    ebitda = operating_profit + revenue_value * item.depreciation_amortization_ratio
    if item.valuation_method == "pe" and net_profit > 0 and valuation_multiple > 0:
        equity = net_profit * valuation_multiple
    elif item.valuation_method == "ev_sales" and valuation_multiple > 0:
        equity = revenue_value * valuation_multiple - item.net_debt
    elif item.valuation_method == "ev_ebitda" and ebitda > 0 and valuation_multiple > 0:
        equity = ebitda * valuation_multiple - item.net_debt
    elif item.valuation_method == "pb" and item.book_value > 0 and valuation_multiple > 0:
        equity = item.book_value * valuation_multiple
    else:
        return None
    return equity / shares if equity >= 0 else None


def calculate_sensitivity_audit(output: ValuationOutput) -> ValuationOutput:
    base = next((item for item in output.scenarios if item.name == "base" and item.projections), None)
    if not base or base.target_price is None:
        output.sensitivity_audit = [SensitivityAuditItem(
            variable="sensitivity", shock="unavailable", warning="Base情景目标价不可用，无法执行敏感性审计"
        )]
        return output
    terminal = base.projections[-1]
    previous_revenue = base.projections[-2].revenue if len(base.projections) > 1 else terminal.revenue / max(1 + terminal.revenue_growth, 1e-9)
    baseline_target = base.target_price
    denominator = output.probability_weighted_target_price or baseline_target
    items: list[SensitivityAuditItem] = []

    def add(variable: str, shock: str, target: float | None, calculation: str) -> None:
        delta = target - baseline_target if target is not None else None
        weighted_delta = base.probability * delta if delta is not None else None
        items.append(SensitivityAuditItem(
            variable=variable,
            shock=shock,
            target_price_change=delta,
            weighted_target_change=weighted_delta,
            weighted_target_change_ratio=(weighted_delta / denominator if weighted_delta is not None and denominator else None),
            calculation=calculation,
        ))

    for shock in (-0.02, 0.02):
        shocked_revenue = previous_revenue * (1 + terminal.revenue_growth + shock)
        add("terminal_revenue_growth", f"{shock:+.0%}", _sensitivity_target_price(
            terminal, output.shares_outstanding, revenue=shocked_revenue
        ), "仅扰动Base终值年度收入增长率，其他利润率与倍数不变")
    for shock in (-0.01, 0.01):
        add("gross_margin", f"{shock:+.0%}", _sensitivity_target_price(
            terminal, output.shares_outstanding, gross_margin=terminal.gross_margin + shock
        ), "仅扰动Base终值毛利率，期间费用率与倍数不变")
    for shock in (-0.10, 0.10):
        add(f"{terminal.valuation_method}_multiple", f"{shock:+.0%}", _sensitivity_target_price(
            terminal, output.shares_outstanding, multiple=terminal.valuation_multiple * (1 + shock)
        ), "仅扰动Base终值估值倍数")

    methods = {item.valuation_method for item in output.scenarios}
    if len(methods) > 1:
        items.append(SensitivityAuditItem(
            variable="valuation_method_switch",
            shock="cross-scenario",
            calculation="比较三个情景末期估值方法",
            warning=(
                f"情景间使用不同估值方法（{', '.join(sorted(methods))}）；方法切换可能比连续参数扰动更重要，"
                "必须依赖目标价排序审计"
            ),
        ))
    numeric = [item for item in items if item.weighted_target_change is not None]
    if numeric:
        max(numeric, key=lambda item: abs(item.weighted_target_change or 0)).is_primary_driver = True
    output.sensitivity_audit = items
    output.sensitivity = [
        f"{item.variable} {item.shock}: Base目标价变化 {item.target_price_change:+.2f} 元，概率加权贡献 {item.weighted_target_change:+.2f} 元"
        for item in numeric
    ] + [item.warning for item in items if item.warning]
    return output


def derive_decision_probabilities(
    valuation: ValuationOutput,
    financial: FinancialOutput,
    research: dict[str, Any],
) -> dict[str, Any]:
    base_revenue, base_profit, _ = _financial_baseline(financial)
    snapshot = research.get("market_snapshot") or {}
    scenarios = valuation.scenarios
    bull = next((item for item in scenarios if item.name == "bull"), None)
    bear = next((item for item in scenarios if item.name == "bear"), None)
    fundamental = sum(
        item.probability for item in scenarios
        if base_profit is not None and item.net_profit is not None and item.net_profit > base_profit
    )
    repricing_names: list[str] = []
    for item in scenarios:
        current_multiple = None
        if item.valuation_method == "pe":
            current_multiple = snapshot.get("pe_ttm")
        elif item.valuation_method == "pb":
            current_multiple = snapshot.get("pb")
        elif item.valuation_method == "ev_sales" and base_revenue not in {None, 0} and item.projections:
            current_multiple = (
                (valuation.current_market_cap + item.projections[-1].net_debt) / base_revenue
                if valuation.current_market_cap is not None else None
            )
        if current_multiple is not None and item.valuation_multiple is not None and item.valuation_multiple > current_multiple:
            repricing_names.append(item.name)
    repricing = sum(item.probability for item in scenarios if item.name in repricing_names)
    negative_return = sum(
        item.probability for item in scenarios
        if item.target_price is not None and valuation.current_price is not None and item.target_price < valuation.current_price
    )
    return {
        "bull_thesis": bull.probability if bull else 0,
        "fundamental_improvement": min(1.0, fundamental),
        "valuation_repricing": min(1.0, repricing),
        "downside_event": bear.probability if bear else 0,
        "negative_return": min(1.0, negative_return),
        "repricing_scenarios": repricing_names,
        "fundamental_scenarios": [
            item.name for item in scenarios
            if base_profit is not None and item.net_profit is not None and item.net_profit > base_profit
        ],
        "baseline_profit": base_profit,
    }


def derive_investment_verdict(
    valuation: ValuationOutput,
    probability_of_negative_return: float,
) -> InvestmentVerdict:
    required = next((item.required_return for item in valuation.reverse_valuation), 0.10)
    audit = valuation.valuation_audit
    forecast_audit = valuation.forecast_reasonableness_audit
    implied = next((item for item in valuation.reverse_valuation if item.implied_net_profit is not None), None)
    derivation: list[str] = []
    positive: list[str] = []
    negative: list[str] = []
    scenario_assessments: dict[str, str] = {}
    current_requirement = "Model-Implied Earnings不可用"
    if implied:
        current_requirement = (
            f"当前价格在{required:.1%}要求回报率、{implied.assumed_pe or 0:g}倍退出PE假设下，"
            f"要求{implied.year} Model-Implied Earnings {implied.implied_net_profit:.2f}亿元"
        )
        derivation.append(current_requirement)
        for scenario in valuation.scenarios:
            projection = next((p for p in scenario.projections if str(p.year) == str(implied.year)), None)
            if projection:
                gap = projection.net_profit - implied.implied_net_profit
                line = (
                    f"{scenario.name.title()} {implied.year} Agent Forecast Earnings "
                    f"{projection.net_profit:.2f}亿元，较Model-Implied {gap:+.2f}亿元"
                )
                scenario_assessments[scenario.name] = (
                    f"{'达到' if gap >= 0 else '未达到'}；{line}"
                )
                derivation.append(line)
                (positive if gap > 0 else negative).append(line)
    if audit.passed:
        derivation.extend([audit.weighted_target_formula, audit.expected_return_formula])
    else:
        derivation.append("前置数据/预测/估值审计失败，概率加权目标价与Expected Return已禁用")

    failed_layer = ""
    valuation_block_reason = ""
    evidence_needed = list(forecast_audit.evidence_needed)
    evidence_needed.extend(valuation.pre_valuation_probability_audit.evidence_needed)
    evidence_needed.extend(valuation.comparable_valuation_evidence_audit.evidence_needed)
    if not valuation.probability_audit.passed:
        evidence_needed.append("补充互斥且穷尽的Bear/Base/Bull基础概率依据，并确保Probability Agent生成阶段合计100%")
    if not audit.complete_targets:
        for scenario in valuation.scenarios:
            if scenario.target_price is not None or not scenario.projections:
                continue
            terminal = scenario.projections[-1]
            unavailable = "；".join(terminal.unavailable_methods) or terminal.method_selection_reason
            evidence_needed.append(
                f"补充{scenario.name.title()} {terminal.year}终值可用估值输入及倍数证据：{unavailable}"
            )
    if not audit.valuation_methods_comparable:
        evidence_needed.append("补充三情景共同估值方法或可证明跨方法经济可比性的终值证据")
    evidence_needed = list(dict.fromkeys(evidence_needed))
    key_driver = forecast_audit.key_driver or "关键财务驱动无法可靠识别"
    if not audit.passed:
        status, label = "blocked", "审计阻断"
        coverage = False
        if not valuation.pre_valuation_probability_audit.passed:
            failed_layer = "probability_generation"
        elif not forecast_audit.margin_expansion_evidence_passed:
            failed_layer = "margin_expansion_evidence"
        elif not forecast_audit.passed:
            failed_layer = "财务预测合理性层"
        elif not valuation.comparable_valuation_evidence_audit.passed:
            failed_layer = "comparable_valuation_evidence"
        elif not valuation.probability_audit.passed or not audit.probability_sum_valid:
            failed_layer = "情景概率层"
        elif not audit.complete_targets or not audit.valuation_methods_comparable:
            failed_layer = "估值可比性层"
        else:
            failed_layer = "Expected Return审计层"
        valuation_block_reason = "；".join(audit.issues) or "估值审计未通过"
        base_text = scenario_assessments.get("base", "Base Forecast与Model-Implied Earnings无法同年比较")
        bull_text = scenario_assessments.get("bull", "Bull Forecast与Model-Implied Earnings无法同年比较")
        summary = (
            f"失败层级：{failed_layer}。①估值阻断：{valuation_block_reason}；"
            f"②当前价格盈利要求：{current_requirement}；③Base：{base_text}；Bull：{bull_text}；"
            f"④关键财务变量：{key_driver}；⑤解除阻断所需证据："
            f"{'；'.join(evidence_needed) if evidence_needed else '补齐审计列明的缺失数据或证据'}。"
        )
    else:
        annualized = valuation.annualized_expected_return
        coverage = bool(
            annualized is not None and annualized >= required and probability_of_negative_return <= 0.5
        )
        concentration = max(
            (abs(item.weighted_target_change_ratio or 0) for item in valuation.sensitivity_audit),
            default=0,
        )
        if coverage and concentration < 0.10:
            status, label = "favorable", "回报覆盖风险"
            summary = "数据、预测合理性、估值可比性及Expected Return审计均通过。"
        elif annualized is not None and annualized < required - 0.02:
            status, label = "insufficient_return", "回报不足"
            summary = "全部前置审计通过，但年化Expected Return不足以覆盖要求回报率与负回报情景权重。"
        else:
            status, label = "mixed", "收益风险混合"
            summary = "全部前置审计通过，但回报接近门槛或结果集中受单一敏感变量驱动。"
    return InvestmentVerdict(
        status=status,
        label=label,
        summary=summary,
        required_annual_return=required,
        expected_price_return=valuation.expected_price_return,
        annualized_expected_return=valuation.annualized_expected_return,
        probability_of_negative_return=probability_of_negative_return,
        risk_coverage_passed=coverage,
        price_reflection_conclusion=(
            "Model-Implied Earnings仅说明给定假设下当前价格要求的盈利路径；不等于Market Consensus，也不表示已经Price In。"
        ),
        failed_layer=failed_layer,
        valuation_block_reason=valuation_block_reason,
        current_price_required_earnings=current_requirement,
        base_forecast_assessment=scenario_assessments.get("base", "不可用"),
        bull_forecast_assessment=scenario_assessments.get("bull", "不可用"),
        key_financial_driver=key_driver,
        evidence_needed=evidence_needed,
        derivation=derivation,
        positive_gap_drivers=positive,
        negative_gap_drivers=negative,
        blockers=audit.issues if not audit.passed else [],
    )


def finalize_risk_reward_and_decision(
    output: ValuationOutput,
    financial: FinancialOutput,
    research: dict[str, Any],
) -> ValuationOutput:
    """Finalize risk/reward and decisions only after the authoritative valuation gate."""
    output.risk_reward_audit = build_risk_reward_audit(output)
    blockers = list(dict.fromkeys(
        output.valuation_audit.issues
        + output.risk_reward_audit.issues
        + ([output.risk_reward_audit.blocked_reason] if output.risk_reward_audit.blocked_reason else [])
    ))
    if not output.valuation_audit.passed or not output.risk_reward_audit.passed:
        reason = (
            "blocked：最终Valuation Audit未通过；风险收益、决策概率与投资结论数值禁用"
            if not output.valuation_audit.passed
            else "blocked：Risk Reward Audit未通过；决策概率与投资结论数值禁用"
        )
        output.decision_probabilities = DecisionProbabilitySummary(
            status="blocked",
            reasoning=reason,
            blockers=blockers or ["确定性决策前置Gate未通过"],
        )
        output.deterministic_probability_basis = []
        output.deterministic_investment_verdict = InvestmentVerdict(
            status="blocked",
            label="审计阻断",
            summary=reason,
            failed_layer=(
                "valuation_audit" if not output.valuation_audit.passed else "risk_reward_audit"
            ),
            valuation_block_reason="；".join(blockers) or reason,
            blockers=output.decision_probabilities.blockers,
        )
        return output

    # These helpers are intentionally called only after all target prices,
    # probabilities, returns, upside/downside and loss probability are final.
    probabilities = derive_decision_probabilities(output, financial, research)
    negative_return = float(probabilities["negative_return"])
    verdict = derive_investment_verdict(output, negative_return)
    scenarios = output.scenarios
    positive_return = sum(
        scenario.probability for scenario in scenarios
        if scenario.target_price is not None and output.current_price is not None
        and scenario.target_price > output.current_price
    )
    base_or_better = sum(
        scenario.probability for scenario in scenarios if scenario.name in {"base", "bull"}
    )
    confidence = sum(
        scenario.probability * scenario.probability_assessment.confidence
        for scenario in scenarios
    )
    output.decision_probabilities = DecisionProbabilitySummary(
        status="available",
        probability_of_positive_return=min(1.0, positive_return),
        probability_of_negative_return=min(1.0, negative_return),
        probability_of_base_or_better=min(1.0, base_or_better),
        probability_of_thesis_success=min(1.0, float(probabilities["bull_thesis"])),
        expected_return=output.risk_reward_audit.expected_return,
        confidence=min(1.0, confidence),
        reasoning=(
            "完全基于最终归一化Bear/Base/Bull概率、经审计目标价和当前价格确定性汇总；"
            "正/负回报概率分别汇总目标价高于/低于当前价的互斥情景"
        ),
    )
    scenario_refs = list(dict.fromkeys(
        ref for scenario in scenarios for ref in scenario.evidence_refs
    ))
    output.deterministic_probability_basis = [
        ProbabilityBasis(
            metric="Bull Thesis",
            value=min(1.0, float(probabilities["bull_thesis"])),
            formula="P(S=Bull)",
            factors=["最终归一化Bull情景概率"],
            evidence_refs=scenario_refs,
        ),
        ProbabilityBasis(
            metric="基本面改善",
            value=min(1.0, float(probabilities["fundamental_improvement"])),
            formula=f"ΣP(S)，其中终值净利润高于历史基准{probabilities['baseline_profit'] or 0:.6f}",
            factors=[
                f"qualifying scenario: {name}"
                for name in probabilities["fundamental_scenarios"]
            ],
            evidence_refs=scenario_refs,
        ),
        ProbabilityBasis(
            metric="估值重估",
            value=min(1.0, float(probabilities["valuation_repricing"])),
            formula="ΣP(S)，其中终值倍数高于当前同口径倍数",
            factors=[
                f"repricing scenario: {name}"
                for name in probabilities["repricing_scenarios"]
            ],
            evidence_refs=scenario_refs,
        ),
        ProbabilityBasis(
            metric="Downside Event",
            value=min(1.0, float(probabilities["downside_event"])),
            formula="P(S=Bear)",
            factors=["重大基本面恶化区间概率，不等同于负回报概率"],
            evidence_refs=scenario_refs,
        ),
        ProbabilityBasis(
            metric="负回报概率",
            value=min(1.0, negative_return),
            formula="ΣP(S)，其中target_price<current_price",
            factors=[
                f"{scenario.name}: target={scenario.target_price}, probability={scenario.probability:.6f}"
                for scenario in scenarios
            ],
            evidence_refs=scenario_refs,
        ),
    ]
    output.deterministic_investment_verdict = verdict
    return output


def finalize_deterministic_decision_summary(
    output: ValuationOutput,
    financial: FinancialOutput | None = None,
    research: dict[str, Any] | None = None,
) -> ValuationOutput:
    """Backward-compatible wrapper for the strict risk/reward and decision finalizer."""
    if financial is None:
        implied_base = next((
            item for item in output.reverse_valuation
            if item.base_revenue is not None or item.base_net_profit is not None
        ), None)
        financial = FinancialOutput.model_validate({
            "historical": ([{
                "period": "Model-Implied baseline",
                "revenue": implied_base.base_revenue,
                "net_profit": implied_base.base_net_profit,
            }] if implied_base else [])
        })
    if not research:
        research = {"market_snapshot": {
            "price": output.current_price,
            "market_cap": output.current_market_cap,
            "pe_ttm": output.current_valuation_multiples.get("pe"),
            "pb": output.current_valuation_multiples.get("pb"),
        }}
    return finalize_risk_reward_and_decision(output, financial, research)


def evidence_probability_adjustment(
    refs: list[str], evidence: dict[str, dict[str, Any]]
) -> tuple[EvidenceGateResult, float, float, str]:
    gate = evaluate_evidence_gate(refs, evidence)
    items = [evidence[ref] for ref in dict.fromkeys(refs) if ref in evidence]
    future_count = sum(item.get("temporal_scope") == "future" for item in items)
    current_c = sum(
        item.get("source_grade") == "C" and item.get("temporal_scope") == "current"
        for item in items
    )
    if gate.current_a_count:
        factor, confidence, label = 1.05, 0.90, "当前A级原始证据"
    elif gate.current_b_source_count >= 2:
        factor, confidence, label = 1.00, 0.80, "两个独立当前B级来源"
    elif gate.current_b_source_count == 1:
        factor, confidence, label = 0.85, 0.60, "仅一个当前B级来源"
    elif current_c:
        factor, confidence, label = 0.72, 0.45, "仅当前C级二手证据"
    else:
        factor, confidence, label = 0.60, 0.30, "缺少当前有效证据"
    if future_count:
        factor, confidence = min(factor, 0.45), min(confidence, 0.20)
        label += f"；含{future_count}条未来证据，已剔除并下调"
    return gate, factor, confidence, f"基础概率×{factor:.2f}（{label}）"


def validate_market_snapshot(snapshot: MarketSnapshot, analysis_date: Any) -> MarketSnapshot:
    issues: list[ValidationIssue] = []
    def add(check: str, passed: bool, expected: str, actual: str, message: str, required: bool = True) -> None:
        issues.append(ValidationIssue(
            check=check, severity=("info" if passed else ("error" if required else "warning")),
            passed=passed, expected=expected, actual=actual, message=message,
        ))
    if snapshot.as_of > analysis_date:
        add("行情时点不晚于分析日", False, f"≤{analysis_date}", str(snapshot.as_of), "存在Look-ahead Bias")
    else:
        add("行情时点不晚于分析日", True, f"≤{analysis_date}", str(snapshot.as_of), "通过")
    add("当前价格有效", bool(snapshot.price and snapshot.price > 0), ">0", str(snapshot.price), "当前价格必须为正")
    if None not in {snapshot.day_low, snapshot.price, snapshot.day_high}:
        add("价格位于当日高低区间", snapshot.day_low <= snapshot.price <= snapshot.day_high, f"{snapshot.day_low}~{snapshot.day_high}", str(snapshot.price), "当前价格超出当日区间")
    else:
        add("价格位于当日高低区间", False, "需要当日高低价", "数据缺失", "未执行", required=False)
    if None not in {snapshot.low_52w, snapshot.price, snapshot.high_52w}:
        add("价格位于52周高低区间", snapshot.low_52w <= snapshot.price <= snapshot.high_52w, f"{snapshot.low_52w}~{snapshot.high_52w}", str(snapshot.price), "当前价格与52周区间矛盾")
    else:
        add("价格位于52周高低区间", False, "需要52周历史价格", "数据缺失", "未执行", required=False)
    independent_shares = bool(
        snapshot.shares_source
        and "总市值÷当前价格" not in snapshot.shares_source
        and "market cap" not in snapshot.shares_source.lower()
    )
    if independent_shares and None not in {snapshot.price, snapshot.shares_outstanding, snapshot.market_cap} and snapshot.market_cap:
        calculated = snapshot.price * snapshot.shares_outstanding
        relative = abs(calculated - snapshot.market_cap) / abs(snapshot.market_cap)
        add("总市值=股价×独立总股本", relative <= 0.05, "偏差≤5%", f"计算{calculated:.2f} vs 市值{snapshot.market_cap:.2f}，偏差{relative:.2%}", "独立股本一致性")
    else:
        add("总市值=股价×独立总股本", False, "需要独立股本", snapshot.shares_source or "数据缺失", "总市值÷股价得到的股本不能反向验证总市值")
    if None not in {snapshot.price, snapshot.pe_ttm, snapshot.eps_ttm} and snapshot.eps_ttm not in {0, None}:
        calculated = snapshot.price / snapshot.eps_ttm
        relative = abs(calculated - snapshot.pe_ttm) / max(abs(snapshot.pe_ttm), 1e-9)
        add("PE=股价÷TTM EPS", relative <= 0.15, "偏差≤15%", f"计算{calculated:.2f} vs PE{snapshot.pe_ttm:.2f}", "PE/EPS一致性")
    else:
        add("PE=股价÷TTM EPS", False, "需要独立TTM EPS", "数据缺失", "未执行", required=False)
    if None not in {snapshot.price, snapshot.pb, snapshot.book_value_per_share} and snapshot.book_value_per_share not in {0, None}:
        calculated = snapshot.price / snapshot.book_value_per_share
        relative = abs(calculated - snapshot.pb) / max(abs(snapshot.pb), 1e-9)
        add("PB=股价÷每股净资产", relative <= 0.10, "偏差≤10%", f"计算{calculated:.2f} vs PB{snapshot.pb:.2f}", "PB/净资产一致性")
    else:
        add("PB=股价÷每股净资产", False, "需要每股净资产", "数据缺失", "未执行", required=False)
    errors = [issue for issue in issues if issue.severity == "error" and not issue.passed]
    snapshot.consistency_report = DataConsistencyReport(
        passed=not errors, valuation_allowed=not errors, issues=issues,
        summary="通过，可进入估值" if not errors else f"发现{len(errors)}项关键矛盾，禁止进入估值",
    )
    return snapshot