from __future__ import annotations

from collections import Counter
from typing import Any

from .calculations import evaluate_evidence_gate
from .schemas import (
    CatalystOutput,
    EvidenceAcquisitionPlan,
    EvidenceAcquisitionPlanItem,
    EvidenceBasedInvestmentVerdict,
    EvidenceStatusLayer,
    EvidenceStatusReference,
    FalsifiableCondition,
    InvestmentLogicChainAudit,
    MispricingDriver,
    MispricingVerdict,
    UnifiedEvidenceStatusItem,
    ValuationOutput,
)

_STATUS_ORDER = ("已验证", "部分验证", "无法验证", "已证伪", "互相矛盾")
_SCENARIO_LABEL = {"bear": "悲观", "base": "基准", "bull": "乐观"}
_DIRECTION_LABEL = {"increase": "上升", "decrease": "下降", "mixed": "双向变化", "unknown": "方向不明"}
_VARIABLE_LABEL = {
    "revenue_growth": "营业收入增长率", "revenue": "营业收入",
    "gross_margin": "毛利率", "ebit_margin": "营业利润率",
    "rd_expense_ratio": "研发费用率", "selling_expense_ratio": "销售费用率",
    "admin_expense_ratio": "管理费用率", "other_operating_expense_ratio": "其他经营费用率",
    "financial_expense_ratio": "财务费用率", "other_income_ratio": "其他收益率",
    "depreciation_amortization_ratio": "折旧摊销率", "tax_rate": "税率",
    "minority_interest_ratio": "少数股东损益占比", "valuation_multiple": "估值倍数",
}
_CATEGORY_PRIORITY = {
    "情景业务触发": 100,
    "毛利率提升": 98,
    "预测合理性": 96,
    "情景差异": 94,
    "可比估值": 92,
    "催化因素": 90,
    "预期差驱动": 88,
    "投资逻辑": 86,
    "情景概率依据": 80,
}


def _unique(values: list[str] | tuple[str, ...]) -> list[str]:
    return list(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))


def _user_text(value: Any) -> str:
    text = str(value or "")
    replacements = (
        ("Bull", "乐观情景"), ("Base", "基准情景"), ("Bear", "悲观情景"),
        ("Evidence Ref", "证据编号"), ("Evidence", "证据"),
        ("Forecast", "预测"), ("Gate", "检查关卡"),
    )
    for source, target in replacements:
        text = text.replace(source, target)
    return text


def _evidence_date(item: dict[str, Any]) -> str:
    value = item.get("available_at") or item.get("publication_date") or item.get("information_date")
    return str(value or "")[:10]

def _references(
    refs: list[str], evidence: dict[str, dict[str, Any]], accepted: set[str],
) -> list[EvidenceStatusReference]:
    result: list[EvidenceStatusReference] = []
    for ref in _unique(refs):
        item = evidence.get(ref)
        if not item:
            result.append(EvidenceStatusReference(
                evidence_id=ref, note="证据编号不存在于本次研究证据库",
            ))
            continue
        valid = ref in accepted
        note = "满足当前有效性要求" if valid else str(
            item.get("invalid_reason") or "未满足来源等级、日期或独立性要求"
        )
        result.append(EvidenceStatusReference(
            evidence_id=ref,
            fact=str(item.get("fact") or ""),
            source=str(item.get("source") or ""),
            source_grade=str(item.get("source_grade") or ""),
            evidence_date=_evidence_date(item),
            current_and_valid=valid,
            note=note,
        ))
    return result


def _logic_status(issues: list[str], complete: bool = True) -> str:
    text = "；".join(issues)
    if any(marker in text for marker in ("数学桥不一致", "不能解释", "方向相反", "排序反转")):
        return "逻辑错误"
    if not complete or any(marker in text for marker in (
        "为空", "缺少结构化", "未匹配", "缺少可追溯", "缺少对应", "必须明确",
    )):
        return "逻辑不完整"
    if any(marker in text for marker in ("前置确定性条件未通过", "无法审计", "无法判断")):
        return "无法判断"
    return "逻辑完整"


def _make_item(
    *, assumption_id: str, category: str, title: str, hypothesis: str,
    support_refs: list[str], counter_refs: list[str],
    evidence: dict[str, dict[str, Any]], missing: list[str], next_verification: str,
    issues: list[str] | None = None, scenario: str = "", year: str = "",
    variables: list[str] | None = None, critical: bool = True,
    mapping_complete: bool = True, logic_status: str | None = None,
    explicitly_refuted: bool = False,
) -> UnifiedEvidenceStatusItem:
    support_refs = _unique(support_refs)
    counter_refs = _unique(counter_refs)
    support_gate = evaluate_evidence_gate(support_refs, evidence)
    counter_gate = evaluate_evidence_gate(counter_refs, evidence)
    known_support = [ref for ref in support_refs if ref in evidence]
    known_counter = [ref for ref in counter_refs if ref in evidence]
    item_issues = _unique(issues or [])
    resolved_logic = logic_status or _logic_status(item_issues, mapping_complete)

    if support_gate.passed and counter_gate.passed:
        status = "互相矛盾"
    elif explicitly_refuted and counter_gate.passed:
        status = "已证伪"
    elif support_gate.passed and mapping_complete and resolved_logic == "逻辑完整":
        status = "已验证"
    elif known_support or known_counter:
        status = "部分验证"
    else:
        status = "无法验证"

    conclusions = {
        "已验证": f"{title}已有满足要求的证据支持。",
        "部分验证": f"{title}存在相关证据，但尚不足以完成验证。",
        "无法验证": f"{title}目前没有足够有效证据，暂时无法判断。",
        "已证伪": f"{title}存在满足要求的有效反证，当前假设不成立。",
        "互相矛盾": f"{title}的高质量支持证据与反方证据互相矛盾。",
    }
    why = {
        "已验证": "支持证据满足来源等级、日期、独立性和变量对应关系要求。",
        "部分验证": "已找到相关资料，但证据数量、质量、独立来源或变量量化关系仍不完整。",
        "无法验证": "没有找到能够通过当前证据检查的支持证据或有效反证。",
        "已证伪": "反方证据通过证据检查，而支持证据没有形成同等有效的证明。",
        "互相矛盾": "支持与反方两侧都通过证据检查，不能只采用其中一侧得出结论。",
    }
    support_view = _references(
        support_refs, evidence, set(support_gate.accepted_evidence_refs)
    )
    counter_view = _references(
        counter_refs, evidence, set(counter_gate.accepted_evidence_refs)
    )
    data = (
        f"支持证据{len(support_refs)}条，其中当前有效{len(support_gate.accepted_evidence_refs)}条；"
        f"反方证据{len(counter_refs)}条，其中当前有效{len(counter_gate.accepted_evidence_refs)}条。"
    )
    implication = (
        "该假设可以继续进入后续审计，但仍不能替代其他关卡。" if status == "已验证"
        else "这不等于假设错误，也不等于已经被证伪；目标价、预期收益率和可执行投资优势继续保持关闭。"
        if status in {"部分验证", "无法验证"}
        else "在矛盾证据得到解释前，相关投资结论必须保持关闭。"
        if status == "互相矛盾"
        else "该假设不能继续作为投资逻辑的有效前提。"
    )
    resolved_missing = [] if status == "已验证" else _unique(
        [_user_text(item) for item in missing]
    )
    return UnifiedEvidenceStatusItem(
        assumption_id=assumption_id, category=category, title=title,
        hypothesis=hypothesis, scenario=scenario, year=year,
        financial_variables=_unique(variables or []), critical=critical,
        current_status=status, logic_status=resolved_logic,
        one_sentence_conclusion=conclusions[status], why=why[status], data=data,
        implication=implication, support_evidence=support_view,
        counter_evidence=counter_view, missing_evidence=resolved_missing,
        next_verification=_user_text(next_verification), support_gate=support_gate,
        counter_gate=counter_gate, issues=item_issues,
    )

def build_evidence_status_layer(
    valuation: ValuationOutput,
    catalysts: CatalystOutput,
    drivers: list[MispricingDriver],
    investment_logic: InvestmentLogicChainAudit,
    evidence: dict[str, dict[str, Any]],
    falsifiers: list[FalsifiableCondition] | None = None,
) -> EvidenceStatusLayer:
    items: list[UnifiedEvidenceStatusItem] = []
    refuted_driver_ids = {
        item.driver_id for item in (falsifiers or [])
        if item.currently_invalidated and item.current_status == "refuted"
    }

    probability_audit = valuation.pre_valuation_probability_audit
    for scenario in probability_audit.scenarios:
        refs = list(scenario.probability_basis_refs or scenario.evidence_refs)
        complete = bool(
            scenario.event_definition.strip() and scenario.boundary_conditions
            and scenario.horizon.strip() and scenario.reasoning.strip()
        )
        scenario_label = _SCENARIO_LABEL.get(scenario.name, scenario.name)
        items.append(_make_item(
            assumption_id=f"probability-{scenario.name}", category="情景概率依据",
            title=f"{scenario_label}情景概率依据",
            hypothesis=scenario.event_definition, support_refs=refs, counter_refs=[],
            evidence=evidence,
            missing=[f"补充{scenario_label}情景概率调整所依据的独立当前有效证据"],
            next_verification="核对情景边界、概率调整方向及其逐条证据",
            scenario=scenario.name, variables=[probability_audit.common_state_variable],
            mapping_complete=complete, issues=[] if complete else ["概率事件、边界、期限或推理不完整"],
        ))

    trigger_lookup = {
        trigger.trigger_id: trigger
        for scenario in valuation.scenarios for trigger in scenario.business_triggers
    }
    for row in valuation.scenario_trigger_audit.items:
        source = trigger_lookup.get(row.trigger_id)
        counter_refs = list(getattr(source, "counter_evidence_refs", []) or [])
        complete = bool(
            row.business_change.strip() and row.financial_variable.strip()
            and row.mechanism.strip() and row.direction != "unknown"
        )
        scenario_label = _SCENARIO_LABEL.get(row.scenario or "", row.scenario or "未知")
        variable_label = _VARIABLE_LABEL.get(row.financial_variable, row.financial_variable or "关键变量")
        direction_label = _DIRECTION_LABEL.get(row.direction, row.direction)
        items.append(_make_item(
            assumption_id=row.trigger_id, category="情景业务触发",
            title=f"{scenario_label}情景：{row.business_change or row.trigger_id}",
            hypothesis=(
                f"{row.business_change}使{variable_label}{direction_label}，"
                f"原因是{row.mechanism}"
            ),
            support_refs=list(row.evidence_refs), counter_refs=counter_refs,
            evidence=evidence, missing=[
                f"补充{scenario_label}情景{variable_label}的公司公告、财报或独立行业证据"
            ],
            next_verification=f"验证{row.business_change or row.trigger_id}是否真实发生并量化其财务影响",
            issues=list(row.issues), scenario=row.scenario or "",
            variables=[row.financial_variable], mapping_complete=complete,
        ))

    forecast = valuation.forecast_reasonableness_audit
    forecast_refs = _unique(
        [ref for row in forecast.historical_anchors for ref in row.evidence_refs]
        + [ref for row in forecast.forecast_entries for ref in row.evidence_refs]
        + [ref for row in forecast.scenario_separation.drivers for ref in row.evidence_refs]
    )
    forecast_logic = (
        "逻辑错误" if not forecast.margin_bridge_passed
        or any("不能解释" in issue for issue in forecast.issues)
        else "逻辑完整"
    )
    items.append(_make_item(
        assumption_id="forecast-reasonableness", category="预测合理性",
        title="预测合理性总假设",
        hypothesis="三种情景预测能够由历史锚点、财务传导关系和逐变量经营证据共同解释",
        support_refs=forecast_refs, counter_refs=[], evidence=evidence,
        missing=list(forecast.evidence_needed),
        next_verification="补齐未被历史数据或变量级证据解释的预测变量",
        issues=list(forecast.issues), variables=[forecast.key_driver],
        mapping_complete=forecast.passed, logic_status=forecast_logic,
    ))

    for row in forecast.scenario_separation.drivers:
        variable_label = _VARIABLE_LABEL.get(row.variable, row.label)
        items.append(_make_item(
            assumption_id=f"scenario-separation-{row.year}-{row.variable}", category="情景差异",
            title=f"{row.year}年{variable_label}情景差异",
            hypothesis=row.explanation or f"悲观、基准、乐观情景的{variable_label}差异来自真实经营变化",
            support_refs=list(row.evidence_refs), counter_refs=[], evidence=evidence,
            missing=[f"补充{row.year}年{variable_label}在三种情景下的逐变量证据"],
            next_verification=f"用公司经营数据验证{variable_label}的情景差异及方向",
            issues=[] if row.supported else [row.explanation], year=row.year,
            variables=[row.variable], mapping_complete=row.supported,
            logic_status="逻辑完整" if row.variable and row.year else "逻辑不完整",
        ))

    margin = forecast.margin_expansion_evidence
    for row in margin.items:
        support_refs = _unique([
            ref for driver in row.drivers for ref in driver.evidence_refs
        ])
        counter_refs = _unique([
            ref for driver in row.drivers
            for ref in getattr(driver, "counter_evidence_refs", [])
        ])
        logic = _logic_status(list(row.issues), bool(row.drivers))
        items.append(_make_item(
            assumption_id=f"margin-expansion-{row.year}", category="毛利率提升",
            title=f"{row.year}年乐观情景毛利率提升",
            hypothesis=row.causal_chain or "经营驱动能够解释乐观情景相对基准情景的毛利率提升",
            support_refs=support_refs, counter_refs=counter_refs, evidence=evidence,
            missing=list(margin.evidence_needed),
            next_verification="寻找分业务毛利率、产品价格、成本、业务结构和规模效应数据",
            issues=list(row.issues), year=row.year, variables=["gross_margin"],
            critical=row.material, mapping_complete=row.supported, logic_status=logic,
        ))

    for scenario in valuation.scenarios:
        scenario_label = _SCENARIO_LABEL.get(scenario.name, scenario.name)
        terminal = scenario.projections[-1] if scenario.projections else None
        candidates = terminal.valuation_candidates if terminal else []
        refs = _unique([ref for candidate in candidates for ref in candidate.evidence_refs])
        supported = any(candidate.available and candidate.evidence_supported for candidate in candidates)
        candidate_issues = _unique([
            issue for candidate in candidates for issue in candidate.evidence_validation_issues
        ])
        items.append(_make_item(
            assumption_id=f"comparable-{scenario.name}", category="可比估值",
            title=f"{scenario_label}情景终值估值方法适用性",
            hypothesis="至少一种估值方法具有同口径、同期间且适用于该情景的独立可比证据",
            support_refs=refs, counter_refs=[], evidence=evidence,
            missing=[f"补充{scenario_label}情景终值所需的同行估值或公司历史估值区间，并注明日期、期间和会计口径"],
            next_verification="优先寻找同行最新财务与估值数据，以及公司历史估值区间",
            issues=candidate_issues, scenario=scenario.name,
            variables=["valuation_multiple"], mapping_complete=supported,
            logic_status="逻辑完整" if terminal and candidates else "逻辑不完整",
        ))

    for catalyst in catalysts.catalysts:
        complete = bool(
            catalyst.expected_date and catalyst.observable_metric.strip()
            and catalyst.verifies_assumption_ids and catalyst.affected_financial_variables
            and catalyst.favorable_result.strip() and catalyst.adverse_result.strip()
            and catalyst.scenario_probability_effects
        )
        items.append(_make_item(
            assumption_id=catalyst.catalyst_id, category="催化因素",
            title=catalyst.event or catalyst.catalyst_id,
            hypothesis=(
                f"{catalyst.event}将在{catalyst.expected_date or catalyst.timeframe}验证"
                f"{', '.join(catalyst.affected_financial_variables)}并影响市场预期"
            ),
            support_refs=list(catalyst.evidence_refs),
            counter_refs=list(getattr(catalyst, "counter_evidence_refs", []) or []),
            evidence=evidence,
            missing=["补充该事件时间、观察指标及结果方向的公司公告或独立可靠证据"],
            next_verification=catalyst.observable_metric,
            issues=[] if complete else ["催化因素的时间、指标、假设映射或双向结果不完整"],
            variables=list(catalyst.affected_financial_variables), mapping_complete=complete,
        ))

    for driver in drivers:
        complete = bool(
            driver.claim.strip() and driver.mechanism.strip()
            and driver.verification_metric.strip() and driver.time_window.strip()
        )
        items.append(_make_item(
            assumption_id=driver.driver_id, category="预期差驱动",
            title=driver.claim or driver.driver_id, hypothesis=driver.mechanism,
            support_refs=list(driver.evidence_refs),
            counter_refs=list(driver.counter_evidence_refs), evidence=evidence,
            missing=[f"补充能够直接量化‘{driver.claim}’的公司或行业数据"],
            next_verification=driver.verification_metric,
            issues=list(driver.issues), variables=[driver.driver_type],
            mapping_complete=complete,
            explicitly_refuted=driver.driver_id in refuted_driver_ids,
        ))

    for step in investment_logic.steps:
        complete = bool(step.premise.strip() and step.conclusion.strip() and step.mechanism.strip())
        logic = _logic_status(list(step.issues), complete)
        items.append(_make_item(
            assumption_id=step.step_id, category="投资逻辑", title=step.conclusion or step.step_id,
            hypothesis=f"{step.premise} → {step.mechanism} → {step.conclusion}",
            support_refs=list(step.evidence_refs), counter_refs=[], evidence=evidence,
            missing=[f"补充能够直接验证投资逻辑步骤{step.step_id}前提和传导关系的证据"],
            next_verification=step.conclusion or step.premise,
            issues=list(step.issues), mapping_complete=complete and step.passed,
            logic_status=logic,
        ))

    counts = Counter(item.current_status for item in items)
    critical = [item for item in items if item.critical]
    verified = [item for item in critical if item.current_status == "已验证"]
    has_refuted = any(item.current_status == "已证伪" for item in critical)
    has_conflicts = any(item.current_status == "互相矛盾" for item in critical)
    has_logic_error = any(item.logic_status == "逻辑错误" for item in critical)
    all_verified = bool(critical) and len(verified) == len(critical)
    actionable = bool(all_verified and not has_refuted and not has_conflicts and not has_logic_error)
    if has_refuted:
        conclusion = "至少一个关键假设已被有效反证，相关投资逻辑不能成立。"
    elif has_conflicts:
        conclusion = "关键假设存在互相矛盾的高质量证据，当前不能形成单向结论。"
    elif all_verified:
        conclusion = "所有关键假设均已通过统一证据充分性检查。"
    else:
        conclusion = "关键假设仍有证据不足；这表示暂不能判断，不表示假设错误或已被证伪。"
    return EvidenceStatusLayer(
        items=items,
        status_counts={status: counts.get(status, 0) for status in _STATUS_ORDER},
        critical_item_count=len(critical), critical_verified_count=len(verified),
        critical_all_verified=all_verified, has_refuted_assumption=has_refuted,
        has_conflicting_evidence=has_conflicts, has_logic_error=has_logic_error,
        actionable_edge_allowed=actionable, one_sentence_conclusion=conclusion,
        why="统一检查来源等级、证据日期、独立来源、变量对应关系、反方证据和投资逻辑完整性。",
        data="；".join(f"{status}{counts.get(status, 0)}项" for status in _STATUS_ORDER),
        implication=(
            "只有关键项全部已验证且没有有效反证、证据冲突或逻辑错误，才允许继续判断可执行投资优势。"
        ),
    )


def _preferred_sources(item: UnifiedEvidenceStatusItem) -> list[str]:
    text = f"{item.category} {item.title} {' '.join(item.financial_variables)}"
    sources = ["公司公告", "最新财报"]
    if any(word in text for word in ("收入", "订单", "revenue")):
        sources += ["订单或合同", "产品出货量", "市占率数据"]
    if any(word in text for word in ("毛利", "价格", "成本", "gross_margin")):
        sources += ["分业务毛利率数据", "行业价格数据", "产品成本数据"]
    if any(word in text for word in ("估值", "倍数", "valuation")):
        sources += ["同行财务数据", "公司历史估值区间"]
    if item.category == "催化因素":
        sources += ["交易所披露", "正式合同或项目公告"]
    return _unique(sources)


def build_evidence_acquisition_plan(layer: EvidenceStatusLayer) -> EvidenceAcquisitionPlan:
    candidates: list[tuple[int, UnifiedEvidenceStatusItem]] = []
    status_bonus = {"无法验证": 20, "部分验证": 12, "互相矛盾": 10}
    for item in layer.items:
        if not item.critical or item.current_status not in status_bonus:
            continue
        score = _CATEGORY_PRIORITY.get(item.category, 60) + status_bonus[item.current_status]
        candidates.append((score, item))
    candidates.sort(key=lambda pair: (-pair[0], pair[1].assumption_id))
    plans: list[EvidenceAcquisitionPlanItem] = []
    seen_targets: set[str] = set()
    for score, item in candidates:
        target = item.missing_evidence[0] if item.missing_evidence else item.next_verification
        target = target or f"直接验证“{item.hypothesis}”的量化证据"
        target_key = "".join(target.split()).lower()
        if target_key in seen_targets:
            continue
        seen_targets.add(target_key)
        rank = len(plans) + 1
        plans.append(EvidenceAcquisitionPlanItem(
            rank=rank, plan_id=f"evidence-plan-{rank:02d}",
            assumption_id=item.assumption_id, assumption_title=item.title,
            current_status=item.current_status,
            evidence_to_find=target,
            preferred_sources=_preferred_sources(item),
            why_high_value=f"该项属于{item.category}，当前状态为{item.current_status}，直接阻断后续判断。",
            expected_unblock=[item.category, "统一证据充分性检查"],
            suggested_search=f"{item.title} 最新 公告 财报 数据",
            priority_score=score,
        ))
    if plans:
        conclusion = f"优先获取排名前{min(3, len(plans))}项证据，最有可能解除当前关键阻断。"
    else:
        conclusion = "当前没有因证据不足而需要新增获取的关键证据。"
    return EvidenceAcquisitionPlan(
        items=plans, one_sentence_conclusion=conclusion,
        why="按关键关卡的重要性、证据缺口严重程度和对解除阻断的直接价值排序。",
        data=f"共生成{len(plans)}项证据获取任务。",
        implication="证据获取计划只说明下一步查什么，不会自行令任何关卡通过。",
    )

def build_evidence_based_verdict(
    layer: EvidenceStatusLayer,
    plan: EvidenceAcquisitionPlan,
    legacy_verdict: MispricingVerdict,
    valuation: ValuationOutput,
    significant_gap: bool,
    edge_available: bool,
) -> EvidenceBasedInvestmentVerdict:
    critical_refuted = [
        item for item in layer.items if item.critical and item.current_status == "已证伪"
    ]
    critical_conflicts = [
        item for item in layer.items if item.critical and item.current_status == "互相矛盾"
    ]
    logic_errors = [
        item for item in layer.items if item.critical and item.logic_status == "逻辑错误"
    ]
    insufficient = [
        item for item in layer.items
        if item.critical and item.current_status in {"部分验证", "无法验证"}
    ]
    target_allowed = bool(
        valuation.valuation_audit.passed
        and all(item.target_price is not None for item in valuation.scenarios)
    )
    return_allowed = bool(
        target_allowed and valuation.risk_reward_audit.passed
        and valuation.expected_price_return is not None
    )

    if critical_refuted or legacy_verdict.status == "INVALIDATED":
        status = "已证伪：不成立"
        reason_type = "有效反证"
        key_items = critical_refuted
        conclusion = "核心投资假设已经被满足要求的有效反证推翻。"
        why = "至少一个关键假设的反方证据通过来源、日期和独立性检查。"
        implication = "停止沿用该投资逻辑；除非出现新的高质量证据，不生成目标价、预期收益率或可执行投资优势。"
    elif critical_conflicts:
        status = "证据不足：暂不能判断"
        reason_type = "证据矛盾"
        key_items = critical_conflicts
        conclusion = "高质量证据互相矛盾，当前无法判断投资假设是否成立。"
        why = "同一关键假设的支持证据与反方证据都通过检查。"
        implication = "先解释证据口径、时间或样本差异；在矛盾消除前保持全部投资输出关闭。"
    elif logic_errors:
        status = "证据不足：暂不能判断"
        reason_type = "投资逻辑错误"
        key_items = logic_errors
        conclusion = "当前投资逻辑的量化传导尚未闭合，暂未得到充分验证。"
        why = "现有经营变量、量化影响或证据尚不能完整解释预测差异；这不等于存在事实反证。"
        implication = "先补全或校正量化传导，再重新验证；在此之前不把它写成投资逻辑已被证伪。"
    elif insufficient or not layer.critical_all_verified:
        status = "证据不足：暂不能判断"
        reason_type = "证据不足"
        key_items = insufficient
        conclusion = "目前没有足够证据形成明确投资优势。"
        why = "关键假设仍缺少满足来源等级、日期、独立性或变量量化关系要求的证据。"
        implication = "目标价、预期收益率和可执行投资优势继续保持关闭，按证据获取计划补充资料。"
    elif not valuation.valuation_audit.passed or not valuation.risk_reward_audit.passed:
        status = "证据不足：暂不能判断"
        reason_type = "证据不足"
        key_items = []
        conclusion = "当前估值或风险收益无法可靠确认，暂不能形成明确投资优势。"
        why = "估值证据、方法输入或必要的数学条件仍有未解除的阻断。"
        implication = "明确显示估值无法确认，不绕过现有安全机制生成目标价或预期收益率。"
    elif edge_available and legacy_verdict.investable:
        status = "已验证且达到可投资条件"
        reason_type = "达到可投资条件"
        key_items = []
        conclusion = "关键假设、估值、风险收益和预期差均已验证并达到严格条件。"
        why = "统一证据层与全部原有检查关卡均通过。"
        implication = "系统允许输出可执行投资优势，但仍不构成自动交易或投资建议。"
    elif significant_gap:
        status = "已验证且存在预期差"
        reason_type = "已验证预期差"
        key_items = []
        conclusion = "关键假设已经验证并存在预期差，但尚未达到可投资条件。"
        why = "证据链和预期差成立，但严格投资优势阈值尚未全部满足。"
        implication = "继续跟踪催化因素、估值和风险收益，不自动升级为买入结论。"
    else:
        status = "已验证但估值不划算"
        reason_type = "估值不划算"
        key_items = []
        conclusion = "关键假设已经验证，但当前价格没有提供足够风险补偿。"
        why = "估值或预期收益率没有达到系统要求。"
        implication = "逻辑成立不等于价格合适，当前不形成可执行投资优势。"

    key_ids = [item.assumption_id for item in key_items[:8]]
    next_evidence = [item.evidence_to_find for item in plan.items[:5]]
    return EvidenceBasedInvestmentVerdict(
        status=status, primary_reason_type=reason_type,
        one_sentence_conclusion=conclusion, why=why,
        data=(
            f"关键假设{layer.critical_item_count}项，已验证{layer.critical_verified_count}项；"
            f"已证伪{layer.status_counts.get('已证伪', 0)}项，"
            f"互相矛盾{layer.status_counts.get('互相矛盾', 0)}项。"
        ),
        implication=implication,
        target_price_allowed=target_allowed and status not in {"已证伪：不成立", "证据不足：暂不能判断"},
        expected_return_allowed=return_allowed and status not in {"已证伪：不成立", "证据不足：暂不能判断"},
        actionable_edge_allowed=status == "已验证且达到可投资条件",
        key_assumption_ids=key_ids, next_evidence=next_evidence,
    )
