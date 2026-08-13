from __future__ import annotations

from datetime import date, datetime
from math import prod
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


def _normalize_rate(value: Any) -> Any:
    if value is None or value == "":
        return value
    number = float(value)
    return number / 100 if abs(number) > 1 else number


class Evidence(BaseModel):
    id: str = Field(default_factory=lambda: f"ev_{uuid4().hex[:12]}")
    fact: str
    excerpt: str = ""
    source: str
    source_url: str = ""
    source_type: str = "other"
    source_grade: Literal["A", "B", "C"] = "C"
    publication_date: date | None = None
    information_date: date | None = None
    available_at: date | None = None
    analysis_date: date = Field(default_factory=date.today)
    retrieved_at: datetime = Field(default_factory=datetime.now)
    confidence: float = Field(ge=0, le=1)
    freshness_days: int | None = None
    temporal_scope: Literal["current", "historical", "future", "unknown"] = "unknown"
    supports_current_claim: bool = False
    invalid_reason: str = ""

    @model_validator(mode="after")
    def derive_freshness(self) -> "Evidence":
        references = [item for item in (self.available_at, self.publication_date, self.information_date) if item]
        future = [item for item in references if item > self.analysis_date]
        if future:
            self.freshness_days = (self.analysis_date - max(future)).days
            self.temporal_scope = "future"
            self.supports_current_claim = False
            self.invalid_reason = f"Look-ahead Bias：证据日期 {max(future).isoformat()} 晚于分析日 {self.analysis_date.isoformat()}"
            return self
        reference = self.available_at or self.publication_date or self.information_date
        if reference:
            self.freshness_days = (self.analysis_date - reference).days
            self.temporal_scope = "current" if self.freshness_days <= 180 else "historical"
        self.supports_current_claim = bool(
            self.source_grade in {"A", "B"}
            and self.temporal_scope == "current"
            and self.source_type != "collection_error"
        )
        return self


class EvidenceGateResult(BaseModel):
    passed: bool = False
    rule: str = "当前有效A级≥1，或来自两个独立来源的当前有效B级≥2"
    current_a_count: int = 0
    current_b_source_count: int = 0
    accepted_evidence_refs: list[str] = []
    rejected_evidence_refs: list[str] = []
    reason: str = "尚未执行证据门控"


class ScenarioProbabilityAssumption(BaseModel):
    name: Literal["bear", "base", "bull"]
    base_probability: float = Field(ge=0, le=1)
    event_definition: str
    boundary_conditions: list[str]
    horizon: str
    reasoning: str
    evidence_refs: list[str] = []
    probability_basis_refs: list[str] = Field(default_factory=list)


class ProbabilityAdjustmentStep(BaseModel):
    step_id: str = ""
    adjustment_type: str = ""
    prior_probability: float | None = Field(default=None, ge=0, le=1)
    adjustment_factor: float | None = None
    adjusted_probability: float | None = Field(default=None, ge=0, le=1)
    reasoning: str = ""
    formula: str = ""
    evidence_refs: list[str] = Field(default_factory=list)
    status: Literal["available", "unavailable", "blocked"] = "unavailable"


class ScenarioProbabilityOutput(BaseModel):
    common_state_variable: str
    mutually_exclusive_reasoning: str
    exhaustive_reasoning: str
    scenarios: list[ScenarioProbabilityAssumption]
    limitations: list[str] = []


class PreValuationProbabilityAudit(BaseModel):
    passed: bool = False
    exact_scenarios: bool = False
    probability_sum: float = 0
    mutually_exclusive: bool = False
    exhaustive: bool = False
    mutually_exclusive_reasoning: str = ""
    exhaustive_reasoning: str = ""
    common_state_variable: str = ""
    common_horizon: bool = False
    evidence_coverage: dict[str, bool] = {}
    scenarios: list[ScenarioProbabilityAssumption] = []
    issues: list[str] = []
    evidence_needed: list[str] = []


class ComparableValuationEvidenceItem(BaseModel):
    subject: str = ""
    symbol: str = ""
    relationship: Literal["peer", "company_historical", "other"] = "other"
    method: Literal["pe", "ev_ebitda", "ev_sales", "pb"]
    metric: str = ""
    value: float | None = None
    range_low: float | None = None
    range_high: float | None = None
    as_of: str = ""
    period: str = ""
    source: str = ""
    evidence_ref: str = ""
    applicability: str = ""
    accounting_basis: str = ""
    valid: bool = False
    reason: str = ""


class ComparableValuationEvidenceAudit(BaseModel):
    passed: bool = False
    selected_methods_supported: bool = False
    evidence_items: list[ComparableValuationEvidenceItem] = []
    methods_available: dict[str, bool] = {}
    selected_methods: dict[str, str] = {}
    issues: list[str] = []
    evidence_needed: list[str] = []


class MarginExpansionDriverAssumption(BaseModel):
    factor_type: Literal[
        "product_mix", "business_mix", "price", "cost", "scale", "market_share", "other"
    ]
    description: str
    gross_margin_impact: float
    evidence_refs: list[str] = []
    counter_evidence_refs: list[str] = []

    @field_validator("gross_margin_impact", mode="before")
    @classmethod
    def normalize_impact(cls, value: Any) -> Any:
        return _normalize_rate(value)


class ValidationIssue(BaseModel):
    check: str
    severity: Literal["info", "warning", "error"]
    passed: bool
    expected: str = ""
    actual: str = ""
    message: str = ""


class DataConsistencyReport(BaseModel):
    passed: bool = False
    valuation_allowed: bool = False
    checked_at: datetime = Field(default_factory=datetime.now)
    issues: list[ValidationIssue] = []
    summary: str = "尚未执行数据一致性校验"


class MarketSnapshot(BaseModel):
    as_of: date
    price: float | None = None
    previous_close: float | None = None
    day_open: float | None = None
    day_high: float | None = None
    day_low: float | None = None
    high_52w: float | None = None
    low_52w: float | None = None
    shares_outstanding: float | None = None
    shares_source: str = ""
    market_cap: float | None = None
    pe_ttm: float | None = None
    eps_ttm: float | None = None
    pb: float | None = None
    book_value_per_share: float | None = None
    currency: str = "CNY"
    unit: str = "亿元"
    shares_unit: str = "亿股"
    evidence_refs: list[str] = []
    consistency_report: DataConsistencyReport = Field(default_factory=DataConsistencyReport)


class ResearchRequest(BaseModel):
    company: str
    symbol: str = ""
    analysis_date: date = Field(default_factory=date.today)
    search_depth: Literal["basic", "deep"] = "basic"


class ResearchOutput(BaseModel):
    company: str
    symbol: str = ""
    market_snapshot: MarketSnapshot | None = None
    comparable_valuation_catalog: list[ComparableValuationEvidenceItem] = []
    documents: list[str] = []
    events: list[str] = []
    products: list[str] = []
    customers: list[str] = []
    partners: list[str] = []
    capacity: list[str] = []
    industry_events: list[str] = []
    evidence: list[Evidence] = []
    limitations: list[str] = []


class FinancialPeriod(BaseModel):
    period: str
    revenue: float | None = None
    gross_profit: float | None = None
    gross_margin: float | None = None
    operating_profit: float | None = None
    ebit: float | None = None
    ebitda: float | None = None
    depreciation_amortization: float | None = None
    ebit_margin: float | None = None
    ebitda_margin: float | None = None
    depreciation_amortization_ratio: float | None = None
    net_profit: float | None = None
    net_margin: float | None = None
    eps: float | None = None
    operating_cash_flow: float | None = None
    roe: float | None = None
    capex: float | None = None
    receivables: float | None = None
    inventory: float | None = None
    debt: float | None = None
    ebit_basis: str = ""
    evidence_refs: list[str] = []
    field_evidence_refs: dict[str, list[str]] = {}

    @field_validator(
        "gross_margin", "ebit_margin", "ebitda_margin",
        "depreciation_amortization_ratio", "net_margin", "roe", mode="before",
    )
    @classmethod
    def normalize_rates(cls, value: Any) -> Any:
        return _normalize_rate(value)

    @model_validator(mode="after")
    def derive_historical_margins(self) -> "FinancialPeriod":
        if self.revenue not in {None, 0}:
            if self.gross_margin is None and self.gross_profit is not None:
                self.gross_margin = self.gross_profit / self.revenue
            if self.ebit_margin is None and self.ebit is not None:
                self.ebit_margin = self.ebit / self.revenue
            if self.ebitda_margin is None and self.ebitda is not None:
                self.ebitda_margin = self.ebitda / self.revenue
            if self.depreciation_amortization_ratio is None and self.depreciation_amortization is not None:
                self.depreciation_amortization_ratio = self.depreciation_amortization / self.revenue
            if self.net_margin is None and self.net_profit is not None:
                self.net_margin = self.net_profit / self.revenue
        return self


class FinancialOutput(BaseModel):
    currency: str = "CNY"
    unit: str = "亿元"
    historical: list[FinancialPeriod] = []
    revenue_sources: list[str] = []
    segment_revenue: dict[str, str] = {}
    regional_revenue: dict[str, str] = {}
    trends: list[str] = []
    quality_flags: list[str] = []
    evidence_refs: list[str] = []
    limitations: list[str] = []


class IndustryOutput(BaseModel):
    industry: str
    industry_size: dict[str, str] = {}
    growth_drivers: list[str] = []
    competitors: list[str] = []
    upstream: list[str] = []
    downstream: list[str] = []
    technology_trends: list[str] = []
    policy_drivers: list[str] = []
    nonlinear_profit_drivers: list[str] = []
    industry_risks: list[str] = []
    evidence_refs: list[str] = []
    limitations: list[str] = []


class OpportunityMapping(BaseModel):
    mapping_id: str = Field(default_factory=lambda: f"map_{uuid4().hex[:8]}")
    industry_change: str
    company_capability: str
    capture_mechanism: str
    industry_growth_rate: float | None = None
    company_revenue_growth_rate: float | None = None
    current_market_share: float | None = None
    target_market_share: float | None = None
    market_share_change: float | None = None
    company_capture_rate: float | None = None
    gross_margin_impact: float | None = None
    net_profit_impact: float | None = None
    transmission_lag_months: int | None = None
    target_period: str = ""
    blockers: list[str] = []
    confidence: float = Field(ge=0, le=1)
    evidence_refs: list[str] = []
    evidence_gate: EvidenceGateResult = Field(default_factory=EvidenceGateResult)
    validation_status: Literal["validated", "hypothesis", "invalid"] = "hypothesis"

    @field_validator(
        "industry_growth_rate", "company_revenue_growth_rate", "current_market_share",
        "target_market_share", "market_share_change",
        "gross_margin_impact", "net_profit_impact", mode="before",
    )
    @classmethod
    def normalize_rates(cls, value: Any) -> Any:
        return _normalize_rate(value)


class CompanyOutput(BaseModel):
    products: list[str] = []
    competitive_strengths: list[str] = []
    technology_barriers: list[str] = []
    customers_and_channels: list[str] = []
    capacity: list[str] = []
    market_share: list[str] = []
    cost_and_supply_chain: list[str] = []
    overseas_capability: list[str] = []
    management_and_capex: list[str] = []
    new_businesses: list[str] = []
    opportunity_mapping: list[OpportunityMapping] = []
    limitations: list[str] = []


class ProbabilityAssessment(BaseModel):
    base_probability: float = Field(default=0, ge=0, le=1)
    evidence_factor: float = Field(default=1, ge=0, le=1.5)
    raw_adjusted_probability: float = Field(default=0, ge=0)
    normalization_denominator: float = Field(default=0, ge=0)
    normalized_probability: float = Field(default=0, ge=0, le=1)
    adjusted_probability: float = Field(default=0, ge=0)
    confidence: float = Field(default=0, ge=0, le=1)
    formula: str = "raw=基础概率×证据修正因子；final=raw÷三情景raw合计"
    rationale: str = ""


class LogicTreeNode(BaseModel):
    node_id: str
    label: str
    metric: str = ""
    value: float | None = None
    unit: str = ""
    period: str = ""
    depends_on: list[str] = []
    probability: float | None = Field(default=None, ge=0, le=1)
    evidence_refs: list[str] = []


class LogicTree(BaseModel):
    tree_type: Literal["investment", "financial", "risk"]
    root_id: str = ""
    nodes: list[LogicTreeNode] = []
    summary: str = ""


class ProbabilityCondition(BaseModel):
    condition_id: str
    description: str
    probability: float = Field(ge=0, le=1)
    conditional_on: list[str] = []
    rationale: str
    evidence_refs: list[str] = []


class StoryNode(BaseModel):
    node_id: str
    name: str
    story_type: Literal["old", "current", "potential", "extreme_bull"]
    thesis: str
    depends_on: list[str] = []
    overlaps_with: list[str] = []
    causal_link: str = ""
    why_now: str
    current_progress: str
    demand: str
    product: str
    customers: str
    revenue_path: str
    profit_path: str
    time_horizon: str
    market_awareness: str
    conditions: list[ProbabilityCondition] = []
    base_conditional_probability: float = Field(default=0, ge=0, le=1)
    evidence_factor: float = Field(default=1, ge=0, le=1.5)
    conditional_probability: float = Field(default=0, ge=0, le=1)
    probability: float = Field(default=0, ge=0, le=1)
    probability_assessment: ProbabilityAssessment = Field(default_factory=ProbabilityAssessment)
    probability_trace: list[str] = []
    critical_uncertainty: str = ""
    is_key_node: bool = False
    evidence_for: list[str] = []
    evidence_against: list[str] = []
    evidence_gate: EvidenceGateResult = Field(default_factory=EvidenceGateResult)
    claim_status: Literal["supported", "hypothesis", "unknown"] = "unknown"
    next_evidence: list[str] = []

    @model_validator(mode="after")
    def calculate_conditional_probability(self) -> "StoryNode":
        if self.conditions:
            known = {item.condition_id for item in self.conditions}
            ordered: list[ProbabilityCondition] = []
            pending = list(self.conditions)
            while pending:
                ready = [item for item in pending if all(dep in {x.condition_id for x in ordered} for dep in item.conditional_on if dep in known)]
                item = ready[0] if ready else pending[0]
                ordered.append(item)
                pending.remove(item)
            self.conditions = ordered
            self.base_conditional_probability = prod(item.probability for item in ordered)
        elif self.base_conditional_probability == 0 and (self.conditional_probability > 0 or self.probability > 0):
            self.base_conditional_probability = self.conditional_probability or self.probability
        self.conditional_probability = min(1.0, self.base_conditional_probability * self.evidence_factor)
        self.probability = self.conditional_probability
        prior = self.probability_assessment
        self.probability_assessment = ProbabilityAssessment(
            base_probability=self.base_conditional_probability,
            evidence_factor=self.evidence_factor,
            adjusted_probability=self.conditional_probability,
            confidence=prior.confidence or min(1.0, self.evidence_factor),
            rationale=prior.rationale,
        )
        return self


class StoryOutput(BaseModel):
    stories: list[StoryNode] = []
    investment_tree: LogicTree = Field(default_factory=lambda: LogicTree(tree_type="investment"))
    evolution_summary: str
    key_story_change: str
    key_node_ids: list[str] = []
    overall_probability: float = Field(default=0, ge=0, le=1)
    bottleneck_node_id: str = ""
    probability_method: str = "祖先闭包内各节点条件概率按拓扑顺序联合相乘，每个节点只计一次"
    limitations: list[str] = []

    @model_validator(mode="after")
    def calculate_dependency_probabilities(self) -> "StoryOutput":
        nodes: dict[str, StoryNode] = {}
        for item in self.stories:
            if item.node_id in nodes:
                self.limitations.append(f"重复故事节点 {item.node_id} 已忽略后一个节点")
                continue
            nodes[item.node_id] = item
        self.stories = list(nodes.values())
        for node in self.stories:
            missing = [parent for parent in node.depends_on if parent not in nodes]
            if missing:
                self.limitations.append(f"{node.node_id} 缺少父节点：{', '.join(missing)}")
                node.depends_on = [parent for parent in node.depends_on if parent in nodes]

        visiting: set[str] = set()
        visited: set[str] = set()
        order: list[str] = []
        def visit(node_id: str) -> None:
            if node_id in visited:
                return
            if node_id in visiting:
                self.limitations.append(f"检测到概率依赖环，已断开 {node_id} 的父依赖")
                nodes[node_id].depends_on = []
                return
            visiting.add(node_id)
            for parent in list(nodes[node_id].depends_on):
                visit(parent)
            visiting.discard(node_id)
            visited.add(node_id)
            order.append(node_id)
        for node_id in nodes:
            visit(node_id)

        closures: dict[str, set[str]] = {}
        for node_id in order:
            closure = {node_id}
            for parent in nodes[node_id].depends_on:
                closure.update(closures.get(parent, {parent}))
            closures[node_id] = closure
            factors = [member for member in order if member in closure]
            nodes[node_id].probability = prod(nodes[member].conditional_probability for member in factors)
            nodes[node_id].probability_trace = [f"{member}:{nodes[member].conditional_probability:.6f}" for member in factors]

        key_nodes = [nodes[key] for key in self.key_node_ids if key in nodes]
        if not key_nodes:
            key_nodes = [item for item in self.stories if item.is_key_node] or self.stories[-1:]
        terminal_id = ""
        if key_nodes:
            terminal = max(key_nodes, key=lambda item: (len(closures.get(item.node_id, set())), -item.probability))
            terminal_id = terminal.node_id
            self.overall_probability = terminal.probability
            factors = closures.get(terminal.node_id, {terminal.node_id})
            self.bottleneck_node_id = min(factors, key=lambda item: nodes[item].conditional_probability)
        self.investment_tree = LogicTree(
            tree_type="investment",
            root_id=terminal_id,
            nodes=[
                LogicTreeNode(
                    node_id=item.node_id, label=item.name, metric="Bull Thesis",
                    depends_on=item.depends_on, probability=item.probability,
                    evidence_refs=list(dict.fromkeys(item.evidence_for)),
                )
                for item in self.stories
            ],
            summary="上行投资逻辑树：行业变化→公司捕获/份额→收入→利润→估值；风险在独立 Risk Tree 中计算",
        )
        return self


class MarketImpliedYear(BaseModel):
    year: str
    current_price: float | None = None
    shares_outstanding: float | None = None
    current_market_cap: float | None = None
    years_from_now: int = 1
    required_return: float = Field(default=0.10, ge=-0.5, le=1)
    target_market_cap: float | None = None
    assumed_pe: float | None = None
    implied_net_profit: float | None = None
    assumed_net_margin: float | None = None
    implied_revenue: float | None = None
    base_net_profit: float | None = None
    base_revenue: float | None = None
    implied_growth: float | None = None
    implied_net_profit_growth: float | None = None
    implied_revenue_growth: float | None = None
    implied_net_margin: float | None = None
    implied_gross_margin: float | None = None
    implied_ebit_margin: float | None = None
    implied_ebitda_margin: float | None = None
    implied_ebit: float | None = None
    implied_ebitda: float | None = None
    implied_valuation_method: Literal["pe", "ev_ebitda", "ev_sales", "pb", "unavailable"] = "unavailable"
    implied_valuation_multiple: float | None = None
    solution_status: Literal["solved", "partial", "unavailable", "blocked"] = "unavailable"
    formula: str = ""
    assumptions: list[str] = []
    evidence_refs: list[str] = []

    @field_validator(
        "required_return", "assumed_net_margin", "implied_net_margin",
        "implied_gross_margin", "implied_ebit_margin", "implied_ebitda_margin",
        mode="before",
    )
    @classmethod
    def normalize_rates(cls, value: Any) -> Any:
        return _normalize_rate(value)

    @model_validator(mode="after")
    def describe_calculation(self) -> "MarketImpliedYear":
        self.formula = (
            "当前价格×总股本=当前市值；目标市值=当前市值×(1+要求回报率)^年数；"
            "价格隐含净利润=目标市值÷退出PE；价格隐含收入=隐含净利润÷净利率假设"
        )
        return self


class ConsensusEstimate(BaseModel):
    year: str
    revenue: float | None = None
    net_profit: float | None = None
    source_count: int = 0
    as_of: date | None = None
    evidence_refs: list[str] = []
    available: bool = False


class PriceInAssessment(BaseModel):
    claim: str
    status: Literal["priced_in", "not_priced_in", "partly_priced", "unknown"]
    calculation: str
    confidence: float = Field(ge=0, le=1)
    evidence_refs: list[str] = []
    evidence_gate: EvidenceGateResult = Field(default_factory=EvidenceGateResult)


class ExpectationGap(BaseModel):
    metric: str
    period: str
    model_implied_value: float | None = None
    market_consensus_value: float | None = None
    market_consensus_available: bool = False
    agent_forecast_value: float | None = None
    actual_value: float | None = None
    market_implied: float | None = None
    agent_base: float | None = None
    unit: str = ""
    gap_direction: Literal["positive", "negative", "neutral", "unknown"]
    inference_method: str
    evidence_refs: list[str] = []

    @model_validator(mode="after")
    def keep_legacy_fields(self) -> "ExpectationGap":
        self.market_implied = self.model_implied_value if self.model_implied_value is not None else self.market_implied
        self.agent_base = self.agent_forecast_value if self.agent_forecast_value is not None else self.agent_base
        if not self.market_consensus_available:
            self.market_consensus_value = None
        return self


class ExpectationOutput(BaseModel):
    current_market_story: list[str] = []
    price_implied_path: list[MarketImpliedYear] = []
    market_implied_path: list[MarketImpliedYear] = []
    consensus_estimates: list[ConsensusEstimate] = []
    consensus_available: bool = False
    price_in_assessments: list[PriceInAssessment] = []
    priced_in: list[str] = []
    not_priced_in: list[str] = []
    implied_assumptions: list[str] = []
    expectation_gaps: list[ExpectationGap] = []
    revision_signals: list[str] = []
    data_gaps: list[str] = []
    limitations: list[str] = []

    @model_validator(mode="after")
    def keep_legacy_path(self) -> "ExpectationOutput":
        if self.price_implied_path:
            self.market_implied_path = self.price_implied_path
        elif self.market_implied_path:
            self.price_implied_path = self.market_implied_path
        return self


class ValuationMethodResult(BaseModel):
    method: Literal["pe", "ev_ebitda", "ev_sales", "pb"]
    available: bool = False
    selected: bool = False
    multiple: float | None = None
    multiple_source: str = "scenario_assumption"
    evidence_supported: bool = False
    enterprise_value: float | None = None
    equity_value: float | None = None
    target_price: float | None = None
    formula: str = ""
    reason: str = ""
    evidence_refs: list[str] = []
    evidence_validation_issues: list[str] = []
    comparable_evidence: list[ComparableValuationEvidenceItem] = []


class FinancialBridge(BaseModel):
    year: str
    revenue_growth: float
    gross_margin: float
    rd_expense_ratio: float
    selling_expense_ratio: float
    admin_expense_ratio: float
    other_operating_expense_ratio: float = 0
    financial_expense_ratio: float = 0
    other_income_ratio: float = 0
    tax_rate: float = 25
    minority_interest_ratio: float = 0
    depreciation_amortization_ratio: float = 0
    preferred_loss_method: Literal["pe", "ev_ebitda", "ev_sales", "pb"] = "ev_ebitda"
    pe_multiple: float = 0
    ev_sales_multiple: float = 0
    ev_ebitda_multiple: float = 0
    pb_multiple: float = 0
    cash: float = 0
    debt: float = 0
    book_value: float = 0
    revenue: float = 0
    gross_profit: float = 0
    rd_expense: float = 0
    selling_expense: float = 0
    admin_expense: float = 0
    other_operating_expense: float = 0
    operating_profit: float = 0
    financial_expense: float = 0
    other_income: float = 0
    pre_tax_profit: float = 0
    tax: float = 0
    minority_interest: float = 0
    net_profit: float = 0
    eps: float | None = None
    ebit: float = 0
    ebitda: float = 0
    free_cash_flow: float | None = None
    fcf_unavailable_reason: str = "缺少资本开支和营运资本变动数据，未计算FCF"
    net_debt: float = 0
    valuation_method: Literal["pe", "ev_ebitda", "ev_sales", "pb", "unavailable"] = "unavailable"
    valuation_multiple: float = 0
    enterprise_value: float | None = None
    equity_value: float | None = None
    implied_market_cap: float | None = None
    valuation_formula: str = ""
    method_selection_reason: str = ""
    unavailable_methods: list[str] = []
    valuation_candidates: list[ValuationMethodResult] = []
    method_warning: str = ""
    calculation_valid: bool = False
    calculation_checks: list[ValidationIssue] = []
    assumptions: list[str] = []
    evidence_refs: list[str] = []
    assumption_evidence_refs: dict[str, list[str]] = {}
    assumption_counter_evidence_refs: dict[str, list[str]] = {}
    margin_expansion_drivers: list[MarginExpansionDriverAssumption] = []

    @field_validator(
        "revenue_growth", "gross_margin", "rd_expense_ratio", "selling_expense_ratio",
        "admin_expense_ratio", "other_operating_expense_ratio", "financial_expense_ratio",
        "other_income_ratio", "tax_rate", "minority_interest_ratio",
        "depreciation_amortization_ratio", mode="before",
    )
    @classmethod
    def normalize_rates(cls, value: Any) -> Any:
        return _normalize_rate(value)

    @model_validator(mode="after")
    def mark_programmatic_fields(self) -> "FinancialBridge":
        if not self.valuation_formula:
            self.valuation_formula = "派生金额由确定性估值引擎计算；LLM只提供增长率、利润率、费用率和倍数假设"
        return self


class FinancialBridgeAssumption(BaseModel):
    year: str
    revenue_growth: float
    gross_margin: float
    rd_expense_ratio: float
    selling_expense_ratio: float
    admin_expense_ratio: float
    other_operating_expense_ratio: float = 0
    financial_expense_ratio: float = 0
    other_income_ratio: float = 0
    tax_rate: float = 0.25
    minority_interest_ratio: float = 0
    depreciation_amortization_ratio: float = 0
    preferred_loss_method: Literal["pe", "ev_ebitda", "ev_sales", "pb"] = "ev_ebitda"
    pe_multiple: float = 0
    ev_sales_multiple: float = 0
    ev_ebitda_multiple: float = 0
    pb_multiple: float = 0
    cash: float = 0
    debt: float = 0
    book_value: float = 0
    assumptions: list[str] = []
    evidence_refs: list[str] = []
    assumption_evidence_refs: dict[str, list[str]] = {}
    margin_expansion_drivers: list[MarginExpansionDriverAssumption] = []

    @field_validator(
        "revenue_growth", "gross_margin", "rd_expense_ratio", "selling_expense_ratio",
        "admin_expense_ratio", "other_operating_expense_ratio", "financial_expense_ratio",
        "other_income_ratio", "tax_rate", "minority_interest_ratio",
        "depreciation_amortization_ratio", mode="before",
    )
    @classmethod
    def normalize_rates(cls, value: Any) -> Any:
        return _normalize_rate(value)


class ScenarioBusinessTrigger(BaseModel):
    trigger_id: str = ""
    business_change: str = ""
    financial_variable: str = ""
    direction: Literal["increase", "decrease", "mixed", "unknown"] = "unknown"
    mechanism: str = ""
    evidence_refs: list[str] = Field(default_factory=list)
    counter_evidence_refs: list[str] = Field(default_factory=list)
    evidence_gate: EvidenceGateResult = Field(default_factory=EvidenceGateResult)


class ScenarioAssumption(BaseModel):
    name: Literal["bear", "base", "bull"]
    probability: float = Field(ge=0, le=1)
    base_probability: float = Field(default=0, ge=0, le=1)
    event_definition: str
    boundary_conditions: list[str]
    horizon: str
    probability_basis: list[str] = []
    projections: list[FinancialBridgeAssumption]
    assumptions: list[str] = []
    evidence_refs: list[str] = []
    business_triggers: list[ScenarioBusinessTrigger] = Field(default_factory=list)


class ValuationAssumptionOutput(BaseModel):
    current_implied_assumptions: list[str] = []
    scenarios: list[ScenarioAssumption]
    limitations: list[str] = []


class Scenario(BaseModel):
    name: Literal["bear", "base", "bull"]
    probability: float = Field(ge=0, le=1)
    base_probability: float = Field(default=0, ge=0, le=1)
    evidence_factor: float = Field(default=1, ge=0, le=1.5)
    probability_assessment: ProbabilityAssessment = Field(default_factory=ProbabilityAssessment)
    event_definition: str = ""
    boundary_conditions: list[str] = []
    horizon: str = ""
    probability_basis: list[str] = []
    projections: list[FinancialBridge] = []
    target_year: str = ""
    revenue: float | None = None
    operating_margin: float | None = None
    net_profit: float | None = None
    valuation_method: str = ""
    valuation_multiple: float | None = None
    enterprise_value: float | None = None
    net_debt: float | None = None
    equity_value: float | None = None
    implied_market_cap: float | None = None
    target_price: float | None = None
    price_return: float | None = None
    annualized_price_return: float | None = None
    assumptions: list[str] = []
    evidence_refs: list[str] = []
    evidence_gate: EvidenceGateResult = Field(default_factory=EvidenceGateResult)
    business_triggers: list[ScenarioBusinessTrigger] = Field(default_factory=list)

    @model_validator(mode="after")
    def populate_legacy_summary(self) -> "Scenario":
        if self.projections:
            target = self.projections[-1]
            self.target_year = target.year
            self.revenue = target.revenue
            self.net_profit = target.net_profit
            self.valuation_method = target.valuation_method
            self.valuation_multiple = target.valuation_multiple
            self.enterprise_value = target.enterprise_value
            self.net_debt = target.net_debt
            self.equity_value = target.equity_value
            self.implied_market_cap = target.implied_market_cap
            years = [str(item.year) for item in self.projections]
            self.horizon = self.horizon or (f"{years[0]}-{years[-1]}" if years else "")
        if self.base_probability == 0 and self.probability > 0:
            self.base_probability = self.probability
        raw = self.base_probability * self.evidence_factor
        prior = self.probability_assessment
        normalized = (
            prior.normalized_probability
            if prior.normalization_denominator > 0 else min(1.0, raw)
        )
        self.probability = normalized
        self.probability_assessment = ProbabilityAssessment(
            base_probability=self.base_probability,
            evidence_factor=self.evidence_factor,
            raw_adjusted_probability=raw,
            normalization_denominator=prior.normalization_denominator,
            normalized_probability=normalized,
            adjusted_probability=raw,
            confidence=prior.confidence or min(1.0, self.evidence_factor),
            rationale=prior.rationale,
        )
        return self


class ForecastComparison(BaseModel):
    year: str
    model_implied_profit: float | None = None
    market_implied_profit: float | None = None
    consensus_profit: float | None = None
    consensus_available: bool = False
    agent_forecast_profit: float | None = None
    agent_base_profit: float | None = None
    gap_vs_implied: float | None = None
    gap_vs_consensus: float | None = None
    direction: Literal["positive", "negative", "neutral", "unknown"] = "unknown"
    explanation: str = ""

    @model_validator(mode="after")
    def keep_legacy_fields(self) -> "ForecastComparison":
        self.market_implied_profit = self.model_implied_profit if self.model_implied_profit is not None else self.market_implied_profit
        self.agent_base_profit = self.agent_forecast_profit if self.agent_forecast_profit is not None else self.agent_base_profit
        if not self.consensus_available:
            self.consensus_profit = None
        return self


class ProbabilityAuditEntry(BaseModel):
    scenario: Literal["bear", "base", "bull"]
    event_definition: str
    boundary_conditions: list[str] = []
    horizon: str = ""
    base_probability: float = Field(ge=0, le=1)
    evidence_factor: float = Field(ge=0, le=1.5)
    raw_adjusted_probability: float = Field(ge=0)
    normalization_denominator: float = Field(ge=0)
    normalized_probability: float = Field(ge=0, le=1)
    formula: str
    evidence_refs: list[str] = []
    reasoning: str = ""
    adjustment_steps: list[ProbabilityAdjustmentStep] = Field(default_factory=list)


class ProbabilityAuditReport(BaseModel):
    passed: bool = False
    mutually_exclusive: bool = False
    exhaustive: bool = False
    base_probability_sum: float = 0
    probability_sum: float = 0
    state_variable: str = "S ∈ {Bear, Base, Bull}"
    entries: list[ProbabilityAuditEntry] = []
    issues: list[str] = []


class ScenarioValuationAudit(BaseModel):
    scenario: Literal["bear", "base", "bull"]
    target_year: str = ""
    revenue: float | None = None
    ebit: float | None = None
    ebitda: float | None = None
    free_cash_flow: float | None = None
    net_profit: float | None = None
    eps: float | None = None
    valuation_method: str = ""
    method_selection_reason: str = ""
    valuation_multiple: float | None = None
    enterprise_value: float | None = None
    net_debt: float | None = None
    equity_value: float | None = None
    implied_market_cap: float | None = None
    target_price: float | None = None
    probability: float = Field(default=0, ge=0, le=1)
    weighted_contribution: float | None = None
    model_implied_profit: float | None = None
    profit_gap_vs_model_implied: float | None = None
    valuation_chain: str = ""


class RequiredEarningsGapEntry(BaseModel):
    scenario: Literal["bear", "base", "bull"]
    year: str
    model_implied_earnings: float | None = None
    agent_forecast_earnings: float | None = None
    required_earnings_gap: float | None = None
    gap_percentage: float | None = None
    status: Literal["shortfall", "meets", "exceeds", "unavailable"] = "unavailable"
    formula: str = ""


class RequiredEarningsGapAudit(BaseModel):
    year: str = ""
    model_implied_earnings: float | None = None
    entries: list[RequiredEarningsGapEntry] = []
    all_scenarios_below: bool = False
    required_earnings_above_bull: bool = False
    model_implied_vs_bull_gap: float | None = None
    model_implied_vs_bull_gap_percentage: float | None = None
    model_implied_vs_bull_conclusion: str = ""
    conclusion: str = "Model-Implied Earnings不可用"


class ValuationGapEntry(BaseModel):
    scenario: Literal["bear", "base", "bull"]
    current_model_implied_price: float | None = None
    current_model_implied_market_cap: float | None = None
    target_price: float | None = None
    price_gap: float | None = None
    price_return: float | None = None
    scenario_method: str = ""
    scenario_multiple: float | None = None
    current_comparable_multiple: float | None = None
    multiple_change: float | None = None
    multiple_change_percentage: float | None = None
    multiple_comparable: bool = False
    note: str = ""


class ValuationGapAudit(BaseModel):
    current_model_implied_price: float | None = None
    current_model_implied_market_cap: float | None = None
    current_valuation_multiples: dict[str, float | None] = {}
    entries: list[ValuationGapEntry] = []
    conclusion: str = ""


class ValuationComparabilityAudit(BaseModel):
    passed: bool = False
    same_valuation_system: bool = False
    same_primary_method: bool = False
    common_primary_method: str = ""
    methods_by_scenario: dict[str, str] = {}
    cross_method_switch: bool = False
    cross_method_sorting_reversal: bool = False
    all_identity_checks_passed: bool = False
    unsupported_multiple_scenarios: list[str] = []
    issues: list[str] = []


class SensitivityAuditItem(BaseModel):
    variable: str
    shock: str
    scenario: str = "base"
    target_price_change: float | None = None
    weighted_target_change: float | None = None
    weighted_target_change_ratio: float | None = None
    calculation: str = ""
    warning: str = ""
    is_primary_driver: bool = False


class HistoricalAnchorAuditItem(BaseModel):
    variable: str
    label: str
    periods: list[str] = []
    historical_values: list[float] = []
    historical_min: float | None = None
    historical_max: float | None = None
    historical_median: float | None = None
    available: bool = False
    basis: str = ""
    evidence_refs: list[str] = []
    issue: str = ""


class ForecastReasonablenessEntry(BaseModel):
    scenario: Literal["bear", "base", "bull"]
    year: str
    variable: str
    label: str
    forecast_value: float | None = None
    historical_min: float | None = None
    historical_max: float | None = None
    reasonableness_lower: float | None = None
    reasonableness_upper: float | None = None
    evidence_refs: list[str] = []
    valid_evidence_refs: list[str] = []
    status: Literal["within_history", "supported_structural_change", "unsupported_forecast"] = "unsupported_forecast"
    key_variable: bool = True
    reason: str = ""


class MarginBridgeAuditItem(BaseModel):
    scenario: Literal["bear", "base", "bull"]
    year: str
    revenue: float = 0
    gross_profit: float = 0
    ebit: float = 0
    depreciation_amortization: float = 0
    ebitda: float = 0
    ebt: float = 0
    net_profit: float = 0
    gross_margin: float = 0
    ebit_margin: float = 0
    ebitda_margin: float = 0
    net_margin: float = 0
    depreciation_amortization_ratio: float = 0
    formula_passed: bool = False
    explanation: str = ""
    evidence_refs: list[str] = []
    issues: list[str] = []


class MarginExpansionEvidenceAuditItem(BaseModel):
    year: str
    base_gross_margin: float = 0
    bull_gross_margin: float = 0
    gross_margin_delta: float = 0
    material: bool = False
    primary_profit_driver: bool = False
    driver_impact_sum: float = 0
    gross_profit_impact: float = 0
    ebit_impact: float = 0
    net_profit_impact: float = 0
    valuation_impact: float | None = None
    drivers: list[MarginExpansionDriverAssumption] = []
    valid_evidence_refs: list[str] = []
    supported: bool = False
    causal_chain: str = ""
    issues: list[str] = []


class MarginExpansionEvidenceAudit(BaseModel):
    passed: bool = False
    items: list[MarginExpansionEvidenceAuditItem] = []
    bull_profit_improvement: float | None = None
    gross_margin_is_primary_driver: bool = False
    issues: list[str] = []
    evidence_needed: list[str] = []


class ScenarioSeparationDriver(BaseModel):
    year: str
    variable: str
    label: str
    bear_value: float | None = None
    base_value: float | None = None
    bull_value: float | None = None
    bear_vs_base: float | None = None
    bull_vs_base: float | None = None
    estimated_bull_vs_base_profit_impact: float | None = None
    evidence_refs: list[str] = []
    supported: bool = False
    explanation: str = ""


class ScenarioSeparationAudit(BaseModel):
    passed: bool = False
    differentiated_variables: list[str] = []
    drivers: list[ScenarioSeparationDriver] = []
    bull_vs_base_profit_improvement: float | None = None
    key_driver: str = ""
    issues: list[str] = []


class ForecastReasonablenessAudit(BaseModel):
    passed: bool = False
    valuation_allowed: bool = False
    historical_anchor_passed: bool = False
    margin_bridge_passed: bool = False
    scenario_separation_passed: bool = False
    margin_expansion_evidence_passed: bool = False
    historical_anchors: list[HistoricalAnchorAuditItem] = []
    forecast_entries: list[ForecastReasonablenessEntry] = []
    margin_bridges: list[MarginBridgeAuditItem] = []
    margin_expansion_evidence: MarginExpansionEvidenceAudit = Field(default_factory=MarginExpansionEvidenceAudit)
    scenario_separation: ScenarioSeparationAudit = Field(default_factory=ScenarioSeparationAudit)
    unsupported_forecasts: list[str] = []
    key_driver: str = ""
    evidence_needed: list[str] = []
    failed_layer: str = ""
    issues: list[str] = []


class ValuationAuditReport(BaseModel):
    passed: bool = False
    complete_scenarios: bool = False
    common_horizon: bool = False
    complete_targets: bool = False
    ordered_targets: bool = False
    probability_sum_valid: bool = False
    valuation_methods_comparable: bool = False
    all_identity_checks_passed: bool = False
    comparability_audit: ValuationComparabilityAudit = Field(default_factory=ValuationComparabilityAudit)
    scenario_rows: list[ScenarioValuationAudit] = []
    probability_weighted_target_price: float | None = None
    expected_price_return: float | None = None
    annualized_expected_return: float | None = None
    weighted_target_formula: str = ""
    expected_return_formula: str = ""
    issues: list[str] = []


class InvestmentVerdict(BaseModel):
    status: Literal["blocked", "insufficient_return", "mixed", "favorable"] = "blocked"
    label: str = "审计阻断"
    summary: str = ""
    required_annual_return: float | None = None
    expected_price_return: float | None = None
    annualized_expected_return: float | None = None
    probability_of_negative_return: float | None = None
    risk_coverage_passed: bool = False
    price_reflection_conclusion: str = "无法仅凭价格反推证明哪些假设已被市场计价"
    failed_layer: str = ""
    valuation_block_reason: str = ""
    current_price_required_earnings: str = ""
    base_forecast_assessment: str = ""
    bull_forecast_assessment: str = ""
    key_financial_driver: str = ""
    evidence_needed: list[str] = []
    derivation: list[str] = []
    positive_gap_drivers: list[str] = []
    negative_gap_drivers: list[str] = []
    blockers: list[str] = []


class ScenarioTriggerAuditItem(BaseModel):
    scenario: Literal["bear", "base", "bull"] | None = None
    trigger_id: str = ""
    business_change: str = ""
    financial_variable: str = ""
    direction: Literal["increase", "decrease", "mixed", "unknown"] = "unknown"
    mechanism: str = ""
    evidence_refs: list[str] = Field(default_factory=list)
    evidence_gate: EvidenceGateResult = Field(default_factory=EvidenceGateResult)
    passed: bool = False
    status: Literal["verified", "unavailable", "blocked"] = "unavailable"
    issues: list[str] = Field(default_factory=list)


class ScenarioTriggerAudit(BaseModel):
    passed: bool = False
    status: Literal["available", "unavailable", "blocked"] = "unavailable"
    items: list[ScenarioTriggerAuditItem] = Field(default_factory=list)
    missing_scenarios: list[str] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    evidence_needed: list[str] = Field(default_factory=list)


class NearestScenarioDistance(BaseModel):
    scenario: Literal["bear", "base", "bull"] | None = None
    total_distance: float | None = None
    normalized_distance: float | None = None
    variable_distances: dict[str, float | None] = Field(default_factory=dict)
    available: bool = False
    formula: str = ""
    issues: list[str] = Field(default_factory=list)


class NearestScenarioMatch(BaseModel):
    status: Literal["matched", "unavailable", "blocked"] = "unavailable"
    nearest_scenario: Literal["bear", "base", "bull"] | None = None
    distance: float | None = None
    distances: list[NearestScenarioDistance] = Field(default_factory=list)
    reasoning: str = ""
    blocked_reason: str = ""


class ValuationMethodApplicabilityItem(BaseModel):
    scenario: Literal["bear", "base", "bull"] | None = None
    method: Literal["pe", "ev_ebitda", "ev_sales", "pb", "unavailable"] = "unavailable"
    applicable: bool = False
    status: Literal["applicable", "not_applicable", "unavailable", "blocked"] = "unavailable"
    financial_metric: str = ""
    metric_value: float | None = None
    required_inputs: list[str] = Field(default_factory=list)
    missing_inputs: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    reasoning: str = ""


class ValuationMethodApplicabilityAudit(BaseModel):
    passed: bool = False
    status: Literal["available", "unavailable", "blocked"] = "unavailable"
    primary_method: str = ""
    items: list[ValuationMethodApplicabilityItem] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)


class DoubleCountingItem(BaseModel):
    item_id: str = ""
    scenario: Literal["bear", "base", "bull"] | None = None
    first_component: str = ""
    second_component: str = ""
    overlap_description: str = ""
    financial_impact: float | None = None
    detected: bool = False
    resolved: bool = False
    resolution: str = ""
    evidence_refs: list[str] = Field(default_factory=list)
    status: Literal["clear", "detected", "unavailable", "blocked"] = "unavailable"


class DoubleCountingAudit(BaseModel):
    passed: bool = False
    status: Literal["clear", "detected", "unavailable", "blocked"] = "blocked"
    items: list[DoubleCountingItem] = Field(default_factory=list)
    unresolved_item_ids: list[str] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)


class ScenarioRiskReward(BaseModel):
    scenario: Literal["bear", "base", "bull"] | None = None
    probability: float | None = Field(default=None, ge=0, le=1)
    current_price: float | None = None
    target_price: float | None = None
    price_return: float | None = None
    annualized_return: float | None = None
    weighted_return: float | None = None
    downside: float | None = None
    upside: float | None = None
    status: Literal["available", "unavailable", "blocked"] = "unavailable"
    reasoning: str = ""


class RiskRewardAudit(BaseModel):
    passed: bool = False
    status: Literal["available", "unavailable", "blocked"] = "blocked"
    scenarios: list[ScenarioRiskReward] = Field(default_factory=list)
    expected_return: float | None = None
    annualized_expected_return: float | None = None
    probability_of_loss: float | None = Field(default=None, ge=0, le=1)
    expected_upside: float | None = None
    expected_downside: float | None = None
    upside_downside_ratio: float | None = None
    issues: list[str] = Field(default_factory=list)
    blocked_reason: str = ""


class DecisionProbabilitySummary(BaseModel):
    status: Literal["available", "unavailable", "blocked"] = "unavailable"
    probability_of_positive_return: float | None = Field(default=None, ge=0, le=1)
    probability_of_negative_return: float | None = Field(default=None, ge=0, le=1)
    probability_of_base_or_better: float | None = Field(default=None, ge=0, le=1)
    probability_of_thesis_success: float | None = Field(default=None, ge=0, le=1)
    expected_return: float | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    reasoning: str = ""
    blockers: list[str] = Field(default_factory=list)


class ValuationOutput(BaseModel):
    current_price: float | None = None
    shares_outstanding: float | None = None
    current_market_cap: float | None = None
    currency: str = "CNY"
    unit: str = "亿元"
    reverse_valuation: list[MarketImpliedYear] = []
    current_implied_assumptions: list[str] = []
    scenarios: list[Scenario] = []
    forecast_comparison: list[ForecastComparison] = []
    financial_tree: LogicTree = Field(default_factory=lambda: LogicTree(tree_type="financial"))
    probability_weighted_market_cap: float | None = None
    probability_weighted_target_price: float | None = None
    expected_price_return: float | None = None
    annualized_expected_return: float | None = None
    probability_audit: ProbabilityAuditReport = Field(default_factory=ProbabilityAuditReport)
    pre_valuation_probability_audit: PreValuationProbabilityAudit = Field(default_factory=PreValuationProbabilityAudit)
    forecast_reasonableness_audit: ForecastReasonablenessAudit = Field(default_factory=ForecastReasonablenessAudit)
    comparable_valuation_evidence_audit: ComparableValuationEvidenceAudit = Field(default_factory=ComparableValuationEvidenceAudit)
    valuation_audit: ValuationAuditReport = Field(default_factory=ValuationAuditReport)
    required_earnings_gap: RequiredEarningsGapAudit = Field(default_factory=RequiredEarningsGapAudit)
    valuation_gap: ValuationGapAudit = Field(default_factory=ValuationGapAudit)
    primary_valuation_method: str = ""
    current_valuation_multiples: dict[str, float | None] = {}
    sensitivity_audit: list[SensitivityAuditItem] = []
    sensitivity: list[str] = []
    scenario_trigger_audit: ScenarioTriggerAudit = Field(default_factory=ScenarioTriggerAudit)
    nearest_scenario_match: NearestScenarioMatch = Field(default_factory=NearestScenarioMatch)
    valuation_method_applicability_audit: ValuationMethodApplicabilityAudit = Field(
        default_factory=ValuationMethodApplicabilityAudit
    )
    double_counting_audit: DoubleCountingAudit = Field(default_factory=DoubleCountingAudit)
    risk_reward_audit: RiskRewardAudit = Field(default_factory=RiskRewardAudit)
    decision_probabilities: DecisionProbabilitySummary = Field(default_factory=DecisionProbabilitySummary)
    deterministic_probability_basis: list["ProbabilityBasis"] = Field(default_factory=list)
    deterministic_investment_verdict: InvestmentVerdict = Field(default_factory=InvestmentVerdict)
    financial_bridge_method: str = "程序化统一估值桥：经营预测→EBIT/EBITDA（FCF数据不足则明确不可用）→Enterprise Value→减净债务→Equity Value→Target Price；三情景优先使用共同可用方法"
    limitations: list[str] = []


class CatalystItem(BaseModel):
    catalyst_id: str
    timeframe: Literal["30d", "quarter", "half_year", "one_year"]
    event: str
    expected_date: str = ""
    linked_story_nodes: list[str] = []
    repricing_mechanism: str
    observable_metric: str
    surprise_threshold: str
    direction: Literal["positive", "negative", "two_sided"]
    conditions: list[ProbabilityCondition] = []
    base_probability: float = Field(default=0, ge=0, le=1)
    evidence_factor: float = Field(default=1, ge=0, le=1.5)
    probability: float = Field(default=0, ge=0, le=1)
    probability_assessment: ProbabilityAssessment = Field(default_factory=ProbabilityAssessment)
    evidence_refs: list[str] = []
    counter_evidence_refs: list[str] = []
    evidence_gate: EvidenceGateResult = Field(default_factory=EvidenceGateResult)
    verifies_assumption_ids: list[str] = Field(default_factory=list)
    affected_financial_variables: list[str] = Field(default_factory=list)
    favorable_result: str = ""
    adverse_result: str = ""
    scenario_probability_effects: dict[str, float | None] = Field(default_factory=dict)

    @model_validator(mode="after")
    def calculate_probability(self) -> "CatalystItem":
        if self.conditions:
            self.base_probability = prod(item.probability for item in self.conditions)
        elif self.base_probability == 0 and self.probability > 0:
            self.base_probability = self.probability
        self.probability = min(1.0, self.base_probability * self.evidence_factor)
        prior = self.probability_assessment
        self.probability_assessment = ProbabilityAssessment(
            base_probability=self.base_probability, evidence_factor=self.evidence_factor,
            adjusted_probability=self.probability,
            confidence=prior.confidence or min(1.0, self.evidence_factor),
            rationale=prior.rationale,
        )
        return self


class CatalystVerificationItem(BaseModel):
    catalyst_id: str = ""
    verifies_assumption_ids: list[str] = Field(default_factory=list)
    affected_financial_variables: list[str] = Field(default_factory=list)
    favorable_result: str = ""
    adverse_result: str = ""
    scenario_probability_effects: dict[str, float | None] = Field(default_factory=dict)
    observable: bool = False
    time_bound: bool = False
    probability_effects_defined: bool = False
    passed: bool = False
    status: Literal["verified", "unavailable", "blocked"] = "unavailable"
    issues: list[str] = Field(default_factory=list)


class CatalystVerificationAudit(BaseModel):
    passed: bool = False
    status: Literal["available", "unavailable", "blocked"] = "unavailable"
    items: list[CatalystVerificationItem] = Field(default_factory=list)
    unverified_catalyst_ids: list[str] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)


class CatalystOutput(BaseModel):
    catalysts: list[CatalystItem] = []
    repricing_path: list[str] = []
    repricing_probability: float = Field(default=0, ge=0, le=1)
    no_catalyst_risk: str = ""
    limitations: list[str] = []

    @model_validator(mode="after")
    def calculate_repricing_probability(self) -> "CatalystOutput":
        positive = [item.probability for item in self.catalysts if item.direction in {"positive", "two_sided"}]
        self.repricing_probability = max(positive, default=0)
        return self


class RiskItem(BaseModel):
    risk_id: str = Field(default_factory=lambda: f"R{uuid4().hex[:6]}")
    risk: str
    category: Literal[
        "demand", "customer", "technology", "financial", "cash_flow",
        "receivables", "inventory", "litigation", "competition",
        "strategy", "valuation", "governance", "other"
    ]
    depends_on: list[str] = []
    base_probability: float = Field(default=0, ge=0, le=1)
    evidence_factor: float = Field(default=1, ge=0, le=1.5)
    probability: float = Field(ge=0, le=1)
    probability_assessment: ProbabilityAssessment = Field(default_factory=ProbabilityAssessment)
    evidence_gate: EvidenceGateResult = Field(default_factory=EvidenceGateResult)
    impact: Literal["low", "medium", "high", "critical"]
    evidence_refs: list[str] = []
    verification: str

    @model_validator(mode="after")
    def calculate_probability(self) -> "RiskItem":
        if self.base_probability == 0 and self.probability > 0:
            self.base_probability = self.probability
        self.probability = min(1.0, self.base_probability * self.evidence_factor)
        prior = self.probability_assessment
        self.probability_assessment = ProbabilityAssessment(
            base_probability=self.base_probability, evidence_factor=self.evidence_factor,
            adjusted_probability=self.probability,
            confidence=prior.confidence or min(1.0, self.evidence_factor),
            rationale=prior.rationale,
        )
        return self


class CausalAttack(BaseModel):
    from_node: str
    to_node: str
    assumed_causality: str
    alternative_explanation: str
    break_condition: str
    severity: Literal["low", "medium", "high", "critical"]
    evidence_refs: list[str] = []


class BearOutput(BaseModel):
    risks: list[RiskItem] = []
    risk_tree: LogicTree = Field(default_factory=lambda: LogicTree(tree_type="risk"))
    downside_risk_probability: float = Field(default=0, ge=0, le=1)
    causal_attacks: list[CausalAttack] = []
    false_premises: list[str] = []
    strongest_disproof: str = "证据不足，尚未形成最强反证"
    thesis_breakers: list[str] = []
    falsification_tests: list[str] = []
    limitations: list[str] = []


class MispricingDriverDraft(BaseModel):
    driver_id: str = ""
    driver_type: Literal[
        "revenue", "gross_margin", "ebit_margin", "business_mix",
        "market_share", "valuation_multiple", "other",
    ] = "other"
    claim: str = ""
    mechanism: str = ""
    evidence_refs: list[str] = []
    counter_evidence_refs: list[str] = []
    catalyst_ids: list[str] = []
    verification_metric: str = ""
    time_window: str = ""
    rerating_mechanism: str = ""


class FalsifiableConditionDraft(BaseModel):
    condition_id: str = ""
    driver_id: str = ""
    current_judgment: str = ""
    key_variables: list[str] = []
    verification_metric: str = ""
    confirms_if: str = ""
    invalidates_if: str = ""
    latest_verification_time: str = ""
    current_status: Literal["untested", "supporting", "refuted"] = "untested"
    evidence_refs: list[str] = []
    counter_evidence_refs: list[str] = []


class MispricingNarrativeDraft(BaseModel):
    market_bet: str = ""
    agent_disagreement: str = ""
    why_market_may_be_wrong: str = ""
    drivers: list[MispricingDriverDraft] = []
    falsifiable_conditions: list[FalsifiableConditionDraft] = []
    limitations: list[str] = []


class ExpectationPoint(BaseModel):
    series_type: Literal["market_consensus", "model_implied", "agent_forecast"]
    scenario: Literal["bear", "base", "bull"] | None = None
    metric: Literal[
        "revenue", "net_profit", "gross_margin", "ebit_margin", "ebit",
        "ebitda", "net_margin", "valuation_multiple", "target_price",
    ]
    period: str
    value: float | None = None
    unit: str = ""
    available: bool = False
    evidence_refs: list[str] = []
    derivation: str = ""
    assumptions: list[str] = []


class VariableGapContribution(BaseModel):
    variable: str = ""
    period: str = ""
    scenario: Literal["bear", "base", "bull"] | None = None
    benchmark_value: float | None = None
    agent_value: float | None = None
    absolute_gap: float | None = None
    contribution: float | None = None
    contribution_percentage: float | None = None
    direction: Literal["positive", "negative", "neutral", "unavailable"] = "unavailable"
    available: bool = False
    formula: str = ""
    evidence_refs: list[str] = Field(default_factory=list)


class MispricingGap(BaseModel):
    gap_id: str
    scenario: Literal["bear", "base", "bull"]
    metric: str
    period: str
    benchmark_type: Literal["market_consensus", "model_implied"]
    benchmark_value: float | None = None
    agent_value: float | None = None
    absolute_gap: float | None = None
    gap_percentage: float | None = None
    direction: Literal["market_expectation_too_low", "market_expectation_too_high", "neutral", "unavailable"] = "unavailable"
    significant: bool = False
    formula: str = ""


class MispricingDriver(BaseModel):
    driver_id: str
    driver_type: str
    claim: str
    mechanism: str = ""
    evidence_refs: list[str] = []
    counter_evidence_refs: list[str] = []
    catalyst_ids: list[str] = []
    verification_metric: str = ""
    time_window: str = ""
    rerating_mechanism: str = ""
    evidence_gate: EvidenceGateResult = Field(default_factory=EvidenceGateResult)
    supported: bool = False
    issues: list[str] = []


class FalsifiableCondition(BaseModel):
    condition_id: str
    driver_id: str
    current_judgment: str
    key_variables: list[str] = []
    verification_metric: str
    confirms_if: str
    invalidates_if: str
    latest_verification_time: str
    current_status: Literal["untested", "supporting", "refuted"] = "untested"
    evidence_refs: list[str] = []
    counter_evidence_refs: list[str] = []
    evidence_gate: EvidenceGateResult = Field(default_factory=EvidenceGateResult)
    counter_evidence_gate: EvidenceGateResult = Field(default_factory=EvidenceGateResult)
    currently_invalidated: bool = False
    complete: bool = False
    issues: list[str] = []


class ExpectationGapChain(BaseModel):
    driver_id: str
    market_expectation: str = ""
    agent_expectation: str = ""
    gap_id: str = ""
    evidence_refs: list[str] = []
    catalyst_ids: list[str] = []
    verification_metric: str = ""
    time_window: str = ""
    repricing_mechanism: str = ""
    complete: bool = False
    issues: list[str] = []


class InvestmentEdgeComponent(BaseModel):
    name: Literal[
        "market_vs_agent_earnings_gap", "evidence_strength", "forecast_confidence",
        "catalyst_proximity", "valuation_gap", "downside_risk",
    ]
    available: bool = False
    score: float | None = Field(default=None, ge=0, le=100)
    formula: str = ""
    inputs: dict[str, Any] = {}
    issues: list[str] = []


class InvestmentEdgeScore(BaseModel):
    available: bool = False
    total_score: float | None = Field(default=None, ge=0, le=100)
    components: list[InvestmentEdgeComponent] = []
    formula: str = "六个结构化分项等权平均；任一关键Gate失败时总分为null"
    blockers: list[str] = []


class MispricingClassification(BaseModel):
    price_cheap: bool = False
    market_expectation_too_high: bool = False
    market_expectation_too_low: bool = False
    fundamentals_improving: bool = False
    valuation_rerating: bool = False
    basis: dict[str, str] = {}


class MispricingGate(BaseModel):
    passed: bool = False
    agent_output_available: bool = False
    authoritative_series_separated: bool = False
    consensus_gate_passed: bool = False
    benchmark_available: bool = False
    existing_gates_passed: bool = False
    significant_gap_found: bool = False
    driver_evidence_passed: bool = False
    catalyst_gate_passed: bool = False
    falsifier_defined: bool = False
    complete_chain: bool = False
    scenario_trigger_gate_passed: bool = False
    catalyst_verification_passed: bool = False
    investment_logic_chain_passed: bool = False
    double_counting_passed: bool = False
    unified_evidence_status_passed: bool = False
    issues: list[str] = []


class MispricingVerdict(BaseModel):
    status: Literal["NO EDGE", "WATCH", "EMERGING EDGE", "ACTIONABLE EDGE", "INVALIDATED"] = "NO EDGE"
    label: str = "NO EDGE"
    summary: str = "尚未形成可验证的预期差结论"
    investable: bool = False
    blocked: bool = True
    primary_benchmark: Literal["market_consensus", "model_implied", "unavailable"] = "unavailable"
    maximum_gap: str = ""
    repricing_event: str = ""
    verification_deadline: str = ""
    failure_point: str = ""
    blockers: list[str] = []


class InvestmentLogicStep(BaseModel):
    step_id: str = ""
    premise: str = ""
    conclusion: str = ""
    mechanism: str = ""
    depends_on: list[str] = Field(default_factory=list)
    assumption_ids: list[str] = Field(default_factory=list)
    affected_financial_variables: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    evidence_gate: EvidenceGateResult = Field(default_factory=EvidenceGateResult)
    passed: bool = False
    status: Literal["verified", "unavailable", "blocked"] = "unavailable"
    issues: list[str] = Field(default_factory=list)


class InvestmentLogicChainAudit(BaseModel):
    passed: bool = False
    status: Literal["available", "unavailable", "blocked"] = "blocked"
    steps: list[InvestmentLogicStep] = Field(default_factory=list)
    complete: bool = False
    causal_links_verified: bool = False
    failed_step_ids: list[str] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)


class TrackableInvestmentCondition(BaseModel):
    condition_id: str = ""
    description: str = ""
    metric: str = ""
    current_value: float | None = None
    target_value: float | None = None
    unit: str = ""
    deadline: str = ""
    confirms_if: str = ""
    invalidates_if: str = ""
    evidence_refs: list[str] = Field(default_factory=list)
    status: Literal["unverified", "supporting", "invalidated", "unavailable"] = "unavailable"


class FutureValidationItem(BaseModel):
    validation_id: str = ""
    condition_id: str = ""
    question: str = ""
    metric: str = ""
    data_source: str = ""
    expected_date: str = ""
    favorable_result: str = ""
    adverse_result: str = ""
    evidence_refs: list[str] = Field(default_factory=list)
    status: Literal["pending", "completed", "unavailable", "blocked"] = "unavailable"


class FutureValidationChecklist(BaseModel):
    status: Literal["available", "unavailable", "blocked"] = "unavailable"
    items: list[FutureValidationItem] = Field(default_factory=list)
    complete: bool = False
    next_validation_date: str = ""
    issues: list[str] = Field(default_factory=list)


class EvidenceStatusReference(BaseModel):
    evidence_id: str = ""
    fact: str = ""
    source: str = ""
    source_grade: str = ""
    evidence_date: str = ""
    current_and_valid: bool = False
    note: str = ""


class UnifiedEvidenceStatusItem(BaseModel):
    assumption_id: str = ""
    category: str = ""
    title: str = ""
    hypothesis: str = ""
    scenario: str = ""
    year: str = ""
    financial_variables: list[str] = Field(default_factory=list)
    critical: bool = True
    current_status: Literal["已验证", "部分验证", "无法验证", "已证伪", "互相矛盾"] = "无法验证"
    logic_status: Literal["逻辑完整", "逻辑不完整", "逻辑错误", "无法判断"] = "无法判断"
    one_sentence_conclusion: str = ""
    why: str = ""
    data: str = ""
    implication: str = ""
    support_evidence: list[EvidenceStatusReference] = Field(default_factory=list)
    counter_evidence: list[EvidenceStatusReference] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    next_verification: str = ""
    support_gate: EvidenceGateResult = Field(default_factory=EvidenceGateResult)
    counter_gate: EvidenceGateResult = Field(default_factory=EvidenceGateResult)
    issues: list[str] = Field(default_factory=list)


class EvidenceStatusLayer(BaseModel):
    items: list[UnifiedEvidenceStatusItem] = Field(default_factory=list)
    status_counts: dict[str, int] = Field(default_factory=dict)
    critical_item_count: int = 0
    critical_verified_count: int = 0
    critical_all_verified: bool = False
    has_refuted_assumption: bool = False
    has_conflicting_evidence: bool = False
    has_logic_error: bool = False
    actionable_edge_allowed: bool = False
    one_sentence_conclusion: str = "当前关键假设尚未完成统一证据审计。"
    why: str = ""
    data: str = ""
    implication: str = ""


class EvidenceAcquisitionPlanItem(BaseModel):
    rank: int = 0
    plan_id: str = ""
    assumption_id: str = ""
    assumption_title: str = ""
    current_status: str = ""
    evidence_to_find: str = ""
    preferred_sources: list[str] = Field(default_factory=list)
    required_quality: str = "当前有效A级一条，或来自两个独立来源的当前有效B级证据"
    why_high_value: str = ""
    expected_unblock: list[str] = Field(default_factory=list)
    suggested_search: str = ""
    priority_score: int = 0


class EvidenceAcquisitionPlan(BaseModel):
    items: list[EvidenceAcquisitionPlanItem] = Field(default_factory=list)
    one_sentence_conclusion: str = "暂无需要新增获取的关键证据。"
    why: str = ""
    data: str = ""
    implication: str = ""


class EvidenceBasedInvestmentVerdict(BaseModel):
    status: Literal[
        "已证伪：不成立", "证据不足：暂不能判断", "已验证但估值不划算",
        "已验证且存在预期差", "已验证且达到可投资条件",
    ] = "证据不足：暂不能判断"
    primary_reason_type: Literal[
        "有效反证", "证据不足", "证据矛盾", "投资逻辑错误",
        "估值不划算", "已验证预期差", "达到可投资条件",
    ] = "证据不足"
    one_sentence_conclusion: str = "证据不足，暂不能判断。"
    why: str = ""
    data: str = ""
    implication: str = ""
    target_price_allowed: bool = False
    expected_return_allowed: bool = False
    actionable_edge_allowed: bool = False
    key_assumption_ids: list[str] = Field(default_factory=list)
    next_evidence: list[str] = Field(default_factory=list)


class MispricingOutput(BaseModel):
    narrative_draft: MispricingNarrativeDraft = Field(default_factory=MispricingNarrativeDraft)
    market_expectation_summary: str = "Market Consensus unavailable"
    model_implied_summary: str = "Model-Implied Expectation unavailable"
    agent_forecast_summary: str = "Agent Forecast unavailable"
    expectation_points: list[ExpectationPoint] = []
    gaps: list[MispricingGap] = []
    maximum_gap: MispricingGap | None = None
    drivers: list[MispricingDriver] = []
    falsifiable_conditions: list[FalsifiableCondition] = []
    chains: list[ExpectationGapChain] = []
    variable_gap_contributions: list[VariableGapContribution] = Field(default_factory=list)
    largest_gap_variable: str = ""
    catalyst_verification_audit: CatalystVerificationAudit = Field(default_factory=CatalystVerificationAudit)
    investment_logic_chain: InvestmentLogicChainAudit = Field(default_factory=InvestmentLogicChainAudit)
    trackable_conditions: list[TrackableInvestmentCondition] = Field(default_factory=list)
    future_validation_checklist: FutureValidationChecklist = Field(default_factory=FutureValidationChecklist)
    evidence_status_layer: EvidenceStatusLayer = Field(default_factory=EvidenceStatusLayer)
    evidence_acquisition_plan: EvidenceAcquisitionPlan = Field(default_factory=EvidenceAcquisitionPlan)
    evidence_based_verdict: EvidenceBasedInvestmentVerdict = Field(default_factory=EvidenceBasedInvestmentVerdict)
    positive_expectation_gap_score: int = Field(default=1, ge=1, le=5)
    risk_score: int = Field(default=5, ge=1, le=5)
    edge: InvestmentEdgeScore = Field(default_factory=InvestmentEdgeScore)
    classification: MispricingClassification = Field(default_factory=MispricingClassification)
    gate: MispricingGate = Field(default_factory=MispricingGate)
    verdict: MispricingVerdict = Field(default_factory=MispricingVerdict)
    limitations: list[str] = []


class TrackingPlan(BaseModel):
    next_30_days: list[str] = []
    next_quarter: list[str] = []
    next_half_year: list[str] = []


class ProbabilityBasis(BaseModel):
    metric: str
    value: float = Field(ge=0, le=1)
    formula: str
    factors: list[str] = []
    evidence_refs: list[str] = []


class JudgmentAnswer(BaseModel):
    conclusion: str = "证据不足"
    mechanisms: list[str] = []
    most_likely_failure: str = ""
    falsifier: str = ""
    confidence: float = Field(default=0, ge=0, le=1)
    evidence_refs: list[str] = []
    evidence_gate: EvidenceGateResult = Field(default_factory=EvidenceGateResult)
    claim_status: Literal["supported", "hypothesis", "unknown"] = "unknown"


class JudgeOutput(BaseModel):
    current_fundamentals: str
    current_market_story: str
    market_implied_summary: list[str] = []
    agent_forecast_summary: list[str] = []
    expectation_gap_drivers: list[str] = []
    why_market_underestimates: JudgmentAnswer = Field(default_factory=JudgmentAnswer)
    agent_most_likely_wrong: JudgmentAnswer = Field(default_factory=JudgmentAnswer)
    priced_in: list[str] = []
    not_priced_in: list[str] = []
    potential_new_stories: list[str] = []
    necessary_conditions: list[str] = []
    story_chain: list[str] = []
    most_uncertain_link: str = ""
    financial_impact: str
    valuation_impact: str
    maximum_falsifier: str
    story_probability: float = Field(ge=0, le=1)
    story_chain_probability: float = Field(default=0, ge=0, le=1)
    bull_thesis_probability: float = Field(default=0, ge=0, le=1)
    fundamental_delivery_probability: float = Field(ge=0, le=1)
    valuation_expansion_probability: float = Field(ge=0, le=1)
    downside_risk_probability: float = Field(default=0, ge=0, le=1)
    probability_of_negative_return: float = Field(default=0, ge=0, le=1)
    probability_basis: list[ProbabilityBasis] = []
    investment_verdict: InvestmentVerdict = Field(default_factory=InvestmentVerdict)
    mispricing_verdict: MispricingVerdict = Field(default_factory=MispricingVerdict)
    evidence_based_verdict: EvidenceBasedInvestmentVerdict = Field(default_factory=EvidenceBasedInvestmentVerdict)
    positive_expectation_gap_score: int = Field(ge=1, le=5)
    risk_score: int = Field(ge=1, le=5)
    tracking_plan: TrackingPlan
    evidence_refs: list[str] = []
    evidence_gate: EvidenceGateResult = Field(default_factory=EvidenceGateResult)
    disclaimer: str = "仅供研究，不构成投资建议。"


class ReadableConclusion(BaseModel):
    question: str
    conclusion: str = "暂无可靠结论"
    why: str = "现有资料不足"
    data: str = "暂无可靠数据"
    implication: str = "暂时不能据此做出投资判断"


class ReadableBlockExplanation(BaseModel):
    blocked: bool = False
    headline: str = "各项条件满足，可以继续计算"
    reasons: list[str] = []
    why_stop: str = ""


class ReadableMetricExplanation(BaseModel):
    name: str
    explanation: str
    formula: str = ""
    inputs: dict[str, Any] = {}


class NaturalLanguagePresentation(BaseModel):
    language: Literal["简体中文"] = "简体中文"
    terminology_version: str = "investor-readable-v1"
    verdict_label: str = "暂无明确结论"
    verdict_explanation: str = ""
    quick_conclusions: list[ReadableConclusion] = []
    return_block: ReadableBlockExplanation = Field(default_factory=ReadableBlockExplanation)
    metric_explanations: list[ReadableMetricExplanation] = []
    causal_chains: list[list[str]] = []
    important_risks: list[str] = []
    next_observations: list[str] = []
    terminology: dict[str, str] = {}
    limitations: list[str] = []


class ResearchState(BaseModel):
    analysis_id: str
    request: ResearchRequest
    status: Literal["running", "completed", "failed"] = "running"
    research: ResearchOutput | None = None
    financial: FinancialOutput | None = None
    industry: IndustryOutput | None = None
    company: CompanyOutput | None = None
    story: StoryOutput | None = None
    expectation: ExpectationOutput | None = None
    valuation: ValuationOutput | None = None
    catalyst: CatalystOutput | None = None
    bear: BearOutput | None = None
    mispricing: MispricingOutput | None = None
    judge: JudgeOutput | None = None
    presentation: NaturalLanguagePresentation | None = None
    completed_at: datetime | None = None
    error: str = ""
