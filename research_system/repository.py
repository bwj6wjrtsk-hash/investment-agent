import json
import sqlite3
from pathlib import Path
from threading import Lock
from typing import Any

from .schemas import Evidence, ResearchRequest


class ResearchRepository:
    def __init__(self, path: str | Path | None = None) -> None:
        default = Path(__file__).resolve().parents[1] / "data" / "research.db"
        self.path = Path(path or default)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_db(self) -> None:
        with self._connect() as db:
            db.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS analyses (
                id TEXT PRIMARY KEY, company TEXT NOT NULL, symbol TEXT,
                analysis_date TEXT NOT NULL, status TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                completed_at TEXT, error TEXT
            );
            CREATE TABLE IF NOT EXISTS stage_outputs (
                analysis_id TEXT NOT NULL, stage TEXT NOT NULL,
                output_json TEXT NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (analysis_id, stage)
            );
            CREATE TABLE IF NOT EXISTS evidence (
                analysis_id TEXT NOT NULL, evidence_id TEXT NOT NULL,
                fact TEXT NOT NULL, source TEXT NOT NULL, source_url TEXT,
                source_type TEXT, publication_date TEXT, information_date TEXT,
                available_at TEXT, analysis_date TEXT,
                retrieved_at TEXT NOT NULL, confidence REAL NOT NULL, excerpt TEXT,
                source_grade TEXT DEFAULT 'C', freshness_days INTEGER,
                temporal_scope TEXT DEFAULT 'unknown', supports_current_claim INTEGER DEFAULT 0,
                invalid_reason TEXT DEFAULT '',
                PRIMARY KEY (analysis_id, evidence_id)
            );
            CREATE TABLE IF NOT EXISTS expectation_timeline (
                analysis_id TEXT NOT NULL, company TEXT NOT NULL,
                analysis_date TEXT NOT NULL, metric TEXT NOT NULL,
                period TEXT, market_implied REAL, agent_base REAL,
                gap_direction TEXT, PRIMARY KEY (analysis_id, metric, period)
            );
            CREATE TABLE IF NOT EXISTS expectation_series (
                analysis_id TEXT NOT NULL, metric TEXT NOT NULL, period TEXT NOT NULL,
                series_type TEXT NOT NULL CHECK (
                    series_type IN ('model_implied', 'market_consensus', 'agent_forecast', 'actual')
                ),
                value REAL, unit TEXT NOT NULL DEFAULT '',
                available INTEGER NOT NULL DEFAULT 0 CHECK (available IN (0, 1)),
                evidence_refs TEXT NOT NULL DEFAULT '[]',
                inference_method TEXT NOT NULL DEFAULT '',
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (analysis_id, metric, period, series_type)
            );
            """)
            existing = {row[1] for row in db.execute("PRAGMA table_info(evidence)").fetchall()}
            migrations = {
                "source_grade": "TEXT DEFAULT 'C'",
                "freshness_days": "INTEGER",
                "temporal_scope": "TEXT DEFAULT 'unknown'",
                "supports_current_claim": "INTEGER DEFAULT 0",
                "available_at": "TEXT",
                "analysis_date": "TEXT",
                "invalid_reason": "TEXT DEFAULT ''",
            }
            for column, definition in migrations.items():
                if column not in existing:
                    db.execute(f"ALTER TABLE evidence ADD COLUMN {column} {definition}")
            db.execute(
                """UPDATE evidence SET analysis_date=(
                    SELECT analyses.analysis_date FROM analyses WHERE analyses.id=evidence.analysis_id
                ) WHERE analysis_date IS NULL OR analysis_date=''"""
            )
            db.execute(
                """INSERT OR IGNORE INTO expectation_series(
                    analysis_id, metric, period, series_type, value, unit,
                    available, evidence_refs, inference_method
                ) SELECT analysis_id, metric, COALESCE(period, ''), 'model_implied',
                    market_implied, '', 1, '[]', 'legacy expectation_timeline migration'
                  FROM expectation_timeline WHERE market_implied IS NOT NULL"""
            )
            db.execute(
                """INSERT OR IGNORE INTO expectation_series(
                    analysis_id, metric, period, series_type, value, unit,
                    available, evidence_refs, inference_method
                ) SELECT analysis_id, metric, COALESCE(period, ''), 'agent_forecast',
                    agent_base, '', 1, '[]', 'legacy expectation_timeline migration'
                  FROM expectation_timeline WHERE agent_base IS NOT NULL"""
            )

    def create_analysis(self, analysis_id: str, request: ResearchRequest) -> None:
        with self._lock, self._connect() as db:
            db.execute(
                "INSERT INTO analyses(id, company, symbol, analysis_date, status) VALUES(?,?,?,?,?)",
                (analysis_id, request.company, request.symbol, request.analysis_date.isoformat(), "running"),
            )

    def save_stage(self, analysis_id: str, stage: str, output: Any) -> None:
        data = output.model_dump(mode="json") if hasattr(output, "model_dump") else output
        with self._lock, self._connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO stage_outputs(analysis_id, stage, output_json) VALUES(?,?,?)",
                (analysis_id, stage, json.dumps(data, ensure_ascii=False)),
            )
            if stage == "research":
                for item in getattr(output, "evidence", []):
                    self._save_evidence(db, analysis_id, item)
            if stage == "expectation":
                self._save_expectations(db, analysis_id, output)
            if stage == "valuation":
                self._save_agent_forecasts(db, analysis_id, output)

    @staticmethod
    def _save_evidence(db: sqlite3.Connection, analysis_id: str, item: Evidence) -> None:
        db.execute(
            """INSERT OR REPLACE INTO evidence(
                analysis_id, evidence_id, fact, source, source_url, source_type,
                publication_date, information_date, available_at, analysis_date,
                retrieved_at, confidence, excerpt, source_grade, freshness_days,
                temporal_scope, supports_current_claim, invalid_reason
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                analysis_id, item.id, item.fact, item.source, item.source_url,
                item.source_type,
                item.publication_date.isoformat() if item.publication_date else None,
                item.information_date.isoformat() if item.information_date else None,
                item.available_at.isoformat() if item.available_at else None,
                item.analysis_date.isoformat(), item.retrieved_at.isoformat(),
                item.confidence, item.excerpt, item.source_grade, item.freshness_days,
                item.temporal_scope, int(item.supports_current_claim), item.invalid_reason,
            ),
        )

    @staticmethod
    def _upsert_series(
        db: sqlite3.Connection,
        analysis_id: str,
        metric: str,
        period: str,
        series_type: str,
        value: float,
        unit: str = "",
        evidence_refs: list[str] | None = None,
        inference_method: str = "",
    ) -> None:
        db.execute(
            """INSERT INTO expectation_series(
                analysis_id, metric, period, series_type, value, unit,
                available, evidence_refs, inference_method, updated_at
            ) VALUES(?,?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)
            ON CONFLICT(analysis_id, metric, period, series_type) DO UPDATE SET
                value=excluded.value, unit=excluded.unit, available=excluded.available,
                evidence_refs=excluded.evidence_refs,
                inference_method=excluded.inference_method, updated_at=CURRENT_TIMESTAMP""",
            (
                analysis_id, metric, period or "", series_type, value, unit or "", 1,
                json.dumps(evidence_refs or [], ensure_ascii=False), inference_method or "",
            ),
        )

    @classmethod
    def _save_expectation_components(
        cls, db: sqlite3.Connection, analysis_id: str, output: Any
    ) -> None:
        db.execute(
            """DELETE FROM expectation_series WHERE analysis_id=?
               AND series_type IN ('model_implied', 'market_consensus', 'actual')""",
            (analysis_id,),
        )
        saved: set[tuple[str, str, str]] = set()
        for item in getattr(output, "expectation_gaps", []):
            common = (item.metric, item.period or "", item.unit, item.evidence_refs, item.inference_method)
            candidates = (
                ("model_implied", item.model_implied_value, item.model_implied_value is not None),
                (
                    "market_consensus", item.market_consensus_value,
                    item.market_consensus_available and item.market_consensus_value is not None,
                ),
                ("actual", item.actual_value, item.actual_value is not None),
            )
            for series_type, value, available in candidates:
                if available:
                    cls._upsert_series(
                        db, analysis_id, common[0], common[1], series_type, value,
                        common[2], common[3], common[4],
                    )
                    saved.add((common[0], common[1], series_type))

        path_fields = (
            ("net_profit", "implied_net_profit", ""),
            ("revenue", "implied_revenue", ""),
            ("net_profit_growth", "implied_net_profit_growth", "ratio"),
            ("revenue_growth", "implied_revenue_growth", "ratio"),
        )
        path = getattr(output, "price_implied_path", None) or getattr(output, "market_implied_path", [])
        for item in path:
            for metric, field, unit in path_fields:
                value = getattr(item, field, None)
                key = (metric, item.year or "", "model_implied")
                if value is not None and key not in saved:
                    cls._upsert_series(
                        db, analysis_id, metric, item.year, "model_implied", value,
                        unit, item.evidence_refs, item.formula,
                    )
                    saved.add(key)

        for item in getattr(output, "consensus_estimates", []):
            if not item.available:
                continue
            for metric in ("revenue", "net_profit"):
                value = getattr(item, metric, None)
                key = (metric, item.year or "", "market_consensus")
                if value is not None and key not in saved:
                    cls._upsert_series(
                        db, analysis_id, metric, item.year, "market_consensus", value,
                        "", item.evidence_refs,
                        f"reliable external consensus ({item.source_count} independent sources)",
                    )
                    saved.add(key)

    @classmethod
    def _save_expectations(cls, db: sqlite3.Connection, analysis_id: str, output: Any) -> None:
        row = db.execute(
            "SELECT company, analysis_date FROM analyses WHERE id=?", (analysis_id,)
        ).fetchone()
        if not row:
            return
        for item in output.expectation_gaps:
            db.execute(
                """INSERT OR REPLACE INTO expectation_timeline
                VALUES(?,?,?,?,?,?,?,?)""",
                (
                    analysis_id, row["company"], row["analysis_date"], item.metric,
                    item.period, item.market_implied, item.agent_base, item.gap_direction,
                ),
            )
        cls._save_expectation_components(db, analysis_id, output)

    @classmethod
    def _save_agent_forecasts(cls, db: sqlite3.Connection, analysis_id: str, output: Any) -> None:
        db.execute(
            "DELETE FROM expectation_series WHERE analysis_id=? AND series_type='agent_forecast'",
            (analysis_id,),
        )
        unit = getattr(output, "unit", "")
        base = next(
            (scenario for scenario in getattr(output, "scenarios", []) if scenario.name == "base"),
            None,
        )
        if not base:
            return
        base_refs = list(getattr(base, "evidence_refs", []) or [])
        saved: set[tuple[str, str]] = set()
        for projection in getattr(base, "projections", []):
            period = str(projection.year)
            refs = list(dict.fromkeys(base_refs + list(getattr(projection, "evidence_refs", []) or [])))
            ebit_margin = (
                projection.ebit / projection.revenue
                if projection.revenue not in {None, 0} else None
            )
            net_margin = (
                projection.net_profit / projection.revenue
                if projection.revenue not in {None, 0} else None
            )
            metrics = (
                ("revenue", projection.revenue, unit),
                ("net_profit", projection.net_profit, unit),
                ("ebit", projection.ebit, unit),
                ("ebitda", projection.ebitda, unit),
                ("revenue_growth", projection.revenue_growth, "ratio"),
                ("gross_margin", projection.gross_margin, "ratio"),
                ("ebit_margin", ebit_margin, "ratio"),
                ("net_margin", net_margin, "ratio"),
                ("valuation_multiple", projection.valuation_multiple or None, "x"),
            )
            for metric, value, metric_unit in metrics:
                if value is None:
                    continue
                cls._upsert_series(
                    db, analysis_id, metric, period, "agent_forecast", value,
                    metric_unit, refs,
                    "Base Agent Forecast经程序化财务桥计算；与Market Consensus和Model-Implied严格分离",
                )
                saved.add((metric, period))
        for item in getattr(output, "forecast_comparison", []):
            forecast = (
                item.agent_forecast_profit
                if item.agent_forecast_profit is not None
                else item.agent_base_profit
            )
            key = ("net_profit", str(item.year))
            if forecast is not None and key not in saved:
                cls._upsert_series(
                    db, analysis_id, "net_profit", item.year, "agent_forecast",
                    forecast, unit, base_refs, item.explanation,
                )

    def save_expectation_series(
        self, analysis_id: str, expectation: Any, valuation: Any | None = None
    ) -> None:
        """Persist the authoritative combined series after dependent stages are available."""
        with self._lock, self._connect() as db:
            self._save_expectation_components(db, analysis_id, expectation)
            if valuation is not None:
                self._save_agent_forecasts(db, analysis_id, valuation)

    def complete_analysis(self, analysis_id: str) -> None:
        with self._lock, self._connect() as db:
            db.execute(
                "UPDATE analyses SET status='completed', completed_at=CURRENT_TIMESTAMP WHERE id=?",
                (analysis_id,),
            )

    def fail_analysis(self, analysis_id: str, error: str) -> None:
        with self._lock, self._connect() as db:
            db.execute(
                "UPDATE analyses SET status='failed', error=? WHERE id=?",
                (error[:4000], analysis_id),
            )

    def list_analyses(self, limit: int = 20) -> list[dict]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM analyses ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(row) for row in rows]

    def get_analysis(self, analysis_id: str) -> dict | None:
        with self._connect() as db:
            analysis = db.execute(
                "SELECT * FROM analyses WHERE id=?", (analysis_id,)
            ).fetchone()
            if not analysis:
                return None
            stages = db.execute(
                "SELECT stage, output_json FROM stage_outputs WHERE analysis_id=?",
                (analysis_id,),
            ).fetchall()
            evidence = db.execute(
                "SELECT * FROM evidence WHERE analysis_id=? ORDER BY confidence DESC",
                (analysis_id,),
            ).fetchall()
            series = db.execute(
                """SELECT metric, period, series_type, value, unit, available,
                   evidence_refs, inference_method, updated_at
                   FROM expectation_series WHERE analysis_id=?
                   ORDER BY metric, period, series_type""",
                (analysis_id,),
            ).fetchall()
        result = dict(analysis)
        stage_map = {row["stage"]: json.loads(row["output_json"]) for row in stages}
        # Historical analyses are deterministically re-audited on read so stale,
        # pre-audit probability/Expected Return fields are never presented as valid.
        try:
            from .calculations import (
                build_gap_audits,
                calculate_extended_valuation_audits,
                calculate_sensitivity_audit,
                calculate_unified_valuation_engine,
                finalize_risk_reward_and_decision,
                finalize_valuation_probabilities,
                revalidate_pre_valuation_probability_audit,
            )
            from .schemas import FinancialOutput, ValuationOutput
            if {"research", "financial", "valuation"}.issubset(stage_map):
                research_data = stage_map["research"]
                financial_model = FinancialOutput.model_validate(stage_map["financial"])
                valuation_model = ValuationOutput.model_validate(stage_map["valuation"])
                evidence_index = {
                    row["evidence_id"]: {**dict(row), "id": row["evidence_id"], "supports_current_claim": bool(row["supports_current_claim"])}
                    for row in evidence
                }
                valuation_model = revalidate_pre_valuation_probability_audit(
                    valuation_model, evidence_index
                )
                valuation_model = calculate_unified_valuation_engine(
                    valuation_model, research_data, financial_model, evidence_index
                )
                valuation_model = calculate_extended_valuation_audits(
                    valuation_model, evidence_index, financial_model, research_data
                )
                valuation_model = finalize_valuation_probabilities(valuation_model)
                valuation_model = build_gap_audits(valuation_model)
                valuation_model = calculate_sensitivity_audit(valuation_model)
                valuation_model = finalize_risk_reward_and_decision(
                    valuation_model, financial_model, research_data
                )
                stage_map["valuation"] = valuation_model.model_dump(mode="json")

                from .mispricing import replay_mispricing_output
                mispricing_model = replay_mispricing_output(
                    stage_map.get("mispricing"),
                    stage_map.get("expectation") or {},
                    valuation_model,
                    stage_map.get("catalyst") or {},
                    stage_map.get("bear") or {},
                    research_data,
                    financial_model,
                    industry=stage_map.get("industry"),
                    company=stage_map.get("company"),
                )
                mispricing_data = mispricing_model.model_dump(mode="json")
                stage_map["mispricing"] = mispricing_data

                if "judge" in stage_map:
                    stored_judge = dict(stage_map["judge"])
                    judge_data = dict(stored_judge)
                    decision = valuation_model.decision_probabilities
                    basis_by_metric = {
                        item.metric: item.value
                        for item in valuation_model.deterministic_probability_basis
                    }

                    def copied_probability(value):
                        # Judge schema requires concrete display values. Zero is only
                        # a placeholder when the deterministic decision is blocked.
                        return float(value) if value is not None else 0.0

                    judge_data["bull_thesis_probability"] = copied_probability(
                        decision.probability_of_thesis_success
                    )
                    judge_data["fundamental_delivery_probability"] = copied_probability(
                        basis_by_metric.get("基本面改善")
                    )
                    judge_data["valuation_expansion_probability"] = copied_probability(
                        basis_by_metric.get("估值重估")
                    )
                    judge_data["downside_risk_probability"] = copied_probability(
                        basis_by_metric.get("Downside Event")
                    )
                    judge_data["probability_of_negative_return"] = copied_probability(
                        decision.probability_of_negative_return
                    )
                    judge_data["probability_basis"] = [
                        item.model_dump(mode="json")
                        for item in valuation_model.deterministic_probability_basis
                    ]
                    judge_data["investment_verdict"] = (
                        valuation_model.deterministic_investment_verdict.model_dump(mode="json")
                    )
                    judge_data["mispricing_verdict"] = mispricing_data["verdict"]
                    judge_data["evidence_based_verdict"] = mispricing_data[
                        "evidence_based_verdict"
                    ]
                    judge_data["positive_expectation_gap_score"] = mispricing_data[
                        "positive_expectation_gap_score"
                    ]
                    judge_data["risk_score"] = mispricing_data["risk_score"]
                    if "tracking_plan" in mispricing_data:
                        judge_data["tracking_plan"] = mispricing_data["tracking_plan"]

                    story_data = stage_map.get("story")
                    story_probability = (
                        story_data.get("overall_probability")
                        if isinstance(story_data, dict) else None
                    )
                    for field in ("story_probability", "story_chain_probability"):
                        if field in stored_judge:
                            judge_data[field] = stored_judge[field]
                        elif story_probability is not None:
                            judge_data[field] = story_probability
                    stage_map["judge"] = judge_data
        except (KeyError, TypeError, ValueError):
            pass
        # 历史读取时重新生成中文可读性层，确保它使用上面刚完成重审计的结果；
        # 该层只改变表达，不回写或修改任何底层计算字段。
        try:
            from .presentation import build_natural_language_presentation
            stage_map["presentation"] = build_natural_language_presentation(
                {**stage_map, "analysis_id": analysis_id}
            ).model_dump(mode="json")
        except (KeyError, TypeError, ValueError):
            pass
        result["stages"] = stage_map
        result["evidence"] = [dict(row) for row in evidence]
        result["expectation_series"] = [self._decode_series_row(row) for row in series]
        return result

    @staticmethod
    def _decode_series_row(row: sqlite3.Row) -> dict:
        item = dict(row)
        item["available"] = bool(item["available"])
        try:
            item["evidence_refs"] = json.loads(item["evidence_refs"] or "[]")
        except (TypeError, json.JSONDecodeError):
            item["evidence_refs"] = []
        return item

    def get_expectation_timeline(self, company: str) -> list[dict]:
        with self._connect() as db:
            rows = db.execute(
                """SELECT a.analysis_date, s.analysis_id, s.metric, s.period,
                    s.series_type, s.value, s.unit, s.available, s.evidence_refs,
                    s.inference_method, s.updated_at,
                    CASE WHEN s.series_type='model_implied' THEN s.value ELSE l.market_implied END
                        AS market_implied,
                    CASE WHEN s.series_type='agent_forecast' THEN s.value ELSE l.agent_base END
                        AS agent_base,
                    l.gap_direction
                FROM expectation_series s
                JOIN analyses a ON a.id=s.analysis_id
                LEFT JOIN expectation_timeline l
                    ON l.analysis_id=s.analysis_id AND l.metric=s.metric
                    AND COALESCE(l.period, '')=s.period
                WHERE a.company=?
                ORDER BY a.analysis_date, s.metric, s.period, s.series_type""",
                (company,),
            ).fetchall()
            if rows:
                return [self._decode_series_row(row) for row in rows]
            legacy = db.execute(
                """SELECT analysis_date, metric, period, market_implied,
                agent_base, gap_direction FROM expectation_timeline
                WHERE company=? ORDER BY analysis_date, metric, period""",
                (company,),
            ).fetchall()
        return [dict(row) for row in legacy]
