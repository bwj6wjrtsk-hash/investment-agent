from __future__ import annotations

import json
import os
import re
from datetime import date, datetime
from typing import Any
from urllib.parse import urlparse

import requests

from knowledge.rag import search_knowledge
from tools.market_data import get_stock_fundamental, get_stock_history, get_stock_realtime
from tools.news_fetcher import get_stock_news

from .calculations import validate_market_snapshot
from .schemas import ComparableValuationEvidenceItem, Evidence, MarketSnapshot, ResearchRequest


class SourceCollector:
    def resolve_symbol(self, request: ResearchRequest) -> str:
        if request.symbol:
            return request.symbol.strip()
        value = request.company.strip()
        if value.isdigit() and len(value) == 6:
            return value
        try:
            import akshare as ak
            frame = ak.stock_info_a_code_name()
            exact = frame[frame["name"] == value]
            if exact.empty:
                exact = frame[frame["name"].str.contains(value, regex=False)]
            return str(exact.iloc[0]["code"]) if not exact.empty else ""
        except Exception:
            return ""

    @staticmethod
    def _invoke(tool: Any, **kwargs: Any) -> str:
        try:
            return str(tool.invoke(kwargs))
        except Exception as exc:
            return f"采集失败: {exc}"

    @staticmethod
    def _parse_date(value: Any) -> date | None:
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        text = str(value or "").strip()
        quarter_patterns = (
            (r"(20\d{2})\s*年?\s*第?\s*一季度", (3, 31)),
            (r"(20\d{2})\s*年?\s*(?:半年度|中期)", (6, 30)),
            (r"(20\d{2})\s*年?\s*第?\s*三季度", (9, 30)),
            (r"(20\d{2})\s*年?\s*(?:年度报告|年报)", (12, 31)),
        )
        for pattern, (month, day_value) in quarter_patterns:
            match = re.search(pattern, text)
            if match:
                return date(int(match.group(1)), month, day_value)
        for pattern in (r"(20\d{2})[-/.年](\d{1,2})[-/.月](\d{1,2})", r"(20\d{2})[-/.年](\d{1,2})"):
            match = re.search(pattern, text)
            if match:
                try:
                    parts = [int(item) for item in match.groups()]
                    return date(parts[0], parts[1], parts[2] if len(parts) > 2 else 1)
                except ValueError:
                    pass
        year = re.search(r"\b(20\d{2})\b", text)
        return date(int(year.group(1)), 1, 1) if year else None

    @staticmethod
    def _source_grade(url: str, source_type: str) -> str:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        path = parsed.path.lower()
        if any(item in domain for item in ("cninfo.com.cn", "sse.com.cn", "szse.cn", "hkexnews.hk")):
            return "A"
        if any(item in path for item in ("investorrelations", "investor-relations", "quarter_report", "annual_report")) and not any(item in domain for item in ("10jqka", "eastmoney", "sina")):
            return "A"
        if source_type in {"market_data", "financial_data", "company_data"}:
            return "B"
        if any(item in domain for item in ("10jqka.com.cn", "eastmoney.com", "sina.com.cn", "qq.com")):
            return "B"
        return "C"

    @classmethod
    def _evidence(
        cls, fact: str, source: str, source_url: str, source_type: str,
        confidence: float, excerpt: str = "", publication_date: date | None = None,
        information_date: date | None = None, source_grade: str | None = None,
        analysis_date: date | None = None, available_at: date | None = None,
    ) -> Evidence:
        return Evidence(
            fact=fact,
            excerpt=excerpt[:8000],
            source=source,
            source_url=source_url,
            source_type=source_type,
            source_grade=source_grade or cls._source_grade(source_url, source_type),
            publication_date=publication_date,
            information_date=information_date,
            available_at=available_at or publication_date or date.today(),
            analysis_date=analysis_date or date.today(),
            retrieved_at=datetime.now(),
            confidence=confidence,
        )

    @staticmethod
    def _number(text: str, label: str) -> float | None:
        match = re.search(rf"{re.escape(label)}\s*[:：]\s*([-+]?\d+(?:\.\d+)?)", text)
        try:
            return float(match.group(1)) if match else None
        except ValueError:
            return None

    def _market_snapshot(
        self, realtime: str, fundamental: str, history: str, evidence_refs: list[str]
    ) -> MarketSnapshot:
        price = self._number(fundamental, "最新价") or self._number(realtime, "最新价")
        market_cap = self._number(fundamental, "总市值")
        shares = market_cap / price if market_cap is not None and price not in {None, 0} else None
        return MarketSnapshot(
            as_of=date.today(),
            price=price,
            previous_close=self._number(realtime, "昨收"),
            day_open=self._number(realtime, "今开"),
            day_high=self._number(realtime, "最高"),
            day_low=self._number(realtime, "最低"),
            high_52w=self._number(history, "期间最高价"),
            low_52w=self._number(history, "期间最低价"),
            shares_outstanding=shares,
            shares_source="总市值÷当前价格（待独立财务数据校验）" if shares else "",
            market_cap=market_cap,
            pe_ttm=self._number(fundamental, "市盈率(PE)") or self._number(realtime, "市盈率"),
            pb=self._number(fundamental, "市净率(PB)"),
            evidence_refs=evidence_refs,
        )

    @staticmethod
    def _amount(value: Any, to_yi: bool = False) -> float | None:
        text = str(value or "").replace(",", "").strip()
        match = re.search(r"([-+]?\d+(?:\.\d+)?)", text)
        if not match:
            return None
        number = float(match.group(1))
        if not to_yi:
            return number
        if "万" in text and "亿" not in text:
            return number / 10000
        if "亿" in text:
            return number
        if "元" in text:
            return number / 100000000
        return None

    def _enrich_market_snapshot(
        self, snapshot: MarketSnapshot, financial_evidence: list[Evidence], analysis_date: date
    ) -> MarketSnapshot:
        item = next((entry for entry in financial_evidence if entry.source_type == "financial_data"), None)
        if item:
            try:
                records = json.loads(item.excerpt)
                dated = [(self._parse_date(row.get("报告期")), row) for row in records]
                dated = [(day, row) for day, row in dated if day and day <= analysis_date]
                if dated:
                    latest_day, latest = max(dated, key=lambda pair: pair[0])
                    latest_eps = self._amount(latest.get("基本每股收益"))
                    latest_profit = self._amount(latest.get("净利润"), to_yi=True)
                    snapshot.book_value_per_share = self._amount(latest.get("每股净资产"))
                    if latest_eps not in {None, 0} and latest_profit is not None:
                        snapshot.shares_outstanding = latest_profit / latest_eps
                        snapshot.shares_source = f"{latest_day.isoformat()}净利润÷基本EPS（独立财务口径）"
                    annual = next((row for day, row in dated if day.year == latest_day.year - 1 and day.month == 12), None)
                    prior = next((row for day, row in dated if day.year == latest_day.year - 1 and day.month == latest_day.month), None)
                    if annual and prior and latest_eps is not None:
                        annual_eps = self._amount(annual.get("基本每股收益"))
                        prior_eps = self._amount(prior.get("基本每股收益"))
                        if annual_eps is not None and prior_eps is not None:
                            snapshot.eps_ttm = annual_eps + latest_eps - prior_eps
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
        return validate_market_snapshot(snapshot, analysis_date)

    def _akshare_materials(self, symbol: str, analysis_date: date) -> list[Evidence]:
        if not symbol:
            return []
        evidence: list[Evidence] = []
        try:
            import akshare as ak
            financial = ak.stock_financial_abstract_ths(symbol=symbol, indicator="按报告期")
            raw_records = financial.tail(24).where(financial.notna(), None).to_dict("records")
            dated_records = [
                (day, row) for row in raw_records
                if (day := self._parse_date(row.get("报告期"))) and day <= analysis_date
            ]
            records = [row for _, row in dated_records[-16:]]
            latest_date = max((day for day, _ in dated_records), default=None)
            evidence.append(self._evidence(
                "公司历史财务指标（按报告期）", "同花顺/AKShare",
                f"https://basic.10jqka.com.cn/new/{symbol}/finance.html",
                "financial_data", 0.85, json.dumps(records, ensure_ascii=False, default=str),
                information_date=latest_date, available_at=latest_date, source_grade="B",
                analysis_date=analysis_date,
            ))
        except Exception as exc:
            evidence.append(self._evidence(
                "历史财务数据采集失败", "AKShare", "", "collection_error", 0.2, str(exc),
                analysis_date=analysis_date,
            ))

        try:
            import akshare as ak

            exchange_symbol = ("SH" if symbol.startswith(("5", "6", "9")) else "SZ") + symbol
            profit = ak.stock_profit_sheet_by_yearly_em(symbol=exchange_symbol)
            cashflow = ak.stock_cash_flow_sheet_by_yearly_em(symbol=exchange_symbol)

            def amount(value: Any) -> float | None:
                try:
                    number = float(value)
                    return None if number != number else number / 100000000
                except (TypeError, ValueError):
                    return None

            cash_by_period = {
                str(row.get("REPORT_DATE"))[:10]: row
                for row in cashflow.where(cashflow.notna(), None).to_dict("records")
            }
            statement_records: list[dict[str, Any]] = []
            for row in profit.where(profit.notna(), None).to_dict("records"):
                period = str(row.get("REPORT_DATE"))[:10]
                period_date = self._parse_date(period)
                if not period_date or period_date > analysis_date:
                    continue
                cash = cash_by_period.get(period, {})
                revenue = amount(row.get("TOTAL_OPERATE_INCOME"))
                operating_cost = amount(row.get("OPERATE_COST"))
                ebit = amount(row.get("OPERATE_PROFIT"))
                net_profit = amount(row.get("PARENT_NETPROFIT"))
                da_parts = [
                    amount(cash.get("FA_IR_DEPR")), amount(cash.get("IA_AMORTIZE")),
                    amount(cash.get("LPE_AMORTIZE")), amount(cash.get("USERIGHT_ASSET_AMORTIZE")),
                ]
                depreciation = sum(value for value in da_parts if value is not None) if any(
                    value is not None for value in da_parts
                ) else None
                statement_records.append({
                    "period": period,
                    "revenue": revenue,
                    "gross_profit": (
                        revenue - operating_cost
                        if revenue is not None and operating_cost is not None else None
                    ),
                    "operating_profit": ebit,
                    "ebit": ebit,
                    "ebitda": ebit + depreciation if ebit is not None and depreciation is not None else None,
                    "depreciation_amortization": depreciation,
                    "net_profit": net_profit,
                    "ebit_basis": "东方财富营业利润作为历史EBIT proxy；与严格EBIT口径存在差异，审计中明确披露",
                })
            statement_records.sort(key=lambda row: row["period"])
            statement_records = statement_records[-5:]
            latest_statement_date = self._parse_date(statement_records[-1]["period"]) if statement_records else None
            evidence.append(self._evidence(
                "年度利润表与现金流量表历史锚点（含EBIT proxy及D&A）", "东方财富/AKShare",
                f"https://data.eastmoney.com/bbsj/{symbol}.html",
                "financial_statement_data", 0.90,
                json.dumps(statement_records, ensure_ascii=False, default=str),
                information_date=latest_statement_date, available_at=latest_statement_date,
                source_grade="B", analysis_date=analysis_date,
            ))
        except Exception as exc:
            evidence.append(self._evidence(
                "年度利润表/现金流量表历史锚点采集失败", "AKShare", "",
                "collection_error", 0.2, str(exc), analysis_date=analysis_date,
            ))
        try:
            import akshare as ak
            business = ak.stock_zyjs_ths(symbol=symbol)
            records = business.head(30).where(business.notna(), None).to_dict("records")
            evidence.append(self._evidence(
                "公司主营业务与构成", "同花顺/AKShare",
                f"https://basic.10jqka.com.cn/new/{symbol}/operate.html",
                "company_data", 0.8, json.dumps(records, ensure_ascii=False, default=str),
                source_grade="B",
            ))
        except Exception as exc:
            evidence.append(self._evidence(
                "主营业务数据采集失败", "AKShare", "", "collection_error", 0.2, str(exc)
            ))

        try:
            import akshare as ak

            indicators = ak.stock_a_indicator_lg(symbol=symbol)
            records = indicators.where(indicators.notna(), None).to_dict("records")
            dated: list[tuple[date, dict[str, Any]]] = []
            for row in records:
                trade_date = self._parse_date(
                    row.get("trade_date") or row.get("date") or row.get("日期")
                )
                if trade_date and trade_date <= analysis_date:
                    dated.append((trade_date, row))
            dated.sort(key=lambda pair: pair[0])
            cutoff_year = analysis_date.year - 5
            recent = [(day, row) for day, row in dated if day.year >= cutoff_year]
            for method, columns, basis in (
                ("pe", ("pe_ttm", "市盈率(TTM)", "市盈率"), "TTM earnings；公司历史日度估值区间"),
                ("pb", ("pb", "市净率"), "reported book value；公司历史日度估值区间"),
            ):
                values: list[float] = []
                for _, row in recent:
                    raw = next((row.get(column) for column in columns if row.get(column) is not None), None)
                    try:
                        number = float(raw)
                        if number > 0 and number == number:
                            values.append(number)
                    except (TypeError, ValueError):
                        pass
                if not values:
                    continue
                ordered = sorted(values)
                low = ordered[int((len(ordered) - 1) * 0.10)]
                high = ordered[int((len(ordered) - 1) * 0.90)]
                middle = ordered[len(ordered) // 2]
                latest = recent[-1][0] if recent else analysis_date
                metadata = {
                    "subject": symbol,
                    "symbol": symbol,
                    "relationship": "company_historical",
                    "metric": method,
                    "value": middle,
                    "range_low": low,
                    "range_high": high,
                    "as_of": latest.isoformat(),
                    "period": f"{cutoff_year}-{analysis_date.year} historical daily range",
                    "applicability": "用于检验公司终值倍数是否落在自身近5年10%-90%历史区间；不替代同行可比分析",
                    "accounting_basis": basis,
                }
                evidence.append(self._evidence(
                    f"公司近5年历史{method.upper()}估值区间", "乐咕乐股/AKShare",
                    f"https://legulegu.com/stocklist/{symbol}", "comparable_valuation", 0.82,
                    json.dumps(metadata, ensure_ascii=False), information_date=latest,
                    available_at=latest, source_grade="B", analysis_date=analysis_date,
                ))
        except Exception as exc:
            evidence.append(self._evidence(
                "公司历史估值区间采集失败", "AKShare", "", "collection_error", 0.2,
                str(exc), analysis_date=analysis_date,
            ))
        return evidence

    def _tavily_materials(self, company: str, deep: bool) -> list[Evidence]:
        api_key = os.getenv("TAVILY_API_KEY", "")
        if not api_key:
            return []
        query_specs: list[tuple[str, str | None]] = [
            (f"{company} 最新公告 年报 季报 业绩 市场一致预期", None),
            (f"{company} 同行业 可比公司 EV/EBITDA 倍数 TTM 估值", "ev_ebitda"),
            (f"{company} 同行业 可比公司 EV/Sales 倍数 TTM 估值", "ev_sales"),
            (f"{company} 同行业 可比公司 PB 市净率 估值", "pb"),
        ]
        if deep:
            query_specs += [
                (f"{company} 行业 产业链 竞争对手 技术趋势 最新", None),
                (f"{company} 客户 订单 产能 新产品 最新", None),
                (f"{company} 风险 质疑 现金流 毛利率 最新", None),
                (f"{company} 历史估值区间 EV EBITDA EV Sales PB", None),
            ]
        evidence: list[Evidence] = []
        aliases = {
            "ev_ebitda": r"(?:EV\s*/\s*EBITDA|企业价值\s*/?\s*息税折旧摊销前利润)",
            "ev_sales": r"(?:EV\s*/\s*(?:Sales|Revenue)|企业价值\s*/?\s*(?:销售额|收入))",
            "pb": r"(?:P\s*/\s*B|PB|市净率)",
        }
        for query, comparable_method in query_specs:
            try:
                response = requests.post(
                    "https://api.tavily.com/search",
                    json={"api_key": api_key, "query": query, "search_depth": "advanced", "max_results": 8},
                    timeout=30,
                )
                response.raise_for_status()
                for item in response.json().get("results", []):
                    title = item.get("title", query)
                    content = item.get("content", "")
                    url = item.get("url", "")
                    published = self._parse_date(item.get("published_date"))
                    information = published or self._parse_date(f"{title} {content[:800]}")
                    grade = self._source_grade(url, "web")
                    source_type = "official" if grade == "A" else ("research" if "研报" in title else "web")
                    excerpt = content
                    if comparable_method:
                        token = aliases[comparable_method]
                        match = re.search(
                            rf"{token}[^\d]{{0,30}}(\d+(?:\.\d+)?)\s*(?:倍|x|X)?",
                            f"{title} {content}", re.IGNORECASE,
                        )
                        period_match = re.search(
                            r"\b(?:TTM|LTM|20\d{2}[Ee]?|FY20\d{2})\b",
                            f"{title} {content}", re.IGNORECASE,
                        )
                        value = float(match.group(1)) if match else None
                        period = period_match.group(0).upper() if period_match else ""
                        comparison_text = f"{title} {content}"
                        peer_context = bool(
                            company.lower() in comparison_text.lower()
                            and re.search(r"可比公司|同行|同业|peer|comparable", comparison_text, re.IGNORECASE)
                        )
                        metadata = {
                            "subject": title,
                            "relationship": "peer" if peer_context else "other",
                            "metric": comparable_method,
                            "value": value,
                            "range_low": value,
                            "range_high": value,
                            "as_of": information.isoformat() if information else "",
                            "period": period,
                            "applicability": (
                                "检索摘要同时明确公司、同行/可比语境和该估值metric；仍需与盈利阶段、净债务及会计口径一致"
                                if value is not None and peer_context else ""
                            ),
                            "accounting_basis": (
                                f"source-reported {period} multiple" if period else ""
                            ),
                            "source_excerpt": content[:2500],
                        }
                        source_type = "comparable_valuation"
                        excerpt = json.dumps(metadata, ensure_ascii=False)
                    evidence.append(self._evidence(
                        title, title or "Tavily 搜索结果", url, source_type,
                        0.85 if grade == "A" else (0.72 if grade == "B" else 0.58),
                        excerpt, publication_date=published, information_date=information,
                        source_grade=grade,
                    ))
            except Exception as exc:
                evidence.append(self._evidence(
                    f"深度检索失败：{query}", "Tavily", "", "collection_error", 0.2, str(exc)
                ))
        return evidence

    def collect(self, request: ResearchRequest) -> dict[str, Any]:
        symbol = self.resolve_symbol(request)
        company = request.company
        evidence: list[Evidence] = []
        market_snapshot: MarketSnapshot | None = None

        if symbol:
            realtime = self._invoke(get_stock_realtime, symbol=symbol)
            fundamental = self._invoke(get_stock_fundamental, symbol=symbol)
            history = self._invoke(get_stock_history, symbol=symbol, days=260)
            realtime_evidence = self._evidence(
                "当前行情快照", "腾讯证券", "https://stockapp.finance.qq.com",
                "market_data", 0.85, realtime, information_date=date.today(), source_grade="B",
            )
            fundamental_evidence = self._evidence(
                "当前估值与市值快照", "腾讯证券", "https://stockapp.finance.qq.com",
                "market_data", 0.85, fundamental, information_date=date.today(), source_grade="B",
            )
            history_evidence = self._evidence(
                "近260个交易日价格区间", "腾讯证券历史行情", "https://stockapp.finance.qq.com",
                "market_data", 0.82, history, information_date=date.today(), source_grade="B",
            )
            evidence.extend([realtime_evidence, fundamental_evidence, history_evidence])
            financial_materials = self._akshare_materials(symbol, request.analysis_date)
            evidence.extend(financial_materials)
            market_snapshot = self._market_snapshot(
                realtime, fundamental, history,
                [realtime_evidence.id, fundamental_evidence.id, history_evidence.id],
            )
            market_snapshot = self._enrich_market_snapshot(
                market_snapshot, financial_materials, request.analysis_date
            )

        news = self._invoke(get_stock_news, symbol=company, count=10)
        news_date = self._parse_date(news)
        evidence.append(self._evidence(
            "近期财经新闻摘要", "新浪财经", "https://finance.sina.com.cn",
            "news", 0.6, news, information_date=news_date, source_grade="B",
        ))

        try:
            notes = search_knowledge(f"{company} {symbol}", top_k=8)
            for note in notes:
                metadata = note.get("metadata", {})
                info_date = self._parse_date(
                    metadata.get("information_date") or metadata.get("date") or note.get("content", "")[:500]
                )
                evidence.append(self._evidence(
                    metadata.get("title", "本地知识库资料"),
                    metadata.get("source", "本地 Chroma 知识库"), "",
                    metadata.get("type", "knowledge"), 0.65, note.get("content", ""),
                    information_date=info_date,
                    source_grade=(metadata.get("source_grade") if metadata.get("source_grade") in {"A", "B", "C"} else "C"),
                ))
        except Exception:
            pass

        evidence.extend(self._tavily_materials(company, request.search_depth == "deep"))
        evidence = [
            Evidence.model_validate({**item.model_dump(mode="json"), "analysis_date": request.analysis_date})
            for item in evidence
        ]
        comparable_catalog: list[ComparableValuationEvidenceItem] = []
        for item in evidence:
            if item.source_type != "comparable_valuation":
                continue
            try:
                metadata = json.loads(item.excerpt or "{}")
            except (TypeError, json.JSONDecodeError):
                metadata = {}
            method = str(metadata.get("metric") or "")
            if method not in {"pe", "ev_ebitda", "ev_sales", "pb"}:
                continue
            has_value = metadata.get("value") is not None or (
                metadata.get("range_low") is not None and metadata.get("range_high") is not None
            )
            relationship = metadata.get("relationship")
            valid = bool(
                relationship in {"peer", "company_historical"}
                and item.supports_current_claim and item.source_grade in {"A", "B"}
                and metadata.get("as_of") and metadata.get("period")
                and metadata.get("applicability") and metadata.get("accounting_basis") and has_value
            )
            comparable_catalog.append(ComparableValuationEvidenceItem(
                subject=str(metadata.get("subject") or company),
                symbol=str(metadata.get("symbol") or symbol),
                relationship=(
                    metadata.get("relationship")
                    if metadata.get("relationship") in {"peer", "company_historical", "other"}
                    else "other"
                ),
                method=method,
                metric=method,
                value=metadata.get("value"),
                range_low=metadata.get("range_low"),
                range_high=metadata.get("range_high"),
                as_of=str(metadata.get("as_of") or ""),
                period=str(metadata.get("period") or ""),
                source=item.source,
                evidence_ref=item.id,
                applicability=str(metadata.get("applicability") or ""),
                accounting_basis=str(metadata.get("accounting_basis") or ""),
                valid=valid,
                reason=("采集层结构化校验通过" if valid else "缺少当前A/B级来源、倍数值、日期、期间、适用性或会计口径"),
            ))
        limitations = []
        if not symbol:
            limitations.append("未能自动解析股票代码，行情和财务数据未采集。")
        if market_snapshot is not None and market_snapshot.market_cap is None:
            limitations.append("行情源未返回总市值，Model-Implied Earnings 只能保留公式。")
        if market_snapshot is not None and not market_snapshot.consistency_report.valuation_allowed:
            limitations.append(f"市场数据一致性检查失败：{market_snapshot.consistency_report.summary}")
        if not os.getenv("TAVILY_API_KEY"):
            limitations.append("未配置 TAVILY_API_KEY，未执行公告、研报和产业链深度网络检索。")
        return {
            "company": company,
            "symbol": symbol,
            "analysis_date": request.analysis_date.isoformat(),
            "market_snapshot": market_snapshot.model_dump(mode="json") if market_snapshot else None,
            "evidence": [item.model_dump(mode="json") for item in evidence],
            "comparable_valuation_catalog": [item.model_dump(mode="json") for item in comparable_catalog],
            "limitations": limitations,
        }
