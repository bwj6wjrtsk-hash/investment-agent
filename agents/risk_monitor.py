"""
风控监控 Agent - 监控持仓股票异动，触发预警
"""
import os
from datetime import datetime
from statistics import pstdev

from tools import quote as _quote


class RiskMonitor:
    """持仓风控监控"""

    def __init__(self):
        self.portfolio = {}
        self.watchlist = {}
        self._load_portfolio()
        self._load_watchlist()

        # 默认阈值（作为 fallback）
        self.alert_thresholds = {
            "daily_drop_pct": -3.0,
            "daily_rise_pct": 5.0,
            "volume_ratio": 3.0,
        }
        # 动态阈值缓存：{code: {"drop": float, "rise": float}}
        self._dynamic_thresholds = {}
        self._thresholds_loaded = False

    def _load_portfolio(self):
        """从 portfolio.json 加载持仓"""
        import json
        portfolio_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "portfolio.json")
        if os.path.exists(portfolio_file):
            with open(portfolio_file, "r", encoding="utf-8") as f:
                items = json.load(f)
                for item in items:
                    self.portfolio[item["code"]] = {
                        "name": item["name"],
                        "cost": item["cost"],
                        "shares": item["shares"],
                        "stop_loss": item.get("stop_loss", item["cost"] * 0.9),
                        "take_profit": item.get("take_profit", item["cost"] * 1.3),
                        "first_buy_date": item.get("first_buy_date", ""),
                        "sector": item.get("sector", ""),
                        "sector_keywords": item.get("sector_keywords", []),
                    }

    def _load_watchlist(self):
        """从 watchlist.json 加载自选股"""
        import json
        watchlist_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "watchlist.json")
        if os.path.exists(watchlist_file):
            with open(watchlist_file, "r", encoding="utf-8") as f:
                items = json.load(f)
                for item in items:
                    code = item["code"]
                    # 如果在持仓里，用持仓数据；否则构造一个观察用的条目
                    if code in self.portfolio:
                        self.watchlist[code] = self.portfolio[code]
                    else:
                        self.watchlist[code] = {
                            "name": item["name"],
                            "market": item.get("market", ""),
                            "cost": item.get("cost", 0),
                            "shares": item.get("shares", 0),
                            "stop_loss": item.get("stop_loss", 0),
                            "take_profit": item.get("take_profit", 0),
                            "first_buy_date": item.get("first_buy_date", ""),
                            "sector": item.get("sector", ""),
                            "sector_keywords": item.get("sector_keywords", []),
                        }

    def _watch_symbol(self, code):
        """自选记录存在显式市场时优先使用，避免同代码跨市场冲突。"""
        market = str(self.watchlist.get(code, {}).get("market", "")).lower()
        if market in ("sh", "sz", "bj", "hk"):
            return market + code
        return _quote.code_prefix(code)

    def _calc_dynamic_thresholds(self):
        """根据近60日波动率计算预警阈值；首次冷加载时并行读取多只 K 线。"""
        import math
        from concurrent.futures import ThreadPoolExecutor

        if not self.watchlist:
            return

        def calculate(code):
            try:
                closes = _quote.get_closes(self._watch_symbol(code), 60)
                if not closes or len(closes) < 10:
                    return code, None
                daily_returns = [
                    (closes[i] / closes[i - 1] - 1) * 100
                    for i in range(1, len(closes)) if closes[i - 1] != 0
                ]
                if not daily_returns:
                    return code, None
                mean = sum(daily_returns) / len(daily_returns)
                variance = sum((value - mean) ** 2 for value in daily_returns) / len(daily_returns)
                sigma = math.sqrt(variance)
                return code, {
                    "drop": round(max(-max(2 * sigma, 1.5), -8.0), 2),
                    "rise": round(min(max(2 * sigma, 2.0), 10.0), 2),
                    "sigma": round(sigma, 2),
                }
            except Exception:
                return code, None

        codes = list(self.watchlist)
        with ThreadPoolExecutor(max_workers=min(4, len(codes))) as executor:
            for code, thresholds in executor.map(calculate, codes):
                if thresholds:
                    self._dynamic_thresholds[code] = thresholds

    def _get_market_status(self, session=None):
        """获取大盘环境状态（只请求一次，供多只股票复用）"""
        from tools import quote as _q
        try:
            idx_closes = _q.get_closes("sh000001", 120)
            if idx_closes and len(idx_closes) >= 20:
                idx_high = max(idx_closes)
                idx_low = min(idx_closes)
                idx_now = idx_closes[-1]
                idx_drawdown = (idx_now / idx_high - 1) * 100
                idx_position = (idx_now - idx_low) / (idx_high - idx_low) * 100 if idx_high != idx_low else 50

                if idx_drawdown <= -8:
                    return "panic"
                elif idx_position >= 90:
                    return "euphoria"
                elif idx_drawdown <= -5:
                    return "weak"
                elif idx_position >= 75:
                    return "strong"
        except:
            pass
        return "normal"

    def _fetch_klines(self, code, session=None):
        """获取单只股票120日K线收盘价列表（统一走 quote 层，带缓存）"""
        from tools import quote as _q
        closes = _q.get_closes(code, 120)
        return closes if closes else None

    def _get_signal(self, current_price, position=None, info=None, klines=None,
                    market_status=None, market_return_20=None, code=None,
                    is_held=False, watch_config=None):
        """生成保守、可解释的操作建议；持仓与观察标的使用不同动作语义。"""
        try:
            closes = klines if klines is not None else _quote.get_closes(code, 120)
            market_status = market_status or self._get_market_status()
            watch_config = watch_config or {}
            window_days = len(closes) if closes else 0
            market_labels = {
                "panic": "大盘恐慌", "weak": "大盘偏弱", "normal": "大盘中性",
                "strong": "大盘偏强", "euphoria": "大盘偏热",
            }
            # 港股不套用上证指数门控，避免跨市场误判。
            effective_market = "normal" if str(code or "").lower().startswith("hk") else market_status

            metrics = {
                "price": round(float(current_price), 4),
                "pnl_pct": None,
                "drawdown_pct": None,
                "rsi14": None,
                "ma20": None,
                "ma60": None,
                "ma_long": None,
                "ma_long_label": None,
                "return_20_pct": None,
                "relative_strength_20_pct": None,
                "volatility_20_pct": None,
                "ma20_slope_5_pct": None,
                "market_status": effective_market,
                "window_days": window_days,
            }

            def response(action, level, title, reasons):
                icons = {
                    "critical": "🔴", "warning": "🟡", "opportunity": "🟢",
                    "neutral": "⚪", "info": "⚠️",
                }
                reason_text = "；".join(reasons)
                return {
                    "position_state": "held" if is_held else "watch",
                    "action": action,
                    "level": level,
                    "reasons": reasons,
                    "metrics": metrics,
                    "signal": f"{icons.get(level, '⚪')} {title}：{reason_text}",
                }

            def positive_number(value):
                try:
                    number = float(value)
                    return number if number > 0 else None
                except (TypeError, ValueError):
                    return None

            if is_held:
                cost = positive_number((position or {}).get("cost"))
                stop_loss = positive_number((position or {}).get("stop_loss"))
                take_profit = positive_number((position or {}).get("take_profit"))
                profit_pct = (current_price / cost - 1) * 100 if cost else 0
                metrics["pnl_pct"] = round(profit_pct, 2)
                if stop_loss and current_price <= stop_loss:
                    return response("stop_loss", "critical", "执行止损/减仓", [
                        f"现价{current_price:.3f}已触及止损线{stop_loss:.3f}",
                        "停止加仓并优先控制风险",
                    ])
                if take_profit and current_price >= take_profit:
                    return response("take_profit", "warning", "分批止盈", [
                        f"现价{current_price:.3f}已达到止盈线{take_profit:.3f}",
                        f"当前收益{profit_pct:+.1f}%",
                    ])
            else:
                observe_stop = positive_number(watch_config.get("stop_loss"))
                observe_target = positive_number(watch_config.get("target_price"))
                if observe_stop and current_price <= observe_stop:
                    return response("observe", "warning", "暂不介入", [
                        f"现价{current_price:.3f}已低于观察止损价{observe_stop:.3f}",
                        "等待价格重新企稳",
                    ])
                if observe_target and current_price >= observe_target:
                    return response("avoid_chase", "warning", "暂不追高", [
                        f"现价{current_price:.3f}已达到观察目标价{observe_target:.3f}",
                    ])

            if not closes or len(closes) < 60:
                subject = "持有观察" if is_held else "继续观察"
                return response("data_insufficient", "info", subject, [
                    f"历史数据仅{window_days}日，至少需要60日才能确认趋势",
                ])

            window = closes[-120:]
            high_n = max(window)
            # 先归一化再做阈值判断，避免恰好 5%/8%/12% 时受浮点误差影响。
            drawdown = round(min(0.0, (current_price / high_n - 1) * 100), 4) if high_n else 0
            ma20 = sum(closes[-20:]) / 20
            ma60 = sum(closes[-60:]) / 60
            if len(closes) >= 120:
                ma_long = sum(closes[-120:]) / 120
                ma_long_label = "MA120"
            else:
                ma_long = ma60
                ma_long_label = "MA60"
            rsi14 = _quote.compute_rsi(closes)
            trend_ok = current_price >= ma_long and ma20 >= ma60
            near_ma20 = abs(current_price / ma20 - 1) * 100 <= 3 if ma20 else False
            stock_return_20 = (current_price / closes[-21] - 1) * 100 if len(closes) >= 21 and closes[-21] else None
            ma20_5ago = sum(closes[-25:-5]) / 20 if len(closes) >= 25 else None
            ma20_slope_5 = (ma20 / ma20_5ago - 1) * 100 if ma20_5ago else None
            recent_returns = [
                (closes[index] / closes[index - 1] - 1) * 100
                for index in range(len(closes) - 20, len(closes))
                if closes[index - 1]
            ]
            volatility_20 = pstdev(recent_returns) if len(recent_returns) == 20 else None
            relative_strength_20 = (
                stock_return_20 - float(market_return_20)
                if stock_return_20 is not None and market_return_20 is not None else None
            )
            bearish_avoid = (
                not str(code or "").lower().startswith("hk")
                and current_price <= ma_long and ma20 <= ma60
                and stock_return_20 is not None and -30 <= stock_return_20 <= -2
                and rsi14 is not None and 35 <= rsi14 <= 48
                and ma20_slope_5 is not None and ma20_slope_5 <= 0.25
                and relative_strength_20 is not None and relative_strength_20 <= -1
                and volatility_20 is not None and volatility_20 <= 4.5
            )
            metrics.update({
                "drawdown_pct": round(drawdown, 2),
                "rsi14": round(rsi14, 2) if rsi14 is not None else None,
                "ma20": round(ma20, 4),
                "ma60": round(ma60, 4),
                "ma_long": round(ma_long, 4),
                "ma_long_label": ma_long_label,
                "return_20_pct": round(stock_return_20, 2) if stock_return_20 is not None else None,
                "relative_strength_20_pct": round(relative_strength_20, 2) if relative_strength_20 is not None else None,
                "volatility_20_pct": round(volatility_20, 2) if volatility_20 is not None else None,
                "ma20_slope_5_pct": round(ma20_slope_5, 2) if ma20_slope_5 is not None else None,
            })
            context = [
                f"阶段回撤{drawdown:.1f}%",
                f"RSI {rsi14:.1f}" if rsi14 is not None else "RSI不可用",
                market_labels.get(effective_market, "市场状态未知"),
            ]

            if is_held:
                profit_pct = metrics["pnl_pct"] or 0
                if current_price < ma_long and profit_pct <= -8:
                    return response("reduce", "warning", "减仓并暂停加仓", [
                        f"浮亏{profit_pct:.1f}%且价格跌破{ma_long_label}", *context,
                    ])
                trailing_triggers = []
                if current_price < ma20:
                    trailing_triggers.append("跌破MA20")
                if drawdown <= -8:
                    trailing_triggers.append(f"高点回撤{abs(drawdown):.1f}%")
                if rsi14 is not None and rsi14 >= 70:
                    trailing_triggers.append(f"RSI升至{rsi14:.1f}")
                if profit_pct >= 15 and trailing_triggers:
                    return response("take_profit", "warning", "分批止盈", [
                        f"当前收益{profit_pct:+.1f}%", *trailing_triggers,
                    ])
                if bearish_avoid:
                    return response("hold", "warning", "暂停加仓并评估风险", [
                        f"20日跌幅{stock_return_20:.1f}%且弱于上证同期{abs(relative_strength_20):.1f}个百分点",
                        f"价格位于{ma_long_label}下方且MA20不高于MA60",
                        "扩大样本未支持方向预测，仅作为持仓风险提示",
                    ])
                if (effective_market in ("normal", "strong") and trend_ok and
                        -12 <= drawdown <= -5 and rsi14 is not None and 35 <= rsi14 <= 55):
                    return response("hold", "neutral", "保持仓位，暂不追加", [
                        "回撤条件满足，但历史校准尚不支持直接触发加仓", *context,
                    ])
                hold_reasons = [f"当前收益{profit_pct:+.1f}%", *context]
                if effective_market in ("panic", "weak"):
                    hold_reasons.append("市场偏弱，暂停加仓")
                elif effective_market == "euphoria":
                    hold_reasons.append("市场偏热，避免追高")
                elif not trend_ok:
                    hold_reasons.append(f"趋势尚未重新站稳{ma_long_label}")
                return response("hold", "neutral", "继续持有", hold_reasons)

            if bearish_avoid:
                return response("observe", "neutral", "弱势观察，暂不操作", [
                    f"20日跌幅{stock_return_20:.1f}%且弱于上证同期{abs(relative_strength_20):.1f}个百分点",
                    f"价格位于{ma_long_label}下方且MA20不高于MA60",
                    f"RSI {rsi14:.1f}、20日波动率{volatility_20:.1f}%显示趋势偏弱",
                    "扩大样本验证未达到65%，因此不触发方向性操作建议",
                ])
            if effective_market in ("panic", "weak"):
                return response("observe", "neutral", "继续观察", [
                    market_labels[effective_market], "等待市场和价格共同企稳", *context[:2],
                ])
            if effective_market == "euphoria" or (rsi14 is not None and rsi14 >= 70):
                return response("avoid_chase", "warning", "暂不追高", [
                    "市场或个股处于偏热区域", *context,
                ])
            if (effective_market in ("normal", "strong") and trend_ok and near_ma20 and
                    -12 <= drawdown <= -5 and rsi14 is not None and 35 <= rsi14 <= 55):
                return response("observe", "neutral", "等待更高置信度信号", [
                    "趋势回撤条件满足，但180日校准未达到买入建议阈值", *context,
                ])
            observe_reasons = [*context]
            if not trend_ok:
                observe_reasons.append(f"尚未满足价格站上{ma_long_label}且MA20不弱于MA60")
            elif drawdown > -5:
                observe_reasons.append("回撤不足，等待更合适的风险收益位置")
            elif not near_ma20:
                observe_reasons.append("价格距离MA20较远")
            return response("observe", "neutral", "继续观察", observe_reasons)

        except Exception as error:
            return {
                "position_state": "held" if is_held else "watch",
                "action": "data_error",
                "level": "info",
                "reasons": [f"指标计算异常：{type(error).__name__}"],
                "metrics": {"price": current_price},
                "signal": f"⚠️ 数据异常，暂无法给出建议（{type(error).__name__}）",
            }

    def _get_policy_signal(self, session):
        """使用AI评估政策/宏观面对银行板块的影响"""
        import os
        try:
            # 获取银行相关新闻
            keywords = ["银行", "利率", "央行", "降息", "降准", "LPR", "房贷", "信贷",
                        "存款", "贷款", "金融监管", "净息差", "MLF", "逆回购"]
            news = []
            for page in range(1, 3):
                resp = session.get(
                    "https://feed.mix.sina.com.cn/api/roll/get",
                    params={"pageid": "153", "lid": "2516", "num": "40", "page": str(page)},
                    timeout=5
                )
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("result") and data["result"].get("data"):
                        for item in data["result"]["data"]:
                            title = item.get("title", "")
                            if title and any(kw in title for kw in keywords):
                                if title not in news:
                                    news.append(title)
                                    if len(news) >= 5:
                                        break
                if len(news) >= 5:
                    break

            if not news:
                return None

            # 调用AI评估
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                model="deepseek-chat",
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com",
                temperature=0.1,
            )

            news_text = "\n".join(news)
            prompt = f"""根据以下新闻，判断对银行板块的影响方向。

新闻：
{news_text}

请严格按以下JSON格式回答，不要多余文字：
{{"direction": "bullish或bearish或neutral", "summary": "一句话总结原因(10字以内)"}}"""

            resp_ai = llm.invoke(prompt)
            content = resp_ai.content.strip()

            # 解析JSON
            import json
            # 处理可能的markdown代码块
            if "```" in content:
                content = content.split("```")[1].replace("json", "").strip()
            result = json.loads(content)
            return result

        except Exception:
            return None

    def get_signals(self) -> list:
        """Get trading signals for all watchlist stocks (including those not yet held)"""
        import json

        # Load watchlist
        watchlist_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "watchlist.json")
        watchlist = []
        if os.path.exists(watchlist_file):
            with open(watchlist_file, "r", encoding="utf-8") as f:
                watchlist = json.load(f)

        if not watchlist:
            return []

        # Fetch current prices using the persisted market when present.
        def _item_symbol(item):
            market = str(item.get("market", "")).lower()
            if market in ("sh", "sz", "bj", "hk"):
                return market + item["code"]
            return _quote.code_prefix(item["code"])

        symbols_by_code = {item["code"]: _item_symbol(item) for item in watchlist}
        quotes = _quote.get_quotes(list(symbols_by_code.values()))
        if not quotes:
            return []

        prices = {}
        for code, symbol in symbols_by_code.items():
            fields = quotes.get(symbol)
            if not fields or len(fields) <= _quote.F_PRICE:
                continue
            try:
                prices[code] = float(fields[_quote.F_PRICE]) if fields[_quote.F_PRICE] else 0
            except (TypeError, ValueError):
                continue

        signals = []
        # Pre-fetch market context once (instead of per-stock).
        market_status = self._get_market_status()
        market_closes = _quote.get_closes("sh000001", 120)
        market_return_20 = None
        if market_closes and len(market_closes) >= 21 and market_closes[-21]:
            market_return_20 = (market_closes[-1] / market_closes[-21] - 1) * 100

        # Pre-fetch K-line data for all watchlist stocks using thread pool
        from concurrent.futures import ThreadPoolExecutor
        klines_cache = {}
        stock_codes = [item["code"] for item in watchlist if prices.get(item["code"], 0) > 0]

        def _fetch_one(code):
            return code, self._fetch_klines(symbols_by_code[code])

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(_fetch_one, code) for code in stock_codes]
            for future in futures:
                code, kline = future.result()
                if kline:
                    klines_cache[code] = kline

        for item in watchlist:
            code = item["code"]
            name = item["name"]
            current_price = prices.get(code, 0)
            if current_price == 0:
                continue

            is_held = code in self.portfolio
            pos = self.portfolio.get(code)
            signal_result = self._get_signal(
                current_price,
                position=pos,
                info={},
                klines=klines_cache.get(code),
                market_status=market_status,
                market_return_20=market_return_20,
                code=symbols_by_code.get(code, code),
                is_held=is_held,
                watch_config=item,
            )
            signals.append({
                "code": code,
                "market": item.get("market") or symbols_by_code.get(code, "")[:2],
                "name": name,
                **signal_result,
            })

        level_priority = {"critical": 0, "warning": 1, "opportunity": 2, "neutral": 3, "info": 4}
        signals.sort(key=lambda row: (
            0 if row.get("position_state") == "held" else 1,
            level_priority.get(row.get("level"), 9),
            row.get("name", ""),
        ))
        return signals

    def add_position(self, code: str, name: str, cost: float,
                     shares: int, stop_loss: float = None, take_profit: float = None):
        """添加持仓"""
        if stop_loss is None:
            stop_loss = cost * 0.9  # 默认10%止损
        if take_profit is None:
            take_profit = cost * 1.3  # 默认30%止盈

        self.portfolio[code] = {
            "name": name,
            "cost": cost,
            "shares": shares,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
        }

    def remove_position(self, code: str):
        """移除持仓"""
        if code in self.portfolio:
            del self.portfolio[code]

    def check_all(self) -> list:
        """检查所有自选股，返回预警列表"""
        alerts = []
        if not self.watchlist:
            return [{"level": "info", "message": "未配置自选股"}]

        symbols_by_code = {code: self._watch_symbol(code) for code in self.watchlist}
        quotes = _quote.get_quotes(list(symbols_by_code.values()))
        if not quotes:
            return [{"level": "error", "message": "获取行情失败，请稍后重试"}]

        prices = {}
        changes = {}
        for code, symbol in symbols_by_code.items():
            fields = quotes.get(symbol)
            if not fields or len(fields) <= _quote.F_CHANGE_PCT:
                continue
            try:
                prices[code] = float(fields[_quote.F_PRICE]) if fields[_quote.F_PRICE] else 0
                changes[code] = float(fields[_quote.F_CHANGE_PCT]) if fields[_quote.F_CHANGE_PCT] else 0
            except (TypeError, ValueError):
                continue

        for code, position in self.watchlist.items():
            current_price = prices.get(code, 0)
            change_pct = changes.get(code, 0)

            if current_price == 0:
                continue

            # 止损/止盈检查（仅对有持仓的股票）
            if code in self.portfolio:
                if current_price <= position["stop_loss"]:
                    alerts.append({
                        "level": "critical",
                        "stock": f"{position['name']}({code})",
                        "message": f"触发止损！现价 {current_price}，止损线 {position['stop_loss']}",
                        "action": "建议立即卖出"
                    })
                elif current_price >= position["take_profit"]:
                    alerts.append({
                        "level": "warning",
                        "stock": f"{position['name']}({code})",
                        "message": f"触发止盈！现价 {current_price}，止盈线 {position['take_profit']}",
                        "action": "建议分批止盈"
                    })

            # 大幅下跌（使用动态阈值，所有自选股都检查）
            if not self._thresholds_loaded:
                self._calc_dynamic_thresholds()
                self._thresholds_loaded = True
            thresholds = self._dynamic_thresholds.get(code)
            drop_limit = thresholds["drop"] if thresholds else self.alert_thresholds["daily_drop_pct"]
            rise_limit = thresholds["rise"] if thresholds else self.alert_thresholds["daily_rise_pct"]

            if change_pct <= drop_limit:
                alerts.append({
                    "level": "warning",
                    "stock": f"{position['name']}({code})",
                    "message": f"大幅下跌 {change_pct}%（阈值{drop_limit}%），现价 {current_price}",
                    "action": "关注是否有利空消息"
                })

            # 大幅上涨（使用动态阈值）
            if change_pct >= rise_limit:
                alerts.append({
                    "level": "warning",
                    "stock": f"{position['name']}({code})",
                    "message": f"大幅上涨 {change_pct}%（阈值+{rise_limit}%），现价 {current_price}",
                    "action": "关注利好消息，可考虑止盈"
                })

        if not alerts:
            alerts.append({
                "level": "info",
                "message": f"所有自选股正常 ({datetime.now().strftime('%H:%M:%S')})"
            })

        return alerts

    def get_portfolio_summary(self) -> str:
        """获取持仓概览；只计算页面实际展示字段，行情统一复用 TTL 缓存。"""
        if not self.portfolio:
            return "未配置持仓。请编辑 portfolio.json 添加持仓。"

        quotes = _quote.get_quotes(list(self.portfolio.keys()))
        if not quotes:
            return "获取行情失败，请稍后重试"

        result = "=== 持仓概览 ===\n\n"
        total_cost = 0
        total_value = 0

        for code, pos in self.portfolio.items():
            fields = quotes.get(_quote.code_prefix(code))
            if not fields or len(fields) <= _quote.F_CHANGE_PCT:
                continue
            try:
                current_price = float(fields[_quote.F_PRICE]) if fields[_quote.F_PRICE] else 0
                change_pct = float(fields[_quote.F_CHANGE_PCT]) if fields[_quote.F_CHANGE_PCT] else 0
            except (TypeError, ValueError):
                continue
            if current_price <= 0:
                continue

            cost_value = pos["cost"] * pos["shares"]
            market_value = current_price * pos["shares"]
            profit = market_value - cost_value
            profit_pct = (current_price / pos["cost"] - 1) * 100
            total_cost += cost_value
            total_value += market_value

            emoji = "+" if profit >= 0 else "-"
            today_profit = change_pct / 100 * current_price * pos["shares"]
            result += (
                f"[{emoji}] {pos['name']}({code})\n"
                f"   现价: {current_price} | 成本: {pos['cost']} | 份额: {pos['shares']}\n"
                f"   今日涨跌: {change_pct:+.2f}% | 今日盈亏: {today_profit:+.2f}元\n"
                f"   盈亏: {profit:+.2f}元 ({profit_pct:+.2f}%)\n\n"
            )

        total_profit = total_value - total_cost
        total_pct = (total_value / total_cost - 1) * 100 if total_cost > 0 else 0
        result += f"{'='*30}\n"
        result += f"总投入: {total_cost:.2f}元\n"
        result += f"总市值: {total_value:.2f}元\n"
        result += f"总盈亏: {total_profit:+.2f}元 ({total_pct:+.2f}%)\n"
        return result
