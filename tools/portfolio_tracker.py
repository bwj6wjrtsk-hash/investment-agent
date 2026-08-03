"""
收益追踪工具
- 每日持仓市值快照
- 收益曲线数据
- 周/月/年维度统计
"""
import json
import os
import re
import requests
from datetime import datetime, date, timedelta
from langchain_core.tools import tool

from tools import quote as _q

TRACKER_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "portfolio_history.json")


def _period_return(first, last):
    """区间收益率（Modified Dietz，扣除期间加/减仓的资金流），返回百分数或 None。
    净流入用成本基变化近似：total_cost 只随买卖变动、不随行情变动。"""
    v0 = first.get("total_value", 0) or 0
    v1 = last.get("total_value", 0) or 0
    flow = (last.get("total_cost", 0) or 0) - (first.get("total_cost", 0) or 0)
    denom = v0 + flow * 0.5
    if denom <= 0:
        return None
    gain = (v1 - v0) - flow
    return gain / denom * 100


def _daily_return(prev, cur):
    """单日收益率，扣除当日资金流（加/减仓），返回百分数或 None。"""
    v0 = prev.get("total_value", 0) or 0
    flow = (cur.get("total_cost", 0) or 0) - (prev.get("total_cost", 0) or 0)
    if v0 <= 0:
        return None
    return ((cur.get("total_value", 0) or 0) - v0 - flow) / v0 * 100


def _ensure_data_dir():
    """Ensure data directory exists"""
    data_dir = os.path.dirname(TRACKER_FILE)
    os.makedirs(data_dir, exist_ok=True)


def _load_history() -> list:
    """Load portfolio history"""
    _ensure_data_dir()
    if os.path.exists(TRACKER_FILE):
        with open(TRACKER_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def _save_history(history: list):
    """Save portfolio history"""
    _ensure_data_dir()
    with open(TRACKER_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def _load_portfolio():
    """Load current portfolio"""
    portfolio_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "portfolio.json")
    if os.path.exists(portfolio_file):
        with open(portfolio_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def _get_prices(codes: list) -> dict:
    """Batch get current prices"""
    prefixed = [_q.code_prefix(code) for code in codes]

    with requests.Session() as session:
        session.trust_env = False
        resp = session.get(f"https://qt.gtimg.cn/q={','.join(prefixed)}", timeout=10)
        resp.encoding = "gbk"

    prices = {}
    for line in resp.text.strip().split(";"):
        line = line.strip()
        if not line:
            continue
        match = re.match(r'v_(\w+)="(.+)"', line)
        if match:
            fields = match.group(2).split("~")
            if len(fields) > 3 and fields[3]:
                stock_code = fields[2]
                prices[stock_code] = float(fields[3])
    return prices


def record_daily_snapshot():
    """Record today's portfolio snapshot (call once per day)"""
    portfolio = _load_portfolio()
    if not portfolio:
        return None

    codes = [item["code"] for item in portfolio]
    prices = _get_prices(codes)

    today = date.today().isoformat()
    total_cost = 0
    total_value = 0
    positions = []

    for item in portfolio:
        code = item["code"]
        current_price = prices.get(code, 0)
        if current_price == 0:
            continue
        cost_value = item["cost"] * item["shares"]
        market_value = current_price * item["shares"]
        total_cost += cost_value
        total_value += market_value
        positions.append({
            "code": code,
            "name": item["name"],
            "price": current_price,
            "shares": item["shares"],
            "cost": item["cost"],
            "value": round(market_value, 2),
        })

    snapshot = {
        "date": today,
        "total_cost": round(total_cost, 2),
        "total_value": round(total_value, 2),
        "profit": round(total_value - total_cost, 2),
        "profit_pct": round((total_value / total_cost - 1) * 100, 2) if total_cost > 0 else 0,
        "positions": positions,
    }

    # Load existing and append/update
    history = _load_history()

    # Replace if today already exists
    history = [h for h in history if h["date"] != today]
    history.append(snapshot)
    history.sort(key=lambda x: x["date"])

    _save_history(history)
    return snapshot


@tool
def record_portfolio_snapshot() -> str:
    """记录今日持仓快照（每日调用一次，用于生成收益曲线）。"""
    try:
        snapshot = record_daily_snapshot()
        if not snapshot:
            return "无持仓数据可记录"
        return (
            f"✅ 已记录 {snapshot['date']} 持仓快照:\n"
            f"   总投入: {snapshot['total_cost']:.2f}元\n"
            f"   总市值: {snapshot['total_value']:.2f}元\n"
            f"   盈亏: {snapshot['profit']:+.2f}元 ({snapshot['profit_pct']:+.2f}%)"
        )
    except Exception as e:
        return f"记录快照失败: {e}"


@tool
def get_profit_curve(period: str = "all") -> str:
    """获取收益曲线数据。
    参数:
    - period: 时间范围，'week'=最近一周, 'month'=最近一月, 'year'=最近一年, 'all'=全部
    """
    try:
        history = _load_history()
        if not history:
            return "暂无历史数据。请先使用 record_portfolio_snapshot 记录每日快照。"

        # Filter by period
        today = date.today()
        if period == "week":
            start = today - timedelta(days=7)
        elif period == "month":
            start = today - timedelta(days=30)
        elif period == "year":
            start = today - timedelta(days=365)
        else:
            start = date(2000, 1, 1)

        filtered = [h for h in history if h["date"] >= start.isoformat()]
        if not filtered:
            return f"在 {period} 范围内暂无数据"

        result = f"=== 收益曲线 ({period}) ===\n\n"
        result += f"数据范围: {filtered[0]['date']} ~ {filtered[-1]['date']} ({len(filtered)}天)\n\n"

        # Summary stats
        first = filtered[0]
        last = filtered[-1]
        period_return = _period_return(first, last)
        pr_str = f"{period_return:+.2f}%" if period_return is not None else "数据不足"

        result += f"【区间收益】\n"
        result += f"  期初市值: {first['total_value']:.2f}元 (盈亏{first['profit_pct']:+.2f}%)\n"
        result += f"  期末市值: {last['total_value']:.2f}元 (盈亏{last['profit_pct']:+.2f}%)\n"
        result += f"  区间收益率: {pr_str}（已扣除期间加减仓资金流）\n"

        # Max drawdown
        peak = filtered[0]["total_value"]
        max_dd = 0
        for h in filtered:
            if h["total_value"] > peak:
                peak = h["total_value"]
            dd = (h["total_value"] / peak - 1) * 100
            if dd < max_dd:
                max_dd = dd
        result += f"  最大回撤: {max_dd:.2f}%\n"

        # Daily data
        result += f"\n【每日数据】\n"
        for h in filtered[-20:]:  # 最多显示20天
            emoji = "📈" if h["profit"] >= 0 else "📉"
            result += f"  {h['date']} | 市值{h['total_value']:.0f} | {emoji} {h['profit_pct']:+.2f}%\n"

        if len(filtered) > 20:
            result += f"  ... (共{len(filtered)}条记录，仅显示最近20条)\n"

        return result
    except Exception as e:
        return f"获取收益曲线失败: {e}"


@tool
def get_profit_stats() -> str:
    """获取收益统计数据（周/月/年维度）。"""
    try:
        history = _load_history()
        if not history:
            return "暂无历史数据"

        today = date.today()
        result = "=== 收益统计 ===\n\n"

        # Current status
        latest = history[-1]
        result += f"【当前状态】 ({latest['date']})\n"
        result += f"  总市值: {latest['total_value']:.2f}元\n"
        result += f"  总盈亏: {latest['profit']:+.2f}元 ({latest['profit_pct']:+.2f}%)\n\n"

        def _fmt_period(days, label):
            ago = (today - timedelta(days=days)).isoformat()
            data = [h for h in history if h["date"] >= ago]
            if len(data) >= 2:
                r = _period_return(data[0], data[-1])
                if r is not None:
                    return f"【{label}】收益率: {r:+.2f}%\n"
            return ""

        result += _fmt_period(7, "本周")
        result += _fmt_period(30, "本月")
        result += _fmt_period(365, "今年")

        # 单日收益（扣除当日加减仓资金流），跳过无法计算的日子
        daily_returns = []
        for i in range(1, len(history)):
            dr = _daily_return(history[i-1], history[i])
            if dr is not None:
                daily_returns.append({"date": history[i]["date"], "return": dr})

        # Win rate（基于市场涨跌，已剔除现金流影响）
        if daily_returns:
            win_days = sum(1 for d in daily_returns if d["return"] > 0)
            win_rate = win_days / len(daily_returns) * 100
            result += f"\n【胜率】{win_days}/{len(daily_returns)}天上涨 ({win_rate:.1f}%)\n"

            best = max(daily_returns, key=lambda x: x["return"])
            worst = min(daily_returns, key=lambda x: x["return"])
            result += f"【最佳】{best['date']} {best['return']:+.2f}%\n"
            result += f"【最差】{worst['date']} {worst['return']:+.2f}%\n"

        return result
    except Exception as e:
        return f"获取统计失败: {e}"
