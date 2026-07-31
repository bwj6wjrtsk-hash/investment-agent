"""
板块轮动监控工具
- 多板块强弱对比
- 轮动信号检测
- 资金流向追踪
"""
import json
import os
import re
import requests
from datetime import datetime, date, timedelta
from langchain_core.tools import tool

from tools import quote as _q

ROTATION_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "sector_rotation.json")


def _ensure_data_dir():
    data_dir = os.path.dirname(ROTATION_FILE)
    os.makedirs(data_dir, exist_ok=True)


def _load_rotation_history() -> list:
    _ensure_data_dir()
    if os.path.exists(ROTATION_FILE):
        with open(ROTATION_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def _save_rotation_history(data: list):
    _ensure_data_dir()
    with open(ROTATION_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# 主要板块的代表性ETF代码（用于获取精确涨跌数据）
SECTOR_ETFS = {
    "银行": "sh515290",
    "证券": "sh512880",
    "医药": "sh512010",
    "半导体": "sh512480",
    "新能源": "sh516160",
    "消费": "sh159928",
    "军工": "sh512660",
    "房地产": "sh512200",
    "科技": "sh515000",
    "食品饮料": "sh515170",
    "汽车": "sh516110",
    "电力": "sz159611",
    "人工智能": "sh515070",
}


def _get_sector_prices() -> dict:
    """Get all sector ETF prices through the shared short-lived quote cache."""
    codes = list(SECTOR_ETFS.values())
    quotes = _q.get_quotes(codes)
    results = {}
    for code in codes:
        fields = quotes.get(code)
        if not fields or len(fields) <= _q.F_TURNOVER:
            continue
        results[code] = {
            "name": fields[_q.F_NAME],
            "price": float(fields[_q.F_PRICE]) if fields[_q.F_PRICE] else 0,
            "change_pct": float(fields[_q.F_CHANGE_PCT]) if fields[_q.F_CHANGE_PCT] else 0,
            "volume": float(fields[_q.F_VOLUME]) if fields[_q.F_VOLUME] else 0,
            "amount": float(fields[_q.F_AMOUNT]) if fields[_q.F_AMOUNT] else 0,
            "turnover": float(fields[_q.F_TURNOVER]) if fields[_q.F_TURNOVER] else 0,
        }
    return results


@tool
def get_sector_strength_ranking() -> str:
    """获取各板块今日强弱排名（基于ETF涨跌幅）。"""
    try:
        prices = _get_sector_prices()
        if not prices:
            return "获取板块数据失败"

        sectors = []
        for sector_name, etf_code in SECTOR_ETFS.items():
            if etf_code in prices:
                data = prices[etf_code]
                sectors.append({
                    "name": sector_name,
                    "change_pct": data["change_pct"],
                    "turnover": data["turnover"],
                    "amount": data["amount"],
                })

        sectors.sort(key=lambda x: x["change_pct"], reverse=True)

        result = "=== 板块强弱排名（今日）===\n\n"
        result += f"{'排名':<4} {'板块':<8} {'涨跌幅':<10} {'换手率':<8} {'成交额(万)':<10}\n"
        result += "-" * 45 + "\n"

        for i, s in enumerate(sectors, 1):
            emoji = "🔴" if s["change_pct"] > 0 else "🟢" if s["change_pct"] < 0 else "⚪"
            result += f" {i:<3} {emoji} {s['name']:<7} {s['change_pct']:>+6.2f}%   {s['turnover']:>5.2f}%   {s['amount']:>8.0f}\n"

        # 强弱分析
        if sectors:
            strongest = sectors[0]
            weakest = sectors[-1]
            spread = strongest["change_pct"] - weakest["change_pct"]
            result += f"\n💡 今日分化: {spread:.2f}% (强{strongest['name']} vs 弱{weakest['name']})\n"
            if spread > 3:
                result += "   板块分化较大，注意轮动风险\n"
            elif spread < 1:
                result += "   板块同涨同跌，市场趋同\n"

        return result
    except Exception as e:
        return f"获取板块排名失败: {e}"


@tool
def detect_rotation_signal() -> str:
    """检测板块轮动信号（基于近期板块相对强度变化）。"""
    try:
        s = requests.Session()
        s.trust_env = False

        result = "=== 板块轮动信号 ===\n\n"

        # 获取各板块ETF近20日涨跌幅
        sector_performance = {}
        for sector_name, etf_code in SECTOR_ETFS.items():
            try:
                resp = s.get(
                    f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={etf_code},day,,,20,qfq",
                    timeout=5
                )
                data = resp.json()
                klines = data.get("data", {}).get(etf_code, {}).get("day", [])
                if not klines:
                    klines = data.get("data", {}).get(etf_code, {}).get("qfqday", [])

                if klines and len(klines) >= 10:
                    closes = [float(k[2]) for k in klines]
                    # 近5日涨幅
                    pct_5d = (closes[-1] / closes[-5] - 1) * 100 if len(closes) >= 5 else 0
                    # 近10日涨幅
                    pct_10d = (closes[-1] / closes[-10] - 1) * 100 if len(closes) >= 10 else 0
                    # 近20日涨幅
                    pct_20d = (closes[-1] / closes[0] - 1) * 100

                    # 动量变化：近5日 vs 前5日
                    momentum_change = 0
                    if len(closes) >= 10:
                        recent_5 = (closes[-1] / closes[-5] - 1) * 100
                        prev_5 = (closes[-6] / closes[-10] - 1) * 100
                        momentum_change = recent_5 - prev_5

                    sector_performance[sector_name] = {
                        "pct_5d": pct_5d,
                        "pct_10d": pct_10d,
                        "pct_20d": pct_20d,
                        "momentum_change": momentum_change,
                    }
            except:
                continue

        if not sector_performance:
            return "获取板块历史数据失败"

        # 按5日涨幅排序
        sorted_5d = sorted(sector_performance.items(), key=lambda x: x[1]["pct_5d"], reverse=True)

        result += "【近5日板块涨幅排名】\n"
        for name, perf in sorted_5d:
            emoji = "📈" if perf["pct_5d"] > 0 else "📉"
            result += f"  {emoji} {name:<8} 5日:{perf['pct_5d']:>+5.2f}%  10日:{perf['pct_10d']:>+5.2f}%  20日:{perf['pct_20d']:>+5.2f}%\n"

        # 动量加速的板块（轮动入场信号）
        accelerating = [(name, perf) for name, perf in sector_performance.items()
                        if perf["momentum_change"] > 1.0]
        accelerating.sort(key=lambda x: x[1]["momentum_change"], reverse=True)

        if accelerating:
            result += "\n🚀 【动量加速板块（可能的轮入方向）】\n"
            for name, perf in accelerating:
                result += f"  ➡️ {name}: 动量增强 +{perf['momentum_change']:.2f}% (近5日{perf['pct_5d']:+.2f}%)\n"

        # 动量减速的板块（轮动出场信号）
        decelerating = [(name, perf) for name, perf in sector_performance.items()
                        if perf["momentum_change"] < -1.0 and perf["pct_20d"] > 3]
        decelerating.sort(key=lambda x: x[1]["momentum_change"])

        if decelerating:
            result += "\n⚠️ 【动量减速板块（可能的轮出方向）】\n"
            for name, perf in decelerating:
                result += f"  ⬅️ {name}: 动量减弱 {perf['momentum_change']:.2f}% (前期涨幅{perf['pct_20d']:+.2f}%)\n"

        # 综合建议
        result += "\n💡 【轮动建议】\n"
        if accelerating:
            result += f"  关注: {', '.join([a[0] for a in accelerating[:3]])} — 动量正在增强\n"
        if decelerating:
            result += f"  警惕: {', '.join([d[0] for d in decelerating[:3]])} — 前期涨幅较大且动量减弱\n"

        return result
    except Exception as e:
        return f"检测轮动信号失败: {e}"


@tool
def compare_sectors(sector1: str, sector2: str) -> str:
    """对比两个板块的近期表现。
    参数:
    - sector1: 板块名称，如 '银行'
    - sector2: 板块名称，如 '证券'
    """
    try:
        s = requests.Session()
        s.trust_env = False

        if sector1 not in SECTOR_ETFS:
            return f"不支持的板块: {sector1}。可选: {', '.join(SECTOR_ETFS.keys())}"
        if sector2 not in SECTOR_ETFS:
            return f"不支持的板块: {sector2}。可选: {', '.join(SECTOR_ETFS.keys())}"

        result = f"=== {sector1} vs {sector2} 对比 ===\n\n"

        for sector_name in [sector1, sector2]:
            etf_code = SECTOR_ETFS[sector_name]
            resp = s.get(
                f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={etf_code},day,,,60,qfq",
                timeout=10
            )
            data = resp.json()
            klines = data.get("data", {}).get(etf_code, {}).get("day", [])
            if not klines:
                klines = data.get("data", {}).get(etf_code, {}).get("qfqday", [])

            if not klines or len(klines) < 20:
                result += f"{sector_name}: 数据不足\n"
                continue

            closes = [float(k[2]) for k in klines]
            current = closes[-1]

            pct_5d = (closes[-1] / closes[-5] - 1) * 100 if len(closes) >= 5 else 0
            pct_10d = (closes[-1] / closes[-10] - 1) * 100 if len(closes) >= 10 else 0
            pct_20d = (closes[-1] / closes[-20] - 1) * 100 if len(closes) >= 20 else 0
            pct_60d = (closes[-1] / closes[0] - 1) * 100

            # 波动率
            returns = [(closes[i]/closes[i-1]-1)*100 for i in range(1, len(closes))]
            volatility = (sum(r**2 for r in returns) / len(returns)) ** 0.5

            # RSI (标准 Wilder)
            rsi = _q.compute_rsi(closes)
            rsi = rsi if rsi is not None else 50

            result += f"【{sector_name}】\n"
            result += f"  5日: {pct_5d:+.2f}% | 10日: {pct_10d:+.2f}% | 20日: {pct_20d:+.2f}% | 60日: {pct_60d:+.2f}%\n"
            result += f"  波动率: {volatility:.2f}% | RSI: {rsi:.1f}\n\n"

        return result
    except Exception as e:
        return f"板块对比失败: {e}"
