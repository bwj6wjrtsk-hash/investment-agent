"""
Market data tools - using Tencent Finance API (reliable)
"""
import re
import requests
from langchain_core.tools import tool

from tools import quote as _q


def _tencent_get(codes: list) -> dict:
    """Fetch data from Tencent Finance API"""
    s = requests.Session()
    s.trust_env = False
    code_str = ",".join(codes)
    resp = s.get(f"https://qt.gtimg.cn/q={code_str}", timeout=10)
    resp.encoding = "gbk"

    results = {}
    for line in resp.text.strip().split(";"):
        line = line.strip()
        if not line:
            continue
        match = re.match(r'v_(\w+)="(.+)"', line)
        if match:
            code = match.group(1)
            fields = match.group(2).split("~")
            if len(fields) > 40:
                results[code] = fields
    return results


def _code_prefix(symbol: str) -> str:
    """Add exchange prefix to stock code (含北交所，统一走 quote 层)"""
    return _q.code_prefix(symbol)


@tool
def get_stock_realtime(symbol: str) -> str:
    """获取单只股票/ETF的实时行情。输入股票代码，如 '000001' 或 '515290'。"""
    try:
        code = _code_prefix(symbol)
        data = _tencent_get([code])
        if code not in data:
            return f"未找到股票代码 {symbol}"

        f = data[code]
        return (
            f"股票: {f[1]} ({symbol})\n"
            f"最新价: {f[3]}\n"
            f"涨跌幅: {f[32]}%\n"
            f"涨跌额: {f[31]}\n"
            f"成交量: {f[6]}手\n"
            f"成交额: {f[37]}万元\n"
            f"今开: {f[5]} | 昨收: {f[4]}\n"
            f"最高: {f[33]} | 最低: {f[34]}\n"
            f"换手率: {f[38]}%\n"
            f"市盈率: {f[39]}"
        )
    except Exception as e:
        return f"获取行情失败: {e}"


@tool
def get_stock_history(symbol: str, days: int = 30) -> str:
    """获取股票/ETF历史行情数据。
    参数:
    - symbol: 股票代码，如 '515290'
    - days: 获取最近多少天的数据，默认30天
    """
    try:
        s = requests.Session()
        s.trust_env = False
        code = _code_prefix(symbol)

        resp = s.get(
            f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={code},day,,,{days},qfq",
            timeout=10
        )
        data = resp.json()
        klines = data.get("data", {}).get(code, {}).get("day", [])
        if not klines:
            klines = data.get("data", {}).get(code, {}).get("qfqday", [])

        if not klines:
            return f"未获取到 {symbol} 的历史数据"

        summary = f"股票 {symbol} 最近 {len(klines)} 个交易日数据:\n"
        closes = [float(k[2]) for k in klines]
        highs = [float(k[3]) for k in klines]
        lows = [float(k[4]) for k in klines]

        summary += f"期间最高价: {max(highs)}\n"
        summary += f"期间最低价: {min(lows)}\n"
        summary += f"期间均价: {sum(closes)/len(closes):.3f}\n"
        summary += f"\n最近5个交易日:\n"
        for k in klines[-5:]:
            summary += f"  {k[0]} | 开:{k[1]} 收:{k[2]} 高:{k[3]} 低:{k[4]} 量:{k[5]}\n"
        return summary
    except Exception as e:
        return f"获取历史数据失败: {e}"


@tool
def get_stock_technical_indicators(symbol: str) -> str:
    """计算股票/ETF的技术指标（MA均线、MACD、RSI）。输入股票代码。"""
    try:
        s = requests.Session()
        s.trust_env = False
        code = _code_prefix(symbol)

        resp = s.get(
            f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={code},day,,,60,qfq",
            timeout=10
        )
        data = resp.json()
        klines = data.get("data", {}).get(code, {}).get("day", [])
        if not klines:
            klines = data.get("data", {}).get(code, {}).get("qfqday", [])

        if not klines or len(klines) < 20:
            return f"数据不足，无法计算技术指标"

        closes = [float(k[2]) for k in klines]
        current_price = closes[-1]

        # MA
        ma5 = sum(closes[-5:]) / 5
        ma10 = sum(closes[-10:]) / 10
        ma20 = sum(closes[-20:]) / 20

        # RSI(14) - 标准 Wilder
        rsi = _q.compute_rsi(closes, 14)
        # MACD - 标准 DIF/DEA/柱
        dif, dea, hist = _q.compute_macd(closes)

        result = f"股票 {symbol} 技术指标:\n"
        result += f"当前价格: {current_price}\n\n"
        result += f"【均线】\n"
        result += f"  MA5:  {ma5:.3f} {'↑' if current_price > ma5 else '↓'}\n"
        result += f"  MA10: {ma10:.3f} {'↑' if current_price > ma10 else '↓'}\n"
        result += f"  MA20: {ma20:.3f} {'↑' if current_price > ma20 else '↓'}\n"
        result += f"  趋势: {'MA5>MA20 多头排列' if ma5 > ma20 else 'MA5<MA20 空头排列'}\n"

        if rsi is not None:
            result += f"\n【RSI(14)】: {rsi:.1f}"
            if rsi > 70:
                result += " ⚠️ 超买\n"
            elif rsi < 30:
                result += " ⚠️ 超卖\n"
            else:
                result += " 正常\n"
        else:
            result += "\n【RSI(14)】: 数据不足\n"

        if dif is not None:
            cross = "金叉(多头)" if dif > dea else "死叉(空头)"
            result += f"\n【MACD】 DIF: {dif:.4f} | DEA: {dea:.4f} | 柱: {hist:.4f} {cross}\n"
        else:
            result += "\n【MACD】 数据不足\n"

        return result
    except Exception as e:
        return f"计算技术指标失败: {e}"


@tool
def get_market_overview() -> str:
    """获取A股市场整体概况，包括大盘指数和涨跌统计。"""
    try:
        data = _tencent_get(["sh000001", "sz399001", "sz399006"])

        result = "=== A股市场概况 ===\n\n【主要指数】\n"
        names = {"sh000001": "上证指数", "sz399001": "深证成指", "sz399006": "创业板指"}
        for code, name in names.items():
            if code in data:
                f = data[code]
                result += f"  {name}: {f[3]} ({f[32]}%)\n"

        # 真实涨跌家数（拿不到就如实标注，不再按比例编造）
        breadth = _q.get_market_breadth()
        if breadth.get("up") is not None:
            up, down, flat = breadth["up"], breadth["down"], breadth["flat"]
            result += f"\n【涨跌统计】\n  上涨: {up} | 下跌: {down} | 平盘: {flat}\n"
        else:
            result += "\n【涨跌统计】\n  暂无法获取实时涨跌家数\n"

        return result
    except Exception as e:
        return f"获取市场概况失败: {e}"


@tool
def get_stock_fundamental(symbol: str) -> str:
    """获取股票基本面数据。输入股票代码。"""
    try:
        code = _code_prefix(symbol)
        data = _tencent_get([code])
        if code not in data:
            return f"未找到 {symbol}"

        f = data[code]
        result = f"=== {f[1]}({symbol}) 基本面 ===\n"
        result += f"最新价: {f[3]}\n"
        result += f"市盈率(PE): {f[39]}\n"
        result += f"市净率(PB): {f[46] if len(f) > 46 else 'N/A'}\n"
        result += f"总市值: {f[45]}亿\n" if len(f) > 45 and f[45] else ""
        result += f"流通市值: {f[44]}亿\n" if len(f) > 44 and f[44] else ""
        result += f"换手率: {f[38]}%\n"
        result += f"52周最高: {f[48]}\n" if len(f) > 48 and f[48] else ""
        result += f"52周最低: {f[49]}\n" if len(f) > 49 and f[49] else ""
        return result
    except Exception as e:
        return f"获取基本面失败: {e}"
