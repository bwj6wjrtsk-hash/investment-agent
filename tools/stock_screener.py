"""
AI 选股 & 估值分析工具
- PE/PB 分位数分析
- 低估值筛选
- AI 综合评分
"""
import os
import re
import json
import requests
from langchain_core.tools import tool

from tools import quote as _q


def _tencent_get_batch(codes: list) -> dict:
    """Batch fetch from Tencent"""
    code_str = ",".join(codes)
    with requests.Session() as session:
        session.trust_env = False
        resp = session.get(f"https://qt.gtimg.cn/q={code_str}", timeout=10)
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


@tool
def analyze_valuation(symbol: str) -> str:
    """分析股票/ETF的估值水平（PE/PB分位数分析）。
    参数:
    - symbol: 股票/ETF代码
    """
    try:
        prefix = _q.code_prefix(symbol)

        # 获取当前数据
        data = _tencent_get_batch([prefix])
        if prefix not in data:
            return f"未找到 {symbol}"

        fields = data[prefix]
        name = fields[1]
        price = float(fields[3]) if fields[3] else 0
        pe = float(fields[39]) if fields[39] else None
        pb = float(fields[46]) if len(fields) > 46 and fields[46] else None

        result = f"=== {name}({symbol}) 估值分析 ===\n\n"
        result += f"当前价格: {price}\n"

        if pe:
            result += f"市盈率(PE): {pe:.2f}\n"
            # PE 估值评级
            if pe < 0:
                result += "  → 亏损，PE无参考意义\n"
            elif pe < 10:
                result += "  → 极低估值 ⭐⭐⭐\n"
            elif pe < 15:
                result += "  → 低估值 ⭐⭐\n"
            elif pe < 25:
                result += "  → 合理估值 ⭐\n"
            elif pe < 40:
                result += "  → 偏高估值 ⚠️\n"
            else:
                result += "  → 高估值 ⚠️⚠️\n"

        if pb:
            result += f"市净率(PB): {pb:.2f}\n"
            if pb < 0.8:
                result += "  → 破净，极低估值 ⭐⭐⭐\n"
            elif pb < 1.0:
                result += "  → 接近净资产，低估 ⭐⭐\n"
            elif pb < 2.0:
                result += "  → 合理 ⭐\n"
            elif pb < 5.0:
                result += "  → 偏高 ⚠️\n"
            else:
                result += "  → 高估 ⚠️⚠️\n"

        # 获取历史数据计算位置
        with requests.Session() as session:
            session.trust_env = False
            resp = session.get(
                f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={prefix},day,,,250,qfq",
                timeout=10
            )
        kdata = resp.json()
        klines = kdata.get("data", {}).get(prefix, {}).get("day", [])
        if not klines:
            klines = kdata.get("data", {}).get(prefix, {}).get("qfqday", [])

        if klines and len(klines) >= 60:
            closes = [float(k[2]) for k in klines]
            high_250 = max(closes)
            low_250 = min(closes)
            position_250 = (price - low_250) / (high_250 - low_250) * 100 if high_250 != low_250 else 50

            result += f"\n【价格位置分析】\n"
            result += f"  250日最高: {high_250:.3f}\n"
            result += f"  250日最低: {low_250:.3f}\n"
            result += f"  当前分位: {position_250:.1f}%\n"

            if position_250 < 20:
                result += "  → 处于低位区间，可能被低估 ⭐⭐⭐\n"
            elif position_250 < 40:
                result += "  → 偏低位置，估值合理偏低 ⭐⭐\n"
            elif position_250 < 60:
                result += "  → 中间位置，估值中性\n"
            elif position_250 < 80:
                result += "  → 偏高位置，注意风险 ⚠️\n"
            else:
                result += "  → 高位区间，估值偏高 ⚠️⚠️\n"

            # 均线偏离
            ma20 = sum(closes[-20:]) / 20
            ma60 = sum(closes[-60:]) / 60
            deviation_20 = (price / ma20 - 1) * 100
            deviation_60 = (price / ma60 - 1) * 100
            result += f"\n【均线偏离】\n"
            result += f"  偏离MA20: {deviation_20:+.2f}%\n"
            result += f"  偏离MA60: {deviation_60:+.2f}%\n"

        # 综合评分
        score = 50  # 基准分
        if pe and pe > 0:
            if pe < 10:
                score += 20
            elif pe < 15:
                score += 10
            elif pe > 40:
                score -= 15
        if pb:
            if pb < 1.0:
                score += 15
            elif pb < 1.5:
                score += 5
            elif pb > 5:
                score -= 10
        if klines and len(klines) >= 60:
            if position_250 < 30:
                score += 15
            elif position_250 > 70:
                score -= 10

        score = max(0, min(100, score))
        result += f"\n【综合估值评分】: {score}/100"
        if score >= 75:
            result += " → 低估，值得关注 🟢\n"
        elif score >= 50:
            result += " → 合理估值 ⚪\n"
        else:
            result += " → 偏高估值，谨慎 🔴\n"

        return result
    except Exception as e:
        return f"估值分析失败: {e}"


@tool
def screen_undervalued_stocks(sector: str = "银行") -> str:
    """筛选板块内低估值股票。
    参数:
    - sector: 板块名称，如 '银行', '证券', '医药' 等
    """
    try:
        from sector_config import SECTOR_STOCKS

        if sector not in SECTOR_STOCKS:
            return f"不支持的板块: {sector}。可选: {', '.join(SECTOR_STOCKS.keys())}"

        codes = SECTOR_STOCKS[sector]
        data = _tencent_get_batch(codes)

        stocks = []
        for code in codes:
            if code not in data:
                continue
            fields = data[code]
            if len(fields) < 47:
                continue

            pe = float(fields[39]) if fields[39] else None
            pb = float(fields[46]) if fields[46] else None
            price = float(fields[3]) if fields[3] else 0
            change = float(fields[32]) if fields[32] else 0
            name = fields[1]
            stock_code = fields[2]

            if pe and pe > 0 and pb and pb > 0:
                # 简单估值评分
                score = 0
                if pe < 8:
                    score += 30
                elif pe < 12:
                    score += 20
                elif pe < 20:
                    score += 10
                if pb < 0.8:
                    score += 30
                elif pb < 1.0:
                    score += 20
                elif pb < 1.5:
                    score += 10

                stocks.append({
                    "code": stock_code,
                    "name": name,
                    "price": price,
                    "pe": pe,
                    "pb": pb,
                    "change": change,
                    "score": score,
                })

        stocks.sort(key=lambda x: x["score"], reverse=True)

        result = f"=== {sector}板块低估值筛选 ===\n\n"
        result += f"{'名称':<8} {'代码':<8} {'PE':<8} {'PB':<8} {'现价':<8} {'今涨跌':<8} {'评分'}\n"
        result += "-" * 60 + "\n"

        for s in stocks:
            star = "⭐" if s["score"] >= 40 else ""
            result += f"{s['name']:<8} {s['code']:<8} {s['pe']:<8.1f} {s['pb']:<8.2f} {s['price']:<8.2f} {s['change']:>+5.2f}%  {s['score']}{star}\n"

        if stocks:
            top = stocks[0]
            result += f"\n💡 最低估: {top['name']}({top['code']}) PE={top['pe']:.1f} PB={top['pb']:.2f}"

        return result
    except Exception as e:
        return f"筛选失败: {e}"


@tool
def ai_stock_analysis(symbol: str) -> str:
    """使用AI对个股进行综合分析（基本面+技术面+消息面）。
    参数:
    - symbol: 股票代码
    """
    try:
        prefix = _q.code_prefix(symbol)

        # 1. 获取基本面数据
        data = _tencent_get_batch([prefix])
        if prefix not in data:
            return f"未找到 {symbol}"
        fields = data[prefix]
        name = fields[1]
        price = float(fields[3]) if fields[3] else 0
        pe = float(fields[39]) if fields[39] else 0
        pb = float(fields[46]) if len(fields) > 46 and fields[46] else 0
        change_pct = float(fields[32]) if fields[32] else 0
        turnover = float(fields[38]) if fields[38] else 0

        # 2. 获取技术面数据
        with requests.Session() as session:
            session.trust_env = False
            resp = session.get(
                f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={prefix},day,,,60,qfq",
                timeout=10
            )
        kdata = resp.json()
        klines = kdata.get("data", {}).get(prefix, {}).get("day", [])
        if not klines:
            klines = kdata.get("data", {}).get(prefix, {}).get("qfqday", [])

        technical_info = ""
        if klines and len(klines) >= 20:
            closes = [float(k[2]) for k in klines]
            ma5 = sum(closes[-5:]) / 5
            ma20 = sum(closes[-20:]) / 20
            high_60 = max(closes)
            low_60 = min(closes)
            position = (price - low_60) / (high_60 - low_60) * 100 if high_60 != low_60 else 50

            # RSI (标准 Wilder)
            rsi = _q.compute_rsi(closes)
            rsi = rsi if rsi is not None else 50

            technical_info = f"MA5={ma5:.3f}, MA20={ma20:.3f}, RSI={rsi:.1f}, 60日位置={position:.0f}%"

        # 3. 获取相关新闻
        news = []
        try:
            with requests.Session() as session:
                session.trust_env = False
                resp = session.get(
                    "https://feed.mix.sina.com.cn/api/roll/get",
                    params={"pageid": "153", "lid": "2516", "num": "30", "page": "1"},
                    timeout=5
                )
            if resp.status_code == 200:
                news_data = resp.json()
                if news_data.get("result") and news_data["result"].get("data"):
                    for item in news_data["result"]["data"]:
                        title = item.get("title", "")
                        if name[:2] in title or symbol in title:
                            news.append(title)
                            if len(news) >= 3:
                                break
        except:
            pass

        # 4. 调用 AI 综合分析
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key or api_key == "your_api_key_here":
            # 没有API key，返回纯数据分析
            result = f"=== {name}({symbol}) 综合分析 ===\n\n"
            result += f"价格: {price} ({change_pct:+.2f}%)\n"
            result += f"PE: {pe} | PB: {pb} | 换手: {turnover}%\n"
            result += f"技术面: {technical_info}\n"
            if news:
                result += f"相关新闻: {'; '.join(news)}\n"
            result += "\n(配置 DEEPSEEK_API_KEY 后可获得AI综合分析)"
            return result

        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model="deepseek-chat",
            api_key=api_key,
            base_url="https://api.deepseek.com",
            temperature=0.3,
        )

        prompt = f"""请对以下股票进行简短的投资分析（200字以内）：

股票: {name}({symbol})
当前价格: {price}, 今日涨跌: {change_pct:+.2f}%
PE: {pe}, PB: {pb}, 换手率: {turnover}%
技术面: {technical_info}
相关新闻: {'; '.join(news) if news else '无'}

请按以下格式回答：
📊 估值判断: (低估/合理/高估，一句话原因)
📈 技术面: (趋势方向+关键支撑阻力)
📰 消息面: (利好/利空/中性)
💡 操作建议: (买入/持有/卖出/观望)
⚠️ 风险: (主要风险点)"""

        resp_ai = llm.invoke(prompt)
        result = f"=== {name}({symbol}) AI综合分析 ===\n\n"
        result += resp_ai.content
        return result
    except Exception as e:
        return f"AI分析失败: {e}"
