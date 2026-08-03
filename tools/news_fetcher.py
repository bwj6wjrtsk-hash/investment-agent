"""
News fetcher tools - using Sina Finance API (reliable)
"""
import requests
from langchain_core.tools import tool


@tool
def get_financial_news(count: int = 10) -> str:
    """获取最新财经新闻。参数 count 为获取条数，默认10条。"""
    try:
        with requests.Session() as session:
            session.trust_env = False
            resp = session.get(
                "https://feed.mix.sina.com.cn/api/roll/get",
                params={"pageid": "153", "lid": "2516", "num": str(count), "page": "1"},
                timeout=10
            )
        if resp.status_code != 200:
            return "获取新闻失败"

        data = resp.json()
        if not data.get("result") or not data["result"].get("data"):
            return "暂无新闻"

        result = "=== 最新财经新闻 ===\n\n"
        for i, item in enumerate(data["result"]["data"][:count], 1):
            title = item.get("title", "")
            ctime = item.get("ctime", "")
            if title:
                result += f"{i}. {title}\n"
                if ctime:
                    result += f"   时间: {ctime}\n"
                result += "\n"
        return result
    except Exception as e:
        return f"获取新闻失败: {e}"


@tool
def get_stock_news(symbol: str, count: int = 5) -> str:
    """获取个股/板块相关新闻。输入关键词如股票名称或代码。"""
    try:
        with requests.Session() as session:
            session.trust_env = False

            # Search for related news
            resp = session.get(
                "https://feed.mix.sina.com.cn/api/roll/get",
                params={"pageid": "153", "lid": "2516", "num": "30", "page": "1"},
                timeout=10
            )
        if resp.status_code != 200:
            return f"获取 {symbol} 新闻失败"

        data = resp.json()
        if not data.get("result") or not data["result"].get("data"):
            return f"暂无 {symbol} 相关新闻"

        result = f"=== {symbol} 相关新闻 ===\n\n"
        found = 0
        for item in data["result"]["data"]:
            title = item.get("title", "")
            if symbol in title or "银行" in title or "金融" in title:
                found += 1
                result += f"{found}. {title}\n\n"
                if found >= count:
                    break

        if found == 0:
            result += "暂无直接相关新闻\n"
        return result
    except Exception as e:
        return f"获取新闻失败: {e}"


@tool
def get_sector_fund_flow() -> str:
    """获取行业板块资金流向/涨跌情况。"""
    try:
        import json

        # Get multiple sectors
        sectors = {
            "new_jrhy": "金融",
            "new_dzxx": "电子信息",
            "new_yysw": "医药生物",
            "new_fdc": "房地产",
            "new_dlhy": "电力",
            "new_qcgy": "汽车",
            "new_spyl": "食品饮料",
            "new_dqhy": "电气",
            "new_jzjc": "建筑建材",
            "new_cmyl": "传媒娱乐",
        }

        result = "=== 行业板块涨跌 ===\n\n"
        sector_data = []

        for node, name in sectors.items():
            try:
                with requests.Session() as session:
                    session.trust_env = False
                    resp = session.get(
                        "https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/Market_Center.getHQNodeData",
                        params={"page": "1", "num": "3", "sort": "changepercent", "asc": "0", "node": node},
                        timeout=5
                    )
                if resp.status_code == 200 and resp.text.strip() != "[]":
                    items = json.loads(resp.text)
                    if items:
                        avg = sum(float(it.get("changepercent", 0)) for it in items) / len(items)
                        sector_data.append({"name": name, "change": avg})
            except:
                continue

        sector_data.sort(key=lambda x: x["change"], reverse=True)
        for s in sector_data:
            emoji = "📈" if s["change"] > 0 else "📉"
            result += f"  {emoji} {s['name']}: {s['change']:+.2f}%\n"

        return result
    except Exception as e:
        return f"获取板块数据失败: {e}"


@tool
def get_hot_stocks() -> str:
    """获取今日涨幅最大的股票。"""
    try:
        import json

        with requests.Session() as session:
            session.trust_env = False
            resp = session.get(
                "https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/Market_Center.getHQNodeData",
                params={"page": "1", "num": "15", "sort": "changepercent", "asc": "0", "node": "hs_a"},
                timeout=10
            )
        if resp.status_code != 200:
            return "获取热门股票失败"

        items = json.loads(resp.text)
        result = "=== 今日涨幅榜 TOP15 ===\n\n"
        for i, item in enumerate(items[:15], 1):
            name = item.get("name", "")
            code = item.get("symbol", "")
            change = item.get("changepercent", 0)
            price = item.get("trade", 0)
            result += f"{i:>2}. {name}({code}) 现价:{price} 涨幅:{change}%\n"

        return result
    except Exception as e:
        return f"获取热门股票失败: {e}"
