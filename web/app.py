"""
DataBoard - Web Interface
"""
import os
import sys
import re
import math
import threading
import time
import uuid
from collections import OrderedDict
from datetime import datetime

# Fix proxy issues
for key in list(os.environ.keys()):
    if 'proxy' in key.lower():
        del os.environ[key]

import requests
import urllib3
urllib3.disable_warnings()

from flask import Flask, render_template, request, jsonify, g

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

from tools import quote as _quote

app = Flask(__name__)


class _BoundedTTLCache:
    """进程内线程安全的有界 TTL/LRU 缓存。"""

    def __init__(self, maxsize, ttl):
        self.maxsize = maxsize
        self.ttl = ttl
        self._items = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key):
        now = time.monotonic()
        with self._lock:
            item = self._items.get(key)
            if item is None:
                return None
            if now - item[0] > self.ttl:
                self._items.pop(key, None)
                return None
            self._items.move_to_end(key)
            return item[1]

    def set(self, key, value):
        now = time.monotonic()
        with self._lock:
            self._items[key] = (now, value)
            self._items.move_to_end(key)
            expired = [k for k, (created, _) in self._items.items()
                       if now - created > self.ttl]
            for expired_key in expired:
                self._items.pop(expired_key, None)
            while len(self._items) > self.maxsize:
                self._items.popitem(last=False)

    def __len__(self):
        with self._lock:
            return len(self._items)


_agent_executor = None
_agent_lock = threading.Lock()
_news_pool_cache = _BoundedTTLCache(maxsize=1, ttl=120)
_news_fetch_lock = threading.Lock()
_news_sentiment_cache = _BoundedTTLCache(maxsize=256, ttl=6 * 3600)
_sector_analysis_cache = _BoundedTTLCache(maxsize=256, ttl=6 * 3600)
_analysis_llm = None
_analysis_llm_lock = threading.Lock()
_ai_invoke_lock = threading.Lock()


def _get_analysis_llm():
    """复用低温度分析模型客户端，避免每个请求重复构造。"""
    global _analysis_llm
    if _analysis_llm is None:
        with _analysis_llm_lock:
            if _analysis_llm is None:
                from langchain_openai import ChatOpenAI
                _analysis_llm = ChatOpenAI(
                    model="deepseek-chat",
                    api_key=os.getenv("DEEPSEEK_API_KEY"),
                    base_url="https://api.deepseek.com",
                    temperature=0.1,
                )
    return _analysis_llm

# 可选的写接口鉴权：在 .env 配置 DASHBOARD_TOKEN 后，所有写操作(POST)需带
# 请求头 X-Auth-Token 或 ?token= 才放行。未配置则不强制（本地自用），
# 但服务默认只绑定 127.0.0.1（见 main.py / __main__）。
_DASHBOARD_TOKEN = os.getenv("DASHBOARD_TOKEN", "").strip()

# 允许匿名访问的 POST 接口（对话/搜索等读性质操作）
_AUTH_EXEMPT_POST = {"/api/chat", "/api/knowledge/search"}
_DATA_WRITE_LOCK = threading.RLock()


@app.before_request
def _check_write_auth():
    if not _DASHBOARD_TOKEN:
        return  # 未配置 token，不强制（依赖仅绑定本机）
    if request.method != "POST":
        return
    if request.path in _AUTH_EXEMPT_POST:
        return
    token = request.headers.get("X-Auth-Token") or request.args.get("token", "")
    if token != _DASHBOARD_TOKEN:
        return jsonify({"error": "unauthorized"}), 401


@app.before_request
def _serialize_data_writes():
    """串行化本进程内的数据写请求，避免 JSON 读改写互相覆盖。"""
    if request.method == "POST" and request.path not in _AUTH_EXEMPT_POST:
        _DATA_WRITE_LOCK.acquire()
        g.data_write_lock_acquired = True


@app.teardown_request
def _release_data_write_lock(_error=None):
    if getattr(g, "data_write_lock_acquired", False):
        g.data_write_lock_acquired = False
        _DATA_WRITE_LOCK.release()


def get_agent():
    global _agent_executor
    if _agent_executor is None:
        with _agent_lock:
            if _agent_executor is None:
                from agents import create_investment_agent
                _agent_executor = create_investment_agent()
    return _agent_executor


def _tencent_request(codes: list) -> dict:
    """Fetch Tencent quotes through the shared short-lived cache."""
    return _quote.get_quotes(codes)


def _parse_index(code, fields):
    """Parse and verify an index quote from Tencent's ``~`` separated format."""
    required_index = 32
    if len(fields) <= required_index:
        raise ValueError(f"{code} 行情字段不完整")

    try:
        price = float(fields[3])
        prev_close = float(fields[4])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{code} 行情价格无效") from exc
    if price <= 0 or prev_close <= 0:
        raise ValueError(f"{code} 行情价格无效")

    # 使用现价和昨收确定性计算，避免上游涨跌字段缺失或口径漂移。
    change_amt = price - prev_close
    change_pct = change_amt / prev_close * 100
    raw_time = str(fields[30] or "").strip()
    try:
        as_of = datetime.strptime(raw_time, "%Y%m%d%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        as_of = raw_time

    return {
        "code": code,
        "name": fields[1],
        "price": price,
        "prev_close": prev_close,
        "change_amt": change_amt,
        "change_pct": change_pct,
        "as_of": as_of,
        "source": "腾讯行情",
    }


def _parse_stock(fields):
    """Parse stock data from Tencent format"""
    return {
        "code": fields[2],
        "name": fields[1],
        "price": float(fields[3]) if fields[3] else 0,
        "prev_close": float(fields[4]) if fields[4] else 0,
        "open": float(fields[5]) if fields[5] else 0,
        "volume": float(fields[6]) if fields[6] else 0,
        "change_pct": float(fields[32]) if fields[32] else 0,
        "change_amt": float(fields[31]) if fields[31] else 0,
        "high": float(fields[33]) if fields[33] else 0,
        "low": float(fields[34]) if fields[34] else 0,
        "amount": float(fields[37]) if fields[37] else 0,
        "turnover": float(fields[38]) if fields[38] else 0,
        "pe": float(fields[39]) if fields[39] else None,
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json(silent=True) or {}
    user_message = data.get("message", "").strip()
    if not user_message:
        return jsonify({"error": "Message cannot be empty"}), 400
    try:
        from agents import chat
        agent = get_agent()
        response = chat(agent, user_message)
        return jsonify({"response": response, "timestamp": datetime.now().strftime("%H:%M:%S")})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/market-overview", methods=["GET"])
def api_market_overview():
    """首屏只等待完整指数行情；涨跌家数由独立接口渐进刷新。"""
    try:
        index_codes = ["sh000001", "sz399001", "sz399006"]
        data = _tencent_request(index_codes)
        missing = [code for code in index_codes if code not in data]
        if missing:
            return jsonify({
                "error": "指数实时行情不完整，请稍后重试",
                "missing_codes": missing,
            }), 502

        indices = [_parse_index(code, data[code]) for code in index_codes]
        breadth = _quote.get_cached_market_breadth()
        return jsonify({
            "indices": indices,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "stats": {
                "up": breadth.get("up"),
                "down": breadth.get("down"),
                "flat": breadth.get("flat"),
                "limit_up": None,
                "limit_down": None,
            },
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 502
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/market-breadth", methods=["GET"])
def api_market_breadth():
    """独立加载涨跌家数，避免慢上游阻塞指数和整个首屏。"""
    try:
        return jsonify({"stats": _quote.get_market_breadth()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/stock/<symbol>", methods=["GET"])
def api_stock_info(symbol):
    if not _quote.is_valid_symbol(symbol):
        return jsonify({"error": "Invalid stock symbol"}), 400
    try:
        code = _quote.code_prefix(symbol)
        data = _tencent_request([code])
        if code not in data:
            return jsonify({"error": f"Stock {symbol} not found"}), 404
        
        stock = _parse_stock(data[code])
        return jsonify(stock)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/hot-stocks", methods=["GET"])
def api_hot_stocks():
    """按成交额返回热门标的；上游失败时短缓存空结果，避免页面反复报错。"""
    import time as time_mod
    now = time_mod.time()
    cached = getattr(api_hot_stocks, "_cache", None)
    if cached and now - cached[0] <= cached[1]:
        return jsonify(cached[2])

    params = {
        "pn": "1", "pz": "15", "po": "1", "np": "1",
        "fltt": "2", "invt": "2", "fid": "f6",
        "fs": "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23,m:0+t:81+s:2048",
        "fields": "f2,f3,f6,f12,f14",
    }
    response = None
    try:
        with requests.Session() as session:
            session.trust_env = False
            for attempt in range(2):
                try:
                    response = session.get(
                        "https://push2.eastmoney.com/api/qt/clist/get",
                        params=params,
                        timeout=8,
                        headers={"User-Agent": "Mozilla/5.0", "Referer": "https://quote.eastmoney.com/"},
                    )
                    if response.status_code == 200:
                        break
                except requests.RequestException:
                    if attempt == 0:
                        time_mod.sleep(0.2)

        stocks = []
        if response is not None and response.status_code == 200:
            items = (response.json().get("data") or {}).get("diff") or []
            for item in items:
                code = item.get("f12", "")
                name = item.get("f14", "")
                price = item.get("f2", 0)
                change_pct = item.get("f3", 0)
                if code and name and price and price != "-":
                    stocks.append({
                        "code": code, "name": name,
                        "price": float(price),
                        "change_pct": float(change_pct) if change_pct != "-" else 0,
                    })
        payload = {"stocks": stocks}
        cache_ttl = 60 if stocks else 15
        if not stocks:
            payload["warning"] = "热门标的上游暂不可用"
        api_hot_stocks._cache = (time_mod.time(), cache_ttl, payload)
        return jsonify(payload)
    except Exception:
        payload = {"stocks": [], "warning": "热门标的上游暂不可用"}
        api_hot_stocks._cache = (time_mod.time(), 15, payload)
        return jsonify(payload)


@app.route("/api/sector-flow", methods=["GET"])
def api_sector_flow():
    try:
        from tools.sector_rotation import _get_sector_prices, SECTOR_ETFS

        prices = _get_sector_prices()
        sectors = []
        for sector_name, etf_code in SECTOR_ETFS.items():
            if etf_code in prices:
                data = prices[etf_code]
                sectors.append({
                    "name": sector_name,
                    "change_pct": data["change_pct"],
                    "flow": data.get("amount", 0),  # 成交额作为资金流向指标
                })

        # Sort by change_pct descending
        sectors.sort(key=lambda x: x["change_pct"], reverse=True)

        # Show top 5 gainers + top 5 losers
        if len(sectors) > 10:
            top5 = sectors[:5]
            bottom5 = sectors[-5:]
            sectors = top5 + bottom5

        return jsonify({"sectors": sectors})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# 基金名称里对新闻匹配无意义的词（后缀、基金公司名等），提取行业关键词时剔除
_NEWS_STOP_WORDS = [
    "ETF", "指数", "基金", "联接", "LOF", "QDII", "分级",
    "南方", "天弘", "易方达", "华夏", "嘉实", "广发", "富国", "招商",
    "国泰", "博时", "汇添富", "中欧", "华宝", "鹏华", "工银", "银华",
]


# 全市场重磅关键词：命中的新闻即使不在持仓名单，也会送 AI 判级
_HOT_KEYWORDS = [
    "降息", "加息", "降准", "LPR", "MLF", "印花税",
    "立案", "调查", "处罚", "退市", "停牌", "复牌", "重组", "并购", "破产", "违约", "暴雷",
    "涨停", "跌停", "熔断", "增持", "减持", "回购",
    "业绩预增", "业绩预亏", "预增", "预亏", "扭亏",
    "关税", "制裁", "出口管制", "战争", "冲突",
    "注册制", "史上", "创纪录", "暴涨", "暴跌", "大涨", "大跌",
]

# 明显的非财经噪音词（政治/军事/娱乐等），命中且无财经特征时过滤
_NOISE_KEYWORDS = [
    "防长", "导弹", "真主党", "以军", "总统", "外长", "议会", "大选",
    "明星", "电影", "综艺", "球星", "球队", "世界杯", "演唱会", "娱乐圈",
    "枪击", "难民",
]
_FINANCE_HINTS = [
    "股", "市", "板块", "指数", "基金", "券", "银行", "经济", "亿元", "万元",
    "涨", "跌", "净利", "营收", "业绩", "央行", "美元", "人民币", "A股", "港股",
    "美股", "IPO", "公司", "产业", "投资", "融资", "利率", "债", "期货", "黄金",
    "楼市", "房价", "消费", "出口", "GDP", "CPI", "关税",
]

# 含行业词但并非该行业的常见干扰短语，匹配前先剔除，降低误命中
_MATCH_NOISE_PHRASES = [
    "世界银行", "亚洲开发银行", "开发银行", "投资银行", "银行委员会",
    "银行家", "美联储", "欧洲央行",
    # 境外央行/利率新闻对A股板块相关度低，剔除避免误标为持仓相关
    "韩国央行", "日本央行", "美联储", "欧洲央行", "英国央行", "韩元", "日元加息",
]


def _watchlist_keywords():
    """从持仓 + 自选提取用于新闻匹配的关键词。
    个股用全名精确匹配；ETF/指数借助 sector_config 用板块精选关键词，匹配更准。
    返回 {关键词: 展示名称}"""
    import json as json_mod
    items = []
    base = os.path.dirname(os.path.dirname(__file__))
    for fname in ("portfolio.json", "watchlist.json"):
        fpath = os.path.join(base, fname)
        if os.path.exists(fpath):
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    items.extend(json_mod.load(f))
            except Exception:
                pass

    try:
        from sector_config import ETF_SECTOR_MAP, STOCK_SECTOR_HINTS, SECTOR_KEYWORDS
    except Exception:
        ETF_SECTOR_MAP, STOCK_SECTOR_HINTS, SECTOR_KEYWORDS = {}, {}, {}

    name_hints = {
        "银行": "银行", "医药": "医药", "半导体": "半导体", "芯片": "半导体",
        "新能源": "新能源", "锂电": "新能源", "光伏": "新能源", "证券": "证券",
        "券商": "证券", "军工": "军工", "消费": "消费", "白酒": "消费",
        "汽车": "汽车", "电力": "电力", "地产": "房地产", "中车": "机械", "高铁": "机械",
    }

    keywords = {}  # keyword -> 展示名称
    for it in items:
        name = (it.get("name") or "").strip()
        code = (it.get("code") or "").strip()
        if not name:
            continue
        is_fund = ("ETF" in name) or ("指数" in name) or ("基金" in name)
        # 个股：用全名精确匹配
        if not is_fund and len(name) >= 2:
            keywords[name] = name
        # 本地判定板块（不联网，避免拖慢请求）
        sector = ETF_SECTOR_MAP.get(code) or STOCK_SECTOR_HINTS.get(code) or ""
        if not sector:
            for kw, sec in name_hints.items():
                if kw in name:
                    sector = sec
                    break
        # 板块精选关键词：仅对 ETF/指数使用（个股用全名即可，避免板块泛词误命中）
        if is_fund:
            # 过于宽泛、境外新闻也高频出现的词，排除以降低误匹配
            too_broad = {"利率", "央行", "AI", "科技", "消费", "汽车", "电力"}
            for kw in SECTOR_KEYWORDS.get(sector, []):
                if len(kw) >= 2 and kw not in too_broad:
                    keywords.setdefault(kw, name)
            # 兜底：基金去后缀的核心词
            core = name
            for w in _NEWS_STOP_WORDS:
                core = core.replace(w, "")
            core = core.strip()
            if len(core) >= 2:
                keywords.setdefault(core, name)
    return keywords


def _is_finance_news(title, summary):
    """过滤明显与财经无关的噪音新闻"""
    text = title + summary
    if any(w in text for w in _NOISE_KEYWORDS) and not any(w in text for w in _FINANCE_HINTS):
        return False
    return True


def _relative_time(ts):
    """把时间戳转成相对时间：X分钟前 / X小时前 / 昨天 / MM-DD HH:MM"""
    if not ts:
        return ""
    import time as _t
    diff = max(0, int(_t.time()) - ts)
    if diff < 3600:
        return f"{max(1, diff // 60)}分钟前"
    if diff < 86400:
        return f"{diff // 3600}小时前"
    if diff < 172800:
        return "昨天"
    return datetime.fromtimestamp(ts).strftime("%m-%d %H:%M")


_PUSHED_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "news_pushed.json")


def _push_strong_news(strong_items):
    """把新出现的重磅新闻推送到手机（Server酱/企业微信），当天去重避免重复推送"""
    if not strong_items:
        return
    import json as json_mod
    from datetime import date
    today = date.today().isoformat()

    pushed = {"date": today, "titles": []}
    try:
        if os.path.exists(_PUSHED_FILE):
            with open(_PUSHED_FILE, "r", encoding="utf-8") as f:
                data = json_mod.load(f)
            if data.get("date") == today:
                pushed = data
    except Exception:
        pass

    seen = set(pushed["titles"])
    new_items = [n for n in strong_items if n["title"] not in seen]
    if not new_items:
        return

    # 持仓相关的重磅优先展示
    new_items.sort(key=lambda n: (not n.get("related"), -(n.get("ts") or 0)))
    lines = []
    for n in new_items:
        tag = f"【持仓·{'/'.join(n['matched'])}】" if n.get("related") else "【市场】"
        reason = f"（{n['reason']}）" if n.get("reason") else ""
        lines.append(f"- {tag} **{n['sentiment']}** {n['title']}{reason}")
    content = "\n".join(lines)
    n_hold = sum(1 for n in new_items if n.get("related"))
    title = f"⚠ 重磅提醒 {len(new_items)} 条" + (f"（持仓相关 {n_hold}）" if n_hold else "")

    try:
        _send_notification(title, content)
    except Exception:
        pass

    # 无论推送渠道是否配置，都记录，避免下次重复尝试同一批
    pushed["titles"] = list(seen | {n["title"] for n in new_items})
    try:
        os.makedirs(os.path.dirname(_PUSHED_FILE), exist_ok=True)
        with open(_PUSHED_FILE, "w", encoding="utf-8") as f:
            json_mod.dump(pushed, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _classify_news(title, summary):
    """根据标题/摘要做简单分类"""
    text = title + summary
    rules = [
        ("美股", ["美股", "纳斯达克", "道琼斯", "标普", "美联储", "华尔街", "美元", "鲍威尔"]),
        ("港股", ["港股", "恒生", "香港"]),
        ("宏观", ["央行", "GDP", "CPI", "PPI", "通胀", "降息", "加息", "货币政策", "经济数据", "国务院", "发改委"]),
        ("公司", ["公告", "业绩", "净利润", "营收", "财报", "IPO", "增持", "减持", "股东", "预增", "预亏"]),
        ("A股", ["A股", "沪指", "深成", "创业板", "科创", "北向", "涨停", "跌停", "两市", "上证"]),
    ]
    for label, kws in rules:
        if any(k in text for k in kws):
            return label
    return "综合"


def _norm_img(v):
    """图片字段统一成字符串（新浪返回 str，东财返回 list）"""
    if isinstance(v, list):
        return v[0] if v and isinstance(v[0], str) else ""
    return v if isinstance(v, str) else ""


def _fetch_sina_news(session):
    """新浪财经滚动新闻"""
    out = []
    for page in range(1, 6):  # 扩大采集池，增加持仓相关命中概率
        try:
            resp = session.get(
                "https://feed.mix.sina.com.cn/api/roll/get",
                params={"pageid": "153", "lid": "2516", "num": "50", "page": str(page)},
                timeout=8,
            )
            if resp.status_code != 200:
                continue
            items = (resp.json().get("result") or {}).get("data") or []
            for item in items:
                title = (item.get("title") or "").strip()
                if not title:
                    continue
                ts = 0
                try:
                    ts = int(item.get("ctime", 0))
                except (ValueError, TypeError):
                    pass
                summary = (item.get("intro") or item.get("summary") or "").strip()
                out.append({
                    "title": title,
                    "summary": summary[:120],
                    "ts": ts,
                    "url": item.get("url", ""),
                    "source": item.get("media_name", "") or "新浪财经",
                    "img": _norm_img(item.get("img")),
                })
        except Exception:
            continue
    return out


def _fetch_eastmoney_news(session):
    """东方财富 7x24 快讯"""
    out = []
    try:
        resp = session.get(
            "https://np-weblist.eastmoney.com/comm/web/getFastNewsList",
            params={"client": "web", "biz": "web_724", "fastColumn": "102",
                    "sortEnd": "", "pageSize": "80", "req_trace": "1"},
            timeout=8,
            headers={"Referer": "https://kuaixun.eastmoney.com/"},
        )
        if resp.status_code == 200:
            items = (resp.json().get("data") or {}).get("fastNewsList") or []
            for item in items:
                title = (item.get("title") or "").strip()
                summary = (item.get("summary") or "").strip()
                if not title and not summary:
                    continue
                if not title:
                    title = summary[:40]
                ts = 0
                show_time = item.get("showTime", "")
                if show_time:
                    try:
                        ts = int(datetime.strptime(show_time, "%Y-%m-%d %H:%M:%S").timestamp())
                    except ValueError:
                        pass
                code = (item.get("code") or "").strip()
                url = f"https://finance.eastmoney.com/a/{code}.html" if code else "https://kuaixun.eastmoney.com/"
                out.append({
                    "title": title,
                    "summary": summary[:120],
                    "ts": ts,
                    "url": url,
                    "source": "东方财富",
                    "img": _norm_img(item.get("image")),
                    "stockList": item.get("stockList") or [],
                })
    except Exception:
        pass
    return out


def _dedup_news(news_list):
    """用哈希索引缩小候选集，再做精确相似度判断，避免全量 O(n²) 比较。"""
    from difflib import SequenceMatcher

    kept = []  # [(norm, item, stock_set, ts)]
    result = []
    exact_titles = set()
    prefix16 = set()
    buckets = {}

    for news in news_list:
        norm = re.sub(r"[\s\W]+", "", news["title"])
        if not norm or norm in exact_titles:
            continue
        if len(norm) >= 16 and norm[:16] in prefix16:
            continue

        stock_set = {str(code) for code in (news.get("stockList") or [])}
        timestamp = news.get("ts") or 0
        # 同源改写通常保持标题尾部，前16字相同已在上方 O(1) 处理；
        # 首尾联合签名避免财经标题常见前缀造成超大候选桶。
        keys = {("edge", norm[:8], norm[-8:])}
        if len(norm) >= 12:
            keys.add(("tail", norm[-12:]))
        keys.update(("stock", code) for code in stock_set)
        candidate_indexes = set()
        for key in keys:
            candidate_indexes.update(buckets.get(key, ()))

        duplicate = False
        for index in candidate_indexes:
            other_norm, other, other_stocks, other_ts = kept[index]
            short, long = ((norm, other_norm) if len(norm) <= len(other_norm)
                           else (other_norm, norm))
            if len(short) >= 12 and short in long:
                duplicate = True
                break
            if (len(norm) >= 12 and len(other_norm) >= 12 and
                    SequenceMatcher(None, norm, other_norm).ratio() >= 0.82):
                duplicate = True
                break
            if (stock_set and other_stocks and stock_set & other_stocks and
                    abs(timestamp - other_ts) <= 21600):
                duplicate = True
                break

        if duplicate:
            continue

        index = len(kept)
        kept.append((norm, news, stock_set, timestamp))
        result.append(news)
        exact_titles.add(norm)
        if len(norm) >= 16:
            prefix16.add(norm[:16])
        for key in keys:
            buckets.setdefault(key, []).append(index)

    return result


def _clone_news_pool(pool):
    """请求会补充展示字段，返回浅拷贝避免并发请求修改缓存原件。"""
    return [dict(item, stockList=list(item.get("stockList") or [])) for item in pool]


def _get_news_pool():
    """共享两分钟新闻池，并将并发缓存未命中合并为一次上游抓取。"""
    cached = _news_pool_cache.get("all")
    if cached is not None:
        return _clone_news_pool(cached)

    with _news_fetch_lock:
        cached = _news_pool_cache.get("all")
        if cached is None:
            with requests.Session() as session:
                session.trust_env = False
                fetched = _fetch_sina_news(session) + _fetch_eastmoney_news(session)
            cached = _dedup_news(fetched)
            _news_pool_cache.set("all", cached)
    return _clone_news_pool(cached)


def _analyze_news_sentiment(news_items):
    """批量判断新闻影响；结果有界缓存，同一进程内合并并发模型调用。"""
    if not news_items:
        return

    def apply_result(news, result):
        news["sentiment"] = result.get("s")
        news["reason"] = result.get("r", "")
        if news.get("matched") and result.get("rel") == 0:
            news["related"] = False
            news["matched"] = []
            news["ai_rejected"] = True

    import hashlib
    cache_key = hashlib.md5("".join(n["title"] for n in news_items).encode()).hexdigest()[:10]
    cached = _news_sentiment_cache.get(cache_key)
    if cached is not None:
        for news, result in zip(news_items, cached):
            apply_result(news, result)
        return

    try:
        import json as json_mod
        with _ai_invoke_lock:
            # 等待期间相同请求可能已经完成，再检查一次避免重复计费。
            cached = _news_sentiment_cache.get(cache_key)
            if cached is None:
                lines = []
                for index, news in enumerate(news_items):
                    line = f"{index + 1}. {news['title']}"
                    if news.get("summary"):
                        line += f"（{news['summary'][:50]}）"
                    if news.get("matched"):
                        line += f" [关联标的:{'/'.join(news['matched'])}]"
                    lines.append(line)
                prompt = (
                    "你是A股投资分析助手。对下列每条新闻给出三项：\n"
                    "s：影响等级，只能从“非常利好”“利好”“中性”“利空”“非常利空”五档选一，"
                    "只有影响重大、可能引发明显异动才用“非常”；\n"
                    "r：不超过20字的理由；\n"
                    "rel：仅当该条标注了“关联标的”时判断——新闻内容确实关系到该标的/板块基本面填1，"
                    "只是文字上恰好出现该词、实质无关填0；没有关联标的的填1。\n"
                    '严格只输出JSON数组，格式：[{"i":1,"s":"利好","r":"降准利好银行息差","rel":1}]，'
                    "不要多余文字。\n\n新闻：\n" + "\n".join(lines)
                )
                content = _get_analysis_llm().invoke(prompt).content.strip()
                match = re.search(r"\[.*\]", content, re.DOTALL)
                if not match:
                    return
                parsed = json_mod.loads(match.group(0))
                cached = [{} for _ in news_items]
                for item in parsed:
                    index = int(item.get("i", 0)) - 1
                    if 0 <= index < len(news_items):
                        cached[index] = {
                            "s": item.get("s"), "r": item.get("r", ""),
                            "rel": item.get("rel", 1),
                        }
                _news_sentiment_cache.set(cache_key, cached)

        for news, result in zip(news_items, cached):
            apply_result(news, result)
    except Exception:
        pass  # 余额不足或无 key，静默跳过


@app.route("/api/daily-news", methods=["GET"])
def api_daily_news():
    """每日财经新闻：多源聚合 + 持仓联动 + 摘要/分类 + AI 利好利空判断"""
    try:
        count = int(request.args.get("count", 40))
        count = max(1, min(count, 80))
        with_ai = request.args.get("ai", "0") == "1"

        import time as _time
        recent_hours = int(request.args.get("hours", 48))
        cutoff = int(_time.time()) - recent_hours * 3600

        # 共享两分钟新闻池；先做时效/财经过滤，再进入后续分类。
        news_list = _get_news_pool()
        news_list = [
            news for news in news_list
            if ((not news["ts"]) or news["ts"] >= cutoff)
            and _is_finance_news(news["title"], news["summary"])
        ]

        # 5) 分类 + 持仓精准联动 + 重磅标记 + 时间
        keywords = _watchlist_keywords()
        for n in news_list:
            title_clean = n["title"]
            full_clean = n["title"] + n["summary"]
            for p in _MATCH_NOISE_PHRASES:
                title_clean = title_clean.replace(p, "")
                full_clean = full_clean.replace(p, "")
            matched = set()
            for kw, name in keywords.items():
                # 2字以内的中文泛词只在标题里匹配，避免摘要中零散词误命中
                is_short = len(kw) <= 2 and all("\u4e00" <= c <= "\u9fff" for c in kw)
                hay = title_clean if is_short else full_clean
                if kw in hay:
                    matched.add(name)
            matched = sorted(matched)
            n["matched"] = matched
            n["related"] = bool(matched)
            n["category"] = _classify_news(n["title"], n["summary"])
            n["hot"] = any(k in full_clean for k in _HOT_KEYWORDS)
            n["sentiment"] = None
            n["reason"] = ""
            n["time"] = datetime.fromtimestamp(n["ts"]).strftime("%m-%d %H:%M") if n["ts"] else ""
            n["ago"] = _relative_time(n["ts"])

        # 6) 排序：相关 > 重磅 > 时间
        news_list.sort(key=lambda x: (x["related"], x["hot"], x["ts"]), reverse=True)
        news_list = news_list[:count]

        # 7) AI 判级：持仓相关 + 全市场疑似重磅，合并去重后判断（控制成本）
        if with_ai:
            # 持仓相关与全市场重磅各留配额，避免相关新闻挤占、漏判重磅
            related_c = [n for n in news_list if n["related"]]
            hot_c = [n for n in news_list if n["hot"] and not n["related"]]
            candidates = []
            seen_t = set()
            for n in related_c[:8] + hot_c[:8]:
                if n["title"] not in seen_t:
                    candidates.append(n)
                    seen_t.add(n["title"])
            _analyze_news_sentiment(candidates[:14])
            # AI 精排可能改变 related 标记，重新排序保证展示顺序正确
            news_list.sort(key=lambda x: (x["related"], x["hot"], x["ts"]), reverse=True)

        # 8) 持仓情绪概览：只统计“与持仓/自选相关”的已判级新闻，才代表“对你的影响”
        counts = {"非常利好": 0, "利好": 0, "中性": 0, "利空": 0, "非常利空": 0}
        for n in news_list:
            if n["related"] and n["sentiment"] in counts:
                counts[n["sentiment"]] += 1
        graded = sum(counts.values())
        bull = counts["非常利好"] * 2 + counts["利好"]
        bear = counts["非常利空"] * 2 + counts["利空"]
        if graded == 0:
            mood = "无相关"
        else:
            mood = "偏多" if bull > bear else ("偏空" if bear > bull else "中性")

        # 9) 重磅消息推送到手机：仅推“与持仓/自选相关”的重磅，避免全市场噪音打扰
        #    全市场重磅仍保留在页面“市场大事”栏，但不推送到手机
        if with_ai and request.args.get("push", "0") == "1":
            strong = [n for n in news_list
                      if n["related"] and n["sentiment"] in ("非常利好", "非常利空")]
            _push_strong_news(strong)

        return jsonify({
            "news": news_list,
            "related_count": sum(1 for n in news_list if n["related"]),
            "sentiment_summary": counts,
            "mood": mood,
            "updated": datetime.now().strftime("%H:%M:%S"),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/bank-sector", methods=["GET"])
def api_bank_sector():
    """板块行情与相关新闻；共享行情/新闻池，AI 分析仅在显式 ai=1 时启用。"""
    try:
        with_ai = request.args.get("ai", "0") == "1"
        portfolio = _load_portfolio_data()
        watchlist = _load_watchlist()
        all_items = portfolio + watchlist

        from sector_config import detect_sector, get_sector_config

        sectors_seen = set()
        sectors_config = []
        sector_names = {}
        resolved_items = []
        for item in all_items:
            sector_name = item.get("sector") or detect_sector(item["code"], item.get("name", ""))
            resolved_items.append((item, sector_name))
            name = (item.get("name") or "").strip()
            if (sector_name and name and "ETF" not in name and "指数" not in name and
                    "基金" not in name and len(name) >= 2):
                sector_names.setdefault(sector_name, set()).add(name)

        for item, sector_name in resolved_items:
            if not sector_name or sector_name in sectors_seen:
                continue
            sectors_seen.add(sector_name)
            if item.get("sector_stocks") and item.get("sector_keywords"):
                config = {
                    "name": sector_name,
                    "stocks": item["sector_stocks"],
                    "keywords": list(item["sector_keywords"]),
                }
            else:
                config = get_sector_config(sector_name)
                config["keywords"] = list(config.get("keywords", []))
            for name in sector_names.get(sector_name, set()):
                if name not in config["keywords"]:
                    config["keywords"].append(name)
            sectors_config.append(config)

        all_codes = sorted({code for sector in sectors_config for code in sector.get("stocks", [])})

        # 行情和新闻来源互不依赖，并行获取；新闻池跨接口共享并带 single-flight。
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=2) as executor:
            quotes_future = executor.submit(_quote.get_quotes, all_codes)
            news_future = executor.submit(_get_news_pool)
            quotes = quotes_future.result()
            news_pool = news_future.result()

        all_sectors_data = []
        for sector in sectors_config:
            sector_data = {
                "name": sector["name"], "sector": {"change_pct": 0},
                "key_stocks": [], "news": [], "analysis": "",
            }

            key_stocks = []
            total_change = 0.0
            for code in sector.get("stocks", []):
                fields = quotes.get(_quote.code_prefix(code))
                if not fields or len(fields) <= _quote.F_CHANGE_PCT:
                    continue
                try:
                    change = float(fields[_quote.F_CHANGE_PCT] or 0)
                    key_stocks.append({
                        "name": fields[_quote.F_NAME],
                        "code": fields[_quote.F_CODE],
                        "price": float(fields[_quote.F_PRICE] or 0),
                        "change_pct": change,
                    })
                    total_change += change
                except (TypeError, ValueError):
                    continue
            sector_data["sector"] = {
                "change_pct": total_change / len(key_stocks) if key_stocks else 0,
            }
            key_stocks.sort(key=lambda row: row["change_pct"], reverse=True)
            sector_data["key_stocks"] = key_stocks

            keywords = sector.get("keywords", [])
            matched = [
                item for item in news_pool
                if keywords and any(keyword in item["title"] + item.get("summary", "")
                                    for keyword in keywords)
            ]
            matched.sort(key=lambda row: row.get("ts", 0), reverse=True)
            news = []
            for item in matched[:5]:
                timestamp = item.get("ts", 0)
                time_text = datetime.fromtimestamp(timestamp).strftime("%m-%d %H:%M") if timestamp else ""
                entry = f"[{time_text}] {item['title']}" if time_text else item["title"]
                if entry not in news:
                    news.append(entry)
            sector_data["news"] = news

            if not news:
                sector_data["analysis"] = "暂无该板块相关新闻，无法判断消息面"
            elif not with_ai:
                change = sector_data["sector"]["change_pct"]
                state = "偏强" if change >= 1 else ("偏弱" if change <= -1 else "震荡")
                sector_data["analysis"] = (
                    f"板块当前{state}（{change:+.2f}%），已匹配{len(news)}条相关新闻；"
                    "未启用AI方向判断"
                )
            else:
                import hashlib
                news_hash = hashlib.md5("".join(news).encode()).hexdigest()[:8]
                cache_key = f"{sector['name']}_{news_hash}"
                cached_analysis = _sector_analysis_cache.get(cache_key)
                if cached_analysis is not None:
                    sector_data["analysis"] = cached_analysis
                else:
                    try:
                        with _ai_invoke_lock:
                            cached_analysis = _sector_analysis_cache.get(cache_key)
                            if cached_analysis is None:
                                prompt = (
                                    f"根据以下新闻，判断对{sector['name']}板块的影响。用1句话总结，"
                                    '格式：“利好/利空/中性，因为xxx”\n\n新闻：\n' +
                                    "\n".join(news) + "\n\n结论："
                                )
                                cached_analysis = _get_analysis_llm().invoke(prompt).content.strip()
                                _sector_analysis_cache.set(cache_key, cached_analysis)
                        sector_data["analysis"] = cached_analysis
                    except Exception as error:
                        if "Insufficient Balance" in str(error):
                            sector_data["analysis"] = "API余额不足"

            all_sectors_data.append(sector_data)

        if all_sectors_data:
            result = dict(all_sectors_data[0])
            result["all_sectors"] = all_sectors_data
        else:
            result = {
                "sector": {"change_pct": 0}, "key_stocks": [], "news": [],
                "analysis": "", "all_sectors": [],
            }
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/portfolio", methods=["GET"])
def api_portfolio():
    from agents import RiskMonitor
    monitor = RiskMonitor()
    summary = monitor.get_portfolio_summary()
    # 首页不展示本接口的 alerts；默认跳过昂贵的整份自选预警计算。
    alerts = monitor.check_all() if request.args.get("include_alerts") == "1" else []
    return jsonify({"summary": summary, "alerts": alerts})


@app.route("/api/profit-summary", methods=["GET"])
def api_profit_summary():
    """Get realized + unrealized profit summary"""
    try:
        # Realized profit (from sell trades)
        trades = _load_trades()
        realized = sum(t.get("profit", 0) for t in trades if t.get("action") == "sell" and "profit" in t)

        # Unrealized profit (current positions); share the 15-second quote cache
        # with portfolio/watchlist/signals instead of issuing another raw request.
        portfolio = _load_portfolio_data()
        unrealized = 0
        if portfolio:
            symbols = [_market_symbol(item["code"], item.get("market")) for item in portfolio]
            quotes = _quote.get_quotes(symbols)
            for item, symbol in zip(portfolio, symbols):
                fields = quotes.get(symbol)
                price = _fields_info(fields).get("price", 0) if fields else 0
                if price > 0:
                    unrealized += (price - item["cost"]) * item["shares"]

        total = realized + unrealized
        return jsonify({
            "realized": round(realized, 2),
            "unrealized": round(unrealized, 2),
            "total": round(total, 2),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/signals", methods=["GET"])
def api_signals():
    """Get trading signals for all watchlist + portfolio stocks"""
    from agents import RiskMonitor
    monitor = RiskMonitor()
    # get_signals() already processes ALL watchlist stocks (both held and not held)
    signals = monitor.get_signals()

    return jsonify({"signals": signals})


@app.route("/api/portfolio/list", methods=["GET"])
def api_portfolio_list():
    """Get raw portfolio data for editing"""
    import json as json_mod
    portfolio_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "portfolio.json")
    with open(portfolio_file, "r", encoding="utf-8") as f:
        data = json_mod.load(f)
    return jsonify({"portfolio": data})


@app.route("/api/portfolio/add", methods=["POST"])
def api_portfolio_add():
    """新增持仓等价于新增一笔买入交易，保持交易账本为唯一数量/成本来源。"""
    data = request.get_json(silent=True) or {}
    code = str(data.get("code") or "").strip()
    if not re.fullmatch(r"\d{6}", code):
        return jsonify({"error": "code 必须是6位数字"}), 400
    try:
        cost = float(data.get("cost"))
        shares = int(data.get("shares"))
        stop_loss = float(data.get("stop_loss", cost * 0.9))
        take_profit = float(data.get("take_profit", cost * 1.3))
    except (TypeError, ValueError, OverflowError):
        return jsonify({"error": "成本、数量、止盈止损必须是数字"}), 400
    if not all(math.isfinite(value) for value in (cost, stop_loss, take_profit)) or min(
            cost, shares, stop_loss, take_profit) <= 0:
        return jsonify({"error": "成本、数量、止盈止损必须为有限正数"}), 400

    portfolio = _load_portfolio_data()
    if any(item.get("code") == code for item in portfolio):
        return jsonify({"error": f"{code} already exists"}), 400

    name = str(data.get("name") or "").strip()
    if not name:
        try:
            symbol = _market_symbol(code, data.get("market"))
            fields = _quote.get_quotes([symbol]).get(symbol)
            name = _fields_info(fields).get("name", "") if fields else ""
        except Exception:
            name = ""
    name = name or code
    trade = {
        "id": uuid.uuid4().hex,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "code": code,
        "name": name,
        "action": "buy",
        "price": cost,
        "shares": shares,
        "amount": round(cost * shares, 2),
    }
    metadata = portfolio + [{
        "code": code, "name": name, "stop_loss": stop_loss,
        "take_profit": take_profit,
        "first_buy_date": str(data.get("first_buy_date") or datetime.now().date().isoformat()),
    }]
    try:
        trades, rebuilt = _commit_trade_state(_load_trades() + [trade], metadata)
    except LedgerConflictError as exc:
        return jsonify({"error": str(exc)}), 409
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    new_item = next(item for item in rebuilt if item.get("code") == code)
    return jsonify({"success": True, "item": new_item})


@app.route("/api/portfolio/update", methods=["POST"])
def api_portfolio_update():
    """更新止盈止损；账本管理的成本和数量必须通过交易记录调整。"""
    data = request.get_json(silent=True) or {}
    code = str(data.get("code") or "").strip()

    if not code:
        return jsonify({"error": "code is required"}), 400

    portfolio = _load_portfolio_data()
    item = next((entry for entry in portfolio if entry.get("code") == code), None)
    if not item:
        return jsonify({"error": f"{code} not found"}), 404

    try:
        requested_cost = float(data.get("cost", item["cost"]))
        requested_shares = int(data.get("shares", item["shares"]))
        if not math.isfinite(requested_cost) or requested_cost <= 0 or requested_shares <= 0:
            raise ValueError
        if requested_cost != float(item["cost"]) or requested_shares != int(item["shares"]):
            return jsonify({"error": "持仓成本和数量由交易账本维护，请通过新增或编辑交易调整"}), 409
        for key in ("stop_loss", "take_profit"):
            if key in data:
                value = float(data[key])
                if not math.isfinite(value) or value <= 0:
                    raise ValueError
                item[key] = value
    except (TypeError, ValueError, OverflowError):
        return jsonify({"error": "成本、数量、止盈止损必须是有限正数"}), 400

    _save_json_file(PORTFOLIO_FILE, portfolio)
    return jsonify({"success": True})


@app.route("/api/portfolio/delete", methods=["POST"])
def api_portfolio_delete():
    """仅允许删除无交易账本来源的孤立持仓，避免之后被历史交易复活。"""
    data = request.get_json(silent=True) or {}
    code = str(data.get("code") or "").strip()
    if not code:
        return jsonify({"error": "code is required"}), 400

    portfolio = _load_portfolio_data()
    if not any(item.get("code") == code for item in portfolio):
        return jsonify({"error": f"{code} not found"}), 404
    if any(item.get("code") == code for item in _load_json_file(
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "trades.json"), [])):
        return jsonify({"error": "该持仓由交易账本维护，请新增卖出交易完成清仓"}), 409

    _save_json_file(PORTFOLIO_FILE, [item for item in portfolio if item.get("code") != code])
    return jsonify({"success": True})


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
WATCHLIST_FILE = os.path.join(BASE_DIR, "watchlist.json")
PORTFOLIO_FILE = os.path.join(BASE_DIR, "portfolio.json")
EVENTS_FILE = os.path.join(BASE_DIR, "data", "events.json")
ALERT_STATES_FILE = os.path.join(BASE_DIR, "data", "alert_states.json")


def _load_json_file(path, default):
    import json as json_mod
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json_mod.load(f)
    except (OSError, ValueError, TypeError):
        pass
    return default


def _save_json_file(path, data):
    import json as json_mod
    import tempfile
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=".tmp-", dir=os.path.dirname(path) or ".", text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json_mod.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def _load_watchlist():
    return _load_json_file(WATCHLIST_FILE, [])


def _save_watchlist(data):
    _save_json_file(WATCHLIST_FILE, data)


def _load_portfolio_data():
    return _load_json_file(PORTFOLIO_FILE, [])


def _valid_symbol(code):
    return bool(re.fullmatch(r"\d{5,6}", (code or "").strip()))


def _market_symbol(code, market=None):
    """返回带市场前缀的唯一行情代码；显式市场优先，解决跨市场同代码冲突。"""
    market = (market or "").strip().lower()
    if market in ("sh", "sz", "bj", "hk"):
        return market + code
    return _quote.code_prefix(code)


def _watch_symbol(item):
    return _market_symbol(item.get("code", ""), item.get("market"))


def _optional_float(value):
    if value in (None, ""):
        return None
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("数值必须是有限数字")
    return number


def _fields_info(fields):
    def num(index, default=0.0):
        try:
            return float(fields[index]) if len(fields) > index and fields[index] else default
        except (TypeError, ValueError):
            return default
    return {
        "code": fields[2] if len(fields) > 2 else "",
        "name": fields[1] if len(fields) > 1 else "",
        "price": num(3), "prev_close": num(4), "open": num(5),
        "volume": num(6), "change_amt": num(31), "change_pct": num(32),
        "high": num(33), "low": num(34), "amount": num(37),
        "turnover": num(38), "pe": num(39, None), "pb": num(46, None),
    }


@app.route("/api/stock/search", methods=["GET"])
def api_stock_search():
    """按部分证券名称、拼音或代码搜索 A股/港股/ETF 候选。"""
    import json as json_mod
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"results": []})
    if len(query) > 30:
        return jsonify({"error": "搜索内容过长"}), 400
    try:
        with requests.Session() as session:
            session.trust_env = False
            resp = session.get(
                "https://smartbox.gtimg.cn/s3/",
                params={"q": query, "t": "all"},
                timeout=8,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            resp.encoding = "gbk"
            response_text = resp.text
        match = re.search(r'v_hint="(.*)"', response_text.strip())
        if not match:
            return jsonify({"results": []})
        # 接口中的中文使用 JSON unicode 转义，借助 json 安全还原。
        payload = json_mod.loads('"' + match.group(1).replace('"', '\\"') + '"')
        allowed_types = {"GP", "GP-A", "GP-B", "ETF", "LOF", "JJ", "ZS"}
        results, seen = [], set()
        for raw in payload.split("^"):
            fields = raw.split("~")
            if len(fields) < 5:
                continue
            market, code, name, pinyin, security_type = fields[:5]
            if market not in ("sh", "sz", "bj", "hk") or security_type not in allowed_types:
                continue
            identity = f"{market}:{code}"
            if identity in seen or not _valid_symbol(code):
                continue
            seen.add(identity)
            results.append({"code": code, "name": name, "market": market.upper(),
                            "instrument_id": identity, "type": security_type, "pinyin": pinyin})
        q_lower = query.lower()
        results.sort(key=lambda x: (
            0 if x["name"] == query or x["code"] == query else
            1 if x["name"].startswith(query) else
            2 if query in x["name"] else
            3 if x["pinyin"].startswith(q_lower) else 4
        ))
        return jsonify({"results": results[:12]})
    except Exception as e:
        return jsonify({"error": f"证券搜索失败: {e}"}), 502


@app.route("/api/watchlist", methods=["GET"])
def api_watchlist():
    """自选实时行情、客观指标、配置距离和概览；不为列表额外请求K线。"""
    watchlist = [dict(item) for item in _load_watchlist()]
    portfolio = _load_portfolio_data()
    portfolio_symbols = {_watch_symbol(item) for item in portfolio}
    watchlisted_symbols = {_watch_symbol(item) for item in watchlist}
    combined_symbols = set(watchlisted_symbols)
    for item in portfolio:
        symbol = _watch_symbol(item)
        if symbol not in combined_symbols:
            watchlist.append({
                "code": item["code"], "name": item.get("name", item["code"]),
                "market": item.get("market") or symbol[:2],
                "group": "默认", "reason": "", "from_portfolio": True,
            })
            combined_symbols.add(symbol)

    updated_at = datetime.now().isoformat(timespec="seconds")
    if not watchlist:
        return jsonify({
            "stocks": [], "updated_at": updated_at, "source": "腾讯行情",
            "summary": {"total": 0, "available": 0, "up": 0, "down": 0, "flat": 0,
                        "held": 0, "watch": 0, "alert_count": 0, "average_change_pct": None,
                        "strongest": None, "weakest": None},
        })

    symbols = [_watch_symbol(item) for item in watchlist]
    quotes = _quote.get_quotes(symbols)
    stocks = []

    def configured_number(item, key):
        try:
            value = float(item.get(key))
            return value if math.isfinite(value) and value > 0 else None
        except (TypeError, ValueError):
            return None

    for item in watchlist:
        symbol = _watch_symbol(item)
        fields = quotes.get(symbol)
        info = _fields_info(fields) if fields else {}
        price = info.get("price") if fields else None
        prev_close = info.get("prev_close") if fields else None
        quote_available = bool(price and price > 0 and prev_close and prev_close > 0)
        is_portfolio = symbol in portfolio_symbols
        is_watchlisted = symbol in watchlisted_symbols
        portfolio_only = is_portfolio and not is_watchlisted
        watch_group = item.get("group") or "默认"
        target_price = configured_number(item, "target_price")
        stop_loss = configured_number(item, "stop_loss")
        alert_up_pct = configured_number(item, "alert_up_pct")
        alert_down_pct = configured_number(item, "alert_down_pct")
        change_pct = info.get("change_pct") if quote_available else None
        alerts = []
        target_distance_pct = None
        stop_distance_pct = None
        if quote_available:
            if target_price:
                target_distance_pct = (target_price / price - 1) * 100
                if price >= target_price:
                    alerts.append({"kind": "target", "label": "已达到目标价"})
            if stop_loss:
                stop_distance_pct = (price / stop_loss - 1) * 100
                if price <= stop_loss:
                    alerts.append({"kind": "stop", "label": "已跌破止损价"})
            if alert_up_pct and change_pct >= alert_up_pct:
                alerts.append({"kind": "rise", "label": "涨幅达到提醒值"})
            if alert_down_pct and change_pct <= -alert_down_pct:
                alerts.append({"kind": "drop", "label": "跌幅达到提醒值"})

        amplitude_pct = None
        if quote_available and info.get("high") and info.get("low"):
            amplitude_pct = (info["high"] - info["low"]) / prev_close * 100

        stock = dict(item)
        stock.update({
            "name": info.get("name") or item.get("name", item["code"]),
            "market": item.get("market") or symbol[:2],
            "quote_available": quote_available,
            "quote_status": "ok" if quote_available else "unavailable",
            "price": round(price, 4) if quote_available else None,
            "prev_close": round(prev_close, 4) if quote_available else None,
            "open": round(info.get("open", 0), 4) if quote_available else None,
            "high": round(info.get("high", 0), 4) if quote_available else None,
            "low": round(info.get("low", 0), 4) if quote_available else None,
            "change_pct": round(change_pct, 2) if quote_available else None,
            "amplitude_pct": round(amplitude_pct, 2) if amplitude_pct is not None else None,
            "amount": round(info.get("amount", 0), 2) if quote_available else None,
            "turnover": round(info.get("turnover", 0), 2) if quote_available else None,
            "target_distance_pct": round(target_distance_pct, 2) if target_distance_pct is not None else None,
            "stop_distance_pct": round(stop_distance_pct, 2) if stop_distance_pct is not None else None,
            "alerts": alerts,
            "is_portfolio": is_portfolio,
            "is_watchlisted": is_watchlisted,
            "portfolio_only": portfolio_only,
            "from_portfolio": portfolio_only,
            "group": watch_group,
            "display_group": "持仓" if is_portfolio else watch_group,
            "reason": item.get("reason", ""),
            "target_price": target_price,
            "stop_loss": stop_loss,
            "alert_up_pct": alert_up_pct,
            "alert_down_pct": alert_down_pct,
        })
        stocks.append(stock)

    available = [row for row in stocks if row["quote_available"]]
    changes = [row["change_pct"] for row in available]
    strongest = max(available, key=lambda row: row["change_pct"]) if available else None
    weakest = min(available, key=lambda row: row["change_pct"]) if available else None

    def summary_stock(row):
        if not row:
            return None
        return {"code": row["code"], "market": row["market"], "name": row["name"],
                "change_pct": row["change_pct"]}

    summary = {
        "total": len(stocks),
        "available": len(available),
        "up": sum(1 for value in changes if value > 0),
        "down": sum(1 for value in changes if value < 0),
        "flat": sum(1 for value in changes if value == 0),
        "held": sum(1 for row in stocks if row["is_portfolio"]),
        "watch": sum(1 for row in stocks if not row["is_portfolio"]),
        "alert_count": sum(1 for row in stocks if row["alerts"]),
        "average_change_pct": round(sum(changes) / len(changes), 2) if changes else None,
        "strongest": summary_stock(strongest),
        "weakest": summary_stock(weakest),
    }
    return jsonify({
        "stocks": stocks, "summary": summary, "updated_at": updated_at,
        "source": "腾讯行情", "cache_ttl_seconds": _quote.QUOTE_TTL,
    })


@app.route("/api/watchlist/add", methods=["POST"])
def api_watchlist_add():
    data = request.get_json(silent=True) or {}
    code = data.get("code", "").strip()
    if not _valid_symbol(code):
        return jsonify({"error": "代码必须是5位港股或6位A股/ETF数字代码"}), 400
    market = str(data.get("market") or "").strip().lower()
    symbol = _market_symbol(code, market)
    watchlist = _load_watchlist()
    if any(_watch_symbol(item) == symbol for item in watchlist):
        return jsonify({"error": f"{symbol.upper()} already in watchlist"}), 400

    fields = _quote.get_quotes([symbol]).get(symbol)
    info = _fields_info(fields) if fields else {}
    item = {
        "code": code,
        "market": symbol[:2],
        "name": info.get("name") or data.get("name") or code,
        "group": (data.get("group") or "默认").strip()[:20],
        "reason": (data.get("reason") or "").strip()[:200],
        "target_price": _optional_float(data.get("target_price")),
        "stop_loss": _optional_float(data.get("stop_loss")),
        "alert_up_pct": _optional_float(data.get("alert_up_pct")),
        "alert_down_pct": _optional_float(data.get("alert_down_pct")),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    watchlist.append(item)
    _save_watchlist(watchlist)
    return jsonify({"success": True, "name": item["name"], "item": item})


@app.route("/api/watchlist/update", methods=["POST"])
def api_watchlist_update():
    data = request.get_json(silent=True) or {}
    code = str(data.get("code") or "").strip()
    market = str(data.get("market") or "").strip().lower()
    watchlist = _load_watchlist()
    target_symbol = _market_symbol(code, market) if market else None
    candidates = [
        entry for entry in watchlist
        if entry.get("code") == code and (
            not target_symbol or _watch_symbol(entry) == target_symbol
        )
    ]
    if len(candidates) > 1:
        return jsonify({"error": "存在跨市场同代码，请指定 market"}), 400
    item = candidates[0] if candidates else None
    if not item:
        return jsonify({"error": f"{code} not found"}), 404
    for key in ("group", "reason"):
        if key in data:
            item[key] = str(data.get(key) or "").strip()[:200 if key == "reason" else 20]
    try:
        for key in ("target_price", "stop_loss", "alert_up_pct", "alert_down_pct"):
            if key in data:
                item[key] = _optional_float(data.get(key))
    except (TypeError, ValueError):
        return jsonify({"error": "价格和涨跌预警必须是数字"}), 400
    item["updated_at"] = datetime.now().isoformat(timespec="seconds")
    _save_watchlist(watchlist)
    return jsonify({"success": True, "item": item})


@app.route("/api/watchlist/delete", methods=["POST"])
def api_watchlist_delete():
    data = request.get_json(silent=True) or {}
    code = str(data.get("code") or "").strip()
    market = str(data.get("market") or "").strip().lower()
    watchlist = _load_watchlist()
    target_symbol = _market_symbol(code, market) if market else None
    new_list = [
        item for item in watchlist
        if not (item.get("code") == code and (
            not target_symbol or _watch_symbol(item) == target_symbol
        ))
    ]
    if len(new_list) == len(watchlist):
        return jsonify({"error": f"{code} not found"}), 404
    _save_watchlist(new_list)
    return jsonify({"success": True})


@app.route("/api/watchlist/batch", methods=["POST"])
def api_watchlist_batch():
    """一次读写完成全部自选设置，避免并发更新互相覆盖。"""
    data = request.get_json(silent=True) or {}
    operations = data.get("items")
    if not isinstance(operations, list) or not operations:
        return jsonify({"error": "items 必须是非空数组"}), 400
    if len(operations) > 200:
        return jsonify({"error": "单次最多处理 200 项"}), 400

    watchlist = _load_watchlist()
    now = datetime.now().isoformat(timespec="seconds")
    processed = set()
    try:
        for operation in operations:
            if not isinstance(operation, dict):
                raise ValueError("每个自选设置都必须是对象")
            code = str(operation.get("code") or "").strip()
            market = str(operation.get("market") or "").strip().lower()
            if not _valid_symbol(code):
                raise ValueError(f"无效证券代码: {code or '(空)'}")
            target_symbol = _market_symbol(code, market) if market else None
            candidates = [
                item for item in watchlist
                if item.get("code") == code and (
                    not target_symbol or _watch_symbol(item) == target_symbol
                )
            ]
            if len(candidates) != 1:
                reason = "存在跨市场同代码，请指定 market" if len(candidates) > 1 else "未找到"
                raise ValueError(f"{code} {reason}")
            item = candidates[0]
            identity = _watch_symbol(item)
            if identity in processed:
                raise ValueError(f"{code} 重复提交")
            processed.add(identity)

            if operation.get("remove") is True:
                watchlist.remove(item)
                continue
            for key, limit in (("group", 20), ("reason", 200)):
                if key in operation:
                    item[key] = str(operation.get(key) or "").strip()[:limit]
            for key in ("target_price", "stop_loss", "alert_up_pct", "alert_down_pct"):
                if key in operation:
                    item[key] = _optional_float(operation.get(key))
            item["updated_at"] = now
    except (TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400

    _save_watchlist(watchlist)
    return jsonify({"success": True, "updated": len(processed), "stocks": watchlist})


@app.route("/api/stock/<symbol>/detail", methods=["GET"])
def api_stock_detail(symbol):
    """实时行情、K线指标、持仓和自选信息的单标的聚合详情。"""
    if not _valid_symbol(symbol):
        return jsonify({"error": "代码必须是5或6位数字"}), 400
    days = max(30, min(int(request.args.get("days", 120)), 250))
    requested_market = request.args.get("market", "")
    prefix = _market_symbol(symbol, requested_market)
    fields = _quote.get_quotes([prefix]).get(prefix)
    if not fields:
        return jsonify({"error": f"未找到 {symbol} 的行情"}), 404
    stock = _fields_info(fields)
    stock["market"] = {"sh": "沪市", "sz": "深市", "bj": "北交所", "hk": "港股"}.get(prefix[:2], "")

    raw = _quote.get_klines_raw(prefix, days)
    klines = []
    for row in raw:
        if len(row) >= 6:
            try:
                klines.append({
                    "date": row[0], "open": float(row[1]), "close": float(row[2]),
                    "high": float(row[3]), "low": float(row[4]), "volume": float(row[5]),
                })
            except (TypeError, ValueError):
                continue
    closes = [x["close"] for x in klines]
    dif, dea, hist = _quote.compute_macd(closes)
    high_n = max(closes) if closes else None
    low_n = min(closes) if closes else None
    indicators = {
        "rsi14": round(_quote.compute_rsi(closes), 2) if _quote.compute_rsi(closes) is not None else None,
        "ma20": round(sum(closes[-20:]) / min(20, len(closes)), 3) if closes else None,
        "ma60": round(sum(closes[-60:]) / min(60, len(closes)), 3) if closes else None,
        "high": high_n, "low": low_n,
        "drawdown_pct": round((stock["price"] / high_n - 1) * 100, 2) if high_n else None,
        "macd": {"dif": round(dif, 4), "dea": round(dea, 4), "hist": round(hist, 4)} if dif is not None else None,
    }
    holding_item = next((x for x in _load_portfolio_data() if x.get("code") == symbol), None)
    holding = None
    if holding_item:
        holding = dict(holding_item)
        holding["market_value"] = round(stock["price"] * holding_item["shares"], 2)
        holding["pnl"] = round((stock["price"] - holding_item["cost"]) * holding_item["shares"], 2)
        holding["pnl_pct"] = round((stock["price"] / holding_item["cost"] - 1) * 100, 2) if holding_item["cost"] else 0
    watch = next((x for x in _load_watchlist()
                  if x.get("code") == symbol and
                  (not requested_market or _watch_symbol(x)[:2] == requested_market.lower())), None)
    observations = []
    if indicators["rsi14"] is not None:
        observations.append("RSI偏热" if indicators["rsi14"] >= 70 else ("RSI进入超卖区" if indicators["rsi14"] <= 30 else "RSI处于中性区间"))
    if indicators["ma20"]:
        observations.append("价格位于20日均线上方" if stock["price"] >= indicators["ma20"] else "价格位于20日均线下方")
    return jsonify({"stock": stock, "klines": klines, "indicators": indicators,
                    "holding": holding, "watch": watch, "observations": observations})


def _portfolio_health_data():
    portfolio = _load_portfolio_data()
    if not portfolio:
        return {"score": 0, "level": "未配置", "positions": [], "alerts": [],
                "dimensions": {}, "as_of": datetime.now().isoformat(timespec="seconds")}
    quotes = _quote.get_quotes([x["code"] for x in portfolio])
    rows, alerts = [], []
    total_value = 0
    for item in portfolio:
        fields = quotes.get(_quote.code_prefix(item["code"]))
        info = _fields_info(fields) if fields else {}
        price = info.get("price", 0)
        value = price * item["shares"]
        total_value += value
        closes = _quote.get_closes(item["code"], 120) or []
        ma20 = sum(closes[-20:]) / min(20, len(closes)) if closes else None
        high = max(closes) if closes else None
        pnl = (price - item["cost"]) * item["shares"] if price else 0
        pnl_pct = (price / item["cost"] - 1) * 100 if price and item["cost"] else 0
        drawdown = (price / high - 1) * 100 if price and high else None
        score, flags = 100, []
        if not price:
            score, flags = 40, ["行情缺失"]
        else:
            if pnl_pct <= -10: score -= 20; flags.append("浮亏超过10%")
            if drawdown is not None and drawdown <= -15: score -= 15; flags.append("距阶段高点回撤较大")
            if ma20 and price < ma20: score -= 15; flags.append("跌破20日均线")
            if item.get("stop_loss") and price <= item["stop_loss"]: score -= 35; flags.append("触发止损线")
        row = {
            "code": item["code"], "name": item.get("name", item["code"]), "price": price,
            "value": round(value, 2), "pnl": round(pnl, 2), "pnl_pct": round(pnl_pct, 2),
            "today_pnl": round(value * info.get("change_pct", 0) / (100 + info.get("change_pct", 0)), 2) if info.get("change_pct", 0) != -100 else 0,
            "drawdown_120_pct": round(drawdown, 2) if drawdown is not None else None,
            "above_ma20": bool(ma20 and price >= ma20), "health_score": max(0, score), "flags": flags,
        }
        rows.append(row)
        for flag in flags:
            alerts.append({"code": item["code"], "level": "critical" if "止损" in flag else "warning", "message": flag})
    for row in rows:
        row["weight"] = round(row["value"] / total_value * 100, 2) if total_value else 0
    max_weight = max((x["weight"] for x in rows), default=0)
    concentration_penalty = min(30, max(0, max_weight - 40) * 0.6)
    base_score = sum(x["health_score"] for x in rows) / len(rows)
    score = max(0, round(base_score - concentration_penalty))
    level = "健康" if score >= 80 else ("关注" if score >= 60 else "高风险")
    dimensions = {
        "trend": round(sum(1 for x in rows if x["above_ma20"]) / len(rows) * 100),
        "concentration": max(0, round(100 - concentration_penalty * 2)),
        "risk_lines": round(sum(1 for x in rows if not any("止损" in f for f in x["flags"])) / len(rows) * 100),
        "data_quality": round(sum(1 for x in rows if x["price"] > 0) / len(rows) * 100),
    }
    return {"score": score, "level": level, "positions": rows, "alerts": alerts,
            "dimensions": dimensions, "as_of": datetime.now().isoformat(timespec="seconds")}


@app.route("/api/portfolio/health", methods=["GET"])
def api_portfolio_health():
    try:
        return jsonify(_portfolio_health_data())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/portfolio/attribution", methods=["GET"])
def api_portfolio_attribution():
    """按标的汇总已实现与当前未实现盈亏；历史快照不足时明确提示。"""
    try:
        portfolio = _load_portfolio_data()
        trades = _load_trades()
        quotes = _quote.get_quotes([x["code"] for x in portfolio]) if portfolio else {}
        by_code = {}
        for trade in trades:
            code = trade.get("code", "")
            row = by_code.setdefault(code, {"code": code, "name": trade.get("name", code), "realized": 0.0, "unrealized": 0.0})
            if trade.get("action") == "sell":
                row["realized"] += float(trade.get("profit", 0) or 0)
        for item in portfolio:
            row = by_code.setdefault(item["code"], {"code": item["code"], "name": item.get("name", item["code"]), "realized": 0.0, "unrealized": 0.0})
            fields = quotes.get(_quote.code_prefix(item["code"]))
            price = _fields_info(fields).get("price", 0) if fields else 0
            if price:
                row["unrealized"] = (price - item["cost"]) * item["shares"]
        contributions = []
        for row in by_code.values():
            row["realized"] = round(row["realized"], 2)
            row["unrealized"] = round(row["unrealized"], 2)
            row["pnl"] = round(row["realized"] + row["unrealized"], 2)
            contributions.append(row)
        impact = sum(abs(x["pnl"]) for x in contributions)
        for row in contributions:
            row["impact_pct"] = round(abs(row["pnl"]) / impact * 100, 1) if impact else 0
        contributions.sort(key=lambda x: abs(x["pnl"]), reverse=True)
        history = _load_json_file(os.path.join(BASE_DIR, "data", "portfolio_history.json"), [])
        return jsonify({
            "realized": round(sum(x["realized"] for x in contributions), 2),
            "unrealized": round(sum(x["unrealized"] for x in contributions), 2),
            "total_pnl": round(sum(x["pnl"] for x in contributions), 2),
            "contributions": contributions,
            "data_quality": {"snapshot_count": len(history), "warnings": [] if len(history) >= 2 else ["历史快照不足，当前展示累计盈亏归因"]},
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _alert_id(source, code, kind, episode=None):
    """预警 ID 包含触发轮次；风险/行情提醒默认每日重新武装。"""
    import hashlib
    episode = episode or datetime.now().date().isoformat()
    return hashlib.sha1(f"{source}|{code}|{kind}|{episode}".encode("utf-8")).hexdigest()[:14]


def _collect_alerts():
    from agents import RiskMonitor
    alerts = []
    for item in RiskMonitor().check_all():
        if item.get("level") == "info":
            continue
        stock = item.get("stock", "")
        code_match = re.search(r"\((\d{5,6})\)", stock)
        code = code_match.group(1) if code_match else ""
        message = item.get("message", "")
        kind = "stop" if "止损" in message else ("target" if "止盈" in message else ("drop" if "下跌" in message else "rise"))
        alerts.append({
            "id": _alert_id("risk", code, kind), "source": "risk", "level": item.get("level", "warning"),
            "code": code, "title": stock or "风险提醒", "message": message,
            "action": item.get("action", ""), "triggered_at": datetime.now().isoformat(timespec="seconds"),
        })
    watchlist = _load_watchlist()
    symbols = [_watch_symbol(x) for x in watchlist]
    quotes = _quote.get_quotes(symbols) if watchlist else {}
    for item in watchlist:
        fields = quotes.get(_watch_symbol(item))
        if not fields:
            continue
        info = _fields_info(fields)
        checks = [
            ("watch_target", item.get("target_price"), info["price"] >= (item.get("target_price") or float("inf")), "达到目标价"),
            ("watch_stop", item.get("stop_loss"), info["price"] <= (item.get("stop_loss") or 0), "跌破观察止损价"),
            ("watch_up", item.get("alert_up_pct"), info["change_pct"] >= (item.get("alert_up_pct") or float("inf")), "涨幅达到预警值"),
            ("watch_down", item.get("alert_down_pct"), info["change_pct"] <= -(item.get("alert_down_pct") or float("inf")), "跌幅达到预警值"),
        ]
        for kind, threshold, triggered, title in checks:
            if threshold is not None and triggered:
                alerts.append({
                    "id": _alert_id("watch", item["code"], kind), "source": "watch", "level": "warning",
                    "code": item["code"], "title": f"{item.get('name', item['code'])}：{title}",
                    "message": f"现价 {info['price']}，今日涨跌 {info['change_pct']:+.2f}%",
                    "action": item.get("reason") or "请结合投资逻辑复核", "triggered_at": datetime.now().isoformat(timespec="seconds"),
                })
    today = datetime.now().date()
    for event in _load_json_file(EVENTS_FILE, []):
        try:
            days = (datetime.strptime(event["date"], "%Y-%m-%d").date() - today).days
        except (KeyError, ValueError):
            continue
        if 0 <= days <= 3:
            alerts.append({
                "id": _alert_id(
                    "event", event.get("code", ""), event.get("id", event["date"]), event["date"]
                ),
                "source": "event", "level": "info", "code": event.get("code", ""),
                "title": event.get("title", "投资事件"), "message": "今天" if days == 0 else f"还有 {days} 天",
                "action": "提前检查仓位和风险", "triggered_at": datetime.now().isoformat(timespec="seconds"),
            })
    return alerts


@app.route("/api/alerts", methods=["GET"])
def api_alerts():
    try:
        states = _load_json_file(ALERT_STATES_FILE, {})
        items = []
        for alert in _collect_alerts():
            alert["status"] = states.get(alert["id"], {}).get("status", "active")
            if alert["status"] != "dismissed":
                items.append(alert)
        priority = {"critical": 0, "warning": 1, "info": 2}
        items.sort(key=lambda x: priority.get(x["level"], 3))
        return jsonify({"alerts": items, "unread_count": sum(1 for x in items if x["status"] == "active"),
                        "updated": datetime.now().isoformat(timespec="seconds")})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/alerts/action", methods=["POST"])
def api_alert_action():
    data = request.get_json(silent=True) or {}
    alert_id, action = data.get("id", ""), data.get("action", "")
    if not alert_id or action not in ("ack", "dismiss"):
        return jsonify({"error": "invalid alert action"}), 400
    states = _load_json_file(ALERT_STATES_FILE, {})
    states[alert_id] = {"status": "acknowledged" if action == "ack" else "dismissed",
                        "updated_at": datetime.now().isoformat(timespec="seconds")}
    _save_json_file(ALERT_STATES_FILE, states)
    return jsonify({"success": True})


@app.route("/api/events", methods=["GET"])
def api_events():
    from datetime import date, timedelta
    start = request.args.get("from") or date.today().isoformat()
    end = request.args.get("to") or (date.today() + timedelta(days=30)).isoformat()
    items = [x for x in _load_json_file(EVENTS_FILE, []) if start <= x.get("date", "") <= end]
    items.sort(key=lambda x: (x.get("date", ""), x.get("time", "")))
    return jsonify({"events": items, "updated": datetime.now().isoformat(timespec="seconds")})


@app.route("/api/events/add", methods=["POST"])
def api_events_add():
    import uuid
    data = request.get_json(silent=True) or {}
    event_date = str(data.get("date", "")).strip()
    title = str(data.get("title", "")).strip()
    try:
        datetime.strptime(event_date, "%Y-%m-%d")
    except ValueError:
        return jsonify({"error": "日期格式必须为 YYYY-MM-DD"}), 400
    if not title:
        return jsonify({"error": "事件标题不能为空"}), 400
    events = _load_json_file(EVENTS_FILE, [])
    event = {"id": uuid.uuid4().hex[:12], "date": event_date, "time": str(data.get("time", "")).strip(),
             "type": str(data.get("type", "other")), "code": str(data.get("code", "")).strip(),
             "title": title[:100], "note": str(data.get("note", "")).strip()[:300],
             "created_at": datetime.now().isoformat(timespec="seconds")}
    events.append(event)
    _save_json_file(EVENTS_FILE, events)
    return jsonify({"success": True, "event": event})


@app.route("/api/events/delete", methods=["POST"])
def api_events_delete():
    event_id = (request.get_json(silent=True) or {}).get("id", "")
    events = _load_json_file(EVENTS_FILE, [])
    remaining = [x for x in events if x.get("id") != event_id]
    if len(remaining) == len(events):
        return jsonify({"error": "event not found"}), 404
    _save_json_file(EVENTS_FILE, remaining)
    return jsonify({"success": True})


TRADES_FILE = os.path.join(BASE_DIR, "trades.json")
CHAT_HISTORY_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chat_history.json")


def _load_trades():
    return _load_json_file(TRADES_FILE, [])


def _save_trades(trades):
    _save_json_file(TRADES_FILE, trades)


def _save_trade(trade):
    trades = _load_trades()
    trades.append(trade)
    _save_trades(trades)


class LedgerConflictError(ValueError):
    """交易账本与派生持仓不一致，禁止静默覆盖人工数据。"""


def _trade_datetime(value, position):
    text = str(value or "").strip()
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            parsed = datetime.strptime(text, fmt)
            return parsed, parsed.strftime("%Y-%m-%d %H:%M")
        except ValueError:
            continue
    raise ValueError(f"第 {position} 条交易日期格式无效，应为 YYYY-MM-DD HH:MM")


def _portfolio_signature(items):
    """只比较账本派生字段；止盈止损等人工元数据不参与一致性判断。"""
    signature = []
    for item in items or []:
        try:
            signature.append((
                str(item.get("market") or "").lower(),
                str(item.get("code") or "").strip(),
                int(item.get("shares")),
                round(float(item.get("cost")), 4),
            ))
        except (AttributeError, TypeError, ValueError, OverflowError):
            return None
    return sorted(signature)


def _calculate_trade_state(trades, existing_portfolio=None):
    """按交易日期校验账本、重算卖出收益和当前持仓，不允许任何时点超卖。"""
    positions = {}
    normalized = []
    metadata = {item.get("code"): dict(item) for item in (existing_portfolio or [])}
    ordered = []

    for original_index, raw in enumerate(trades):
        if not isinstance(raw, dict):
            raise ValueError(f"第 {original_index + 1} 条交易格式无效")
        parsed, normalized_date = _trade_datetime(raw.get("date"), original_index + 1)
        ordered.append((parsed, original_index, normalized_date, raw))
    ordered.sort(key=lambda entry: (entry[0], entry[1]))

    for index, (_, _original_index, normalized_date, raw) in enumerate(ordered):
        trade = dict(raw)
        code = str(trade.get("code") or "").strip()
        action = str(trade.get("action") or "").strip().lower()
        if not re.fullmatch(r"\d{6}", code):
            raise ValueError(f"第 {index + 1} 条交易代码必须是6位数字")
        if action not in ("buy", "sell"):
            raise ValueError(f"第 {index + 1} 条交易方向无效")
        try:
            price = float(trade.get("price"))
            shares_number = float(trade.get("shares"))
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f"第 {index + 1} 条交易价格和数量必须是数字")
        if (not math.isfinite(price) or not math.isfinite(shares_number) or
                price <= 0 or shares_number <= 0 or not shares_number.is_integer()):
            raise ValueError(f"第 {index + 1} 条交易价格必须为有限正数，数量必须为正整数")
        shares = int(shares_number)
        trade_id = str(trade.get("id") or "").strip()
        if trade_id and not re.fullmatch(r"[A-Za-z0-9_-]{8,64}", trade_id):
            raise ValueError(f"第 {index + 1} 条交易 ID 无效")

        trade.update({
            "id": trade_id or uuid.uuid4().hex,
            "date": normalized_date,
            "code": code,
            "name": str(trade.get("name") or code).strip()[:80] or code,
            "action": action,
            "price": price,
            "shares": shares,
            "amount": round(price * shares, 2),
        })
        trade.pop("profit", None)
        trade.pop("profit_pct", None)
        position = positions.get(code)

        if action == "buy":
            if position:
                total_cost = position["cost"] * position["shares"] + price * shares
                position["shares"] += shares
                position["cost"] = round(total_cost / position["shares"], 4)
            else:
                old_meta = metadata.get(code, {})
                position = {
                    "code": code,
                    "name": trade["name"] or old_meta.get("name") or code,
                    "cost": price,
                    "shares": shares,
                    "stop_loss": old_meta.get("stop_loss", round(price * 0.9, 4)),
                    "take_profit": old_meta.get("take_profit", round(price * 1.3, 4)),
                    "first_buy_date": normalized_date[:10],
                }
                for key in ("market", "notes"):
                    if key in old_meta:
                        position[key] = old_meta[key]
                positions[code] = position
        else:
            if not position:
                raise ValueError(f"第 {index + 1} 条交易超卖：{code} 当前无持仓")
            if shares > position["shares"]:
                raise ValueError(
                    f"第 {index + 1} 条交易超卖：{code} 卖出 {shares}，可用 {position['shares']}"
                )
            cost = position["cost"]
            trade["profit"] = round((price - cost) * shares, 2)
            trade["profit_pct"] = round((price / cost - 1) * 100, 2) if cost else 0
            position["shares"] -= shares
            if position["shares"] == 0:
                del positions[code]
        normalized.append(trade)

    return normalized, list(positions.values())


def _commit_trade_state(trades, metadata=None):
    """校验当前双文件状态后提交；第二次写入失败时尽力回滚，避免静默分叉。"""
    with _DATA_WRITE_LOCK:
        current_trades = _load_trades()
        current_portfolio = _load_portfolio_data()
        _, expected_portfolio = _calculate_trade_state(current_trades, current_portfolio)
        if _portfolio_signature(expected_portfolio) != _portfolio_signature(current_portfolio):
            raise LedgerConflictError(
                "当前持仓与交易账本不一致，已停止自动重建；请先核对或补录期初/调整交易"
            )

        normalized, portfolio = _calculate_trade_state(trades, metadata or current_portfolio)
        try:
            _save_trades(normalized)
            _save_json_file(PORTFOLIO_FILE, portfolio)
        except Exception:
            try:
                _save_trades(current_trades)
                _save_json_file(PORTFOLIO_FILE, current_portfolio)
            except Exception:
                pass
            raise
        return normalized, portfolio


@app.route("/api/trades", methods=["GET"])
def api_trades():
    """Get all trade records"""
    trades = _load_trades()
    # Add original index for deletion
    for i, t in enumerate(trades):
        t["_index"] = i
    # Return newest first
    trades.reverse()
    return jsonify({"trades": trades})


@app.route("/api/trades/add", methods=["POST"])
def api_trades_add():
    """新增交易；按完整账本校验，任何时点都禁止超卖。"""
    data = request.get_json(silent=True) or {}
    code = str(data.get("code") or "").strip()
    action = str(data.get("action") or "").strip().lower()
    if not re.fullmatch(r"\d{6}", code):
        return jsonify({"error": "code 必须是6位数字"}), 400
    if action not in ("buy", "sell"):
        return jsonify({"error": "action 必须是 buy 或 sell"}), 400
    try:
        price = float(data.get("price"))
        shares_number = float(data.get("shares"))
    except (TypeError, ValueError, OverflowError):
        return jsonify({"error": "price、shares 必须是数字"}), 400
    if (not math.isfinite(price) or not math.isfinite(shares_number) or
            price <= 0 or shares_number <= 0 or not shares_number.is_integer()):
        return jsonify({"error": "price 必须为有限正数，shares 必须为正整数"}), 400
    shares = int(shares_number)

    name = str(data.get("name") or "").strip()
    if not name:
        try:
            symbol = _market_symbol(code, data.get("market"))
            fields = _quote.get_quotes([symbol]).get(symbol)
            name = _fields_info(fields).get("name", "") if fields else ""
        except Exception:
            name = ""

    trade = {
        "id": uuid.uuid4().hex,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "code": code,
        "name": name or code,
        "action": action,
        "price": price,
        "shares": shares,
        "amount": round(price * shares, 2),
    }
    trades = _load_trades() + [trade]
    try:
        normalized, _ = _commit_trade_state(trades)
    except LedgerConflictError as exc:
        return jsonify({"error": str(exc)}), 409
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    saved_trade = next(item for item in normalized if item.get("id") == trade["id"])
    return jsonify({"success": True, "trade": saved_trade})


@app.route("/api/trades/delete", methods=["POST"])
def api_trades_delete():
    """删除交易并重算账本；若删除后产生超卖则拒绝。"""
    data = request.get_json(silent=True) or {}
    trades = _load_trades()
    trade_id = str(data.get("id") or "").strip()
    if trade_id:
        index = next((i for i, item in enumerate(trades) if item.get("id") == trade_id), -1)
        if index < 0:
            return jsonify({"error": "交易记录已变化，请刷新后重试"}), 409
    else:
        try:
            index = int(data.get("index"))
        except (TypeError, ValueError):
            return jsonify({"error": "id 或 index 必须提供"}), 400
        if index < 0 or index >= len(trades):
            return jsonify({"error": "交易记录已变化，请刷新后重试"}), 409

    candidate = [dict(item) for item in trades]
    candidate.pop(index)
    try:
        _commit_trade_state(candidate)
    except ValueError as exc:
        return jsonify({"error": f"无法删除：{exc}"}), 409
    return jsonify({"success": True})


@app.route("/api/trades/edit", methods=["POST"])
def api_trades_edit():
    """编辑交易并重算全部卖出收益及当前持仓。"""
    data = request.get_json(silent=True) or {}
    trades = _load_trades()
    trade_id = str(data.get("id") or "").strip()
    if trade_id:
        index = next((i for i, item in enumerate(trades) if item.get("id") == trade_id), -1)
        if index < 0:
            return jsonify({"error": "交易记录已变化，请刷新后重试"}), 409
    else:
        try:
            index = int(data.get("index"))
        except (TypeError, ValueError):
            return jsonify({"error": "id 或 index 必须提供"}), 400
        if index < 0 or index >= len(trades):
            return jsonify({"error": "交易记录已变化，请刷新后重试"}), 409

    candidate = [dict(item) for item in trades]
    trade = candidate[index]
    trade["id"] = str(trade.get("id") or uuid.uuid4().hex)
    target_id = trade["id"]
    for key in ("price", "shares", "action", "date", "code", "name"):
        if key in data:
            trade[key] = data[key]
    trade["code"] = str(trade.get("code") or "").strip()
    trade["action"] = str(trade.get("action") or "").strip().lower()
    trade["date"] = str(trade.get("date") or "").strip()
    if not trade["date"]:
        return jsonify({"error": "date 不能为空"}), 400

    try:
        normalized, _ = _commit_trade_state(candidate)
    except LedgerConflictError as exc:
        return jsonify({"error": str(exc)}), 409
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    saved_trade = next(item for item in normalized if item.get("id") == target_id)
    return jsonify({"success": True, "trade": saved_trade})


def _rebuild_portfolio_from_trades(trades):
    """兼容旧调用：使用统一校验与重算流程。"""
    return _commit_trade_state(trades)[1]


@app.route("/api/knowledge/search", methods=["POST"])
def api_knowledge_search():
    from knowledge.rag import search_knowledge
    data = request.get_json(silent=True) or {}
    query = data.get("query", "")
    if not query:
        return jsonify({"error": "Query cannot be empty"}), 400
    results = search_knowledge(query, top_k=5)
    return jsonify({"results": results})


@app.route("/api/knowledge/add", methods=["POST"])
def api_knowledge_add():
    from knowledge.rag import add_investment_note
    data = request.get_json(silent=True) or {}
    note = data.get("note", "")
    stock_code = data.get("stock_code", "")
    category = data.get("category", "note")
    if not note:
        return jsonify({"error": "Note content cannot be empty"}), 400
    doc_id = add_investment_note(note, stock_code, category)
    return jsonify({"success": True, "doc_id": doc_id})


@app.route("/api/knowledge/stats", methods=["GET"])
def api_knowledge_stats():
    from knowledge.rag import get_knowledge_stats
    stats = get_knowledge_stats()
    return jsonify(stats)


@app.route("/api/chat-history", methods=["GET"])
def api_chat_history():
    """Get saved chat sessions"""
    import json as json_mod
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
            return jsonify(json_mod.load(f))
    return jsonify({"sessions": []})


@app.route("/api/chat-history/save", methods=["POST"])
def api_chat_history_save():
    """Save chat sessions"""
    import json as json_mod
    data = request.get_json(silent=True) or {}
    with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
        json_mod.dump(data, f, ensure_ascii=False, indent=2)
    return jsonify({"success": True})


@app.route("/api/chat-history/clear", methods=["POST"])
def api_chat_history_clear():
    """Clear all chat history"""
    if os.path.exists(CHAT_HISTORY_FILE):
        os.remove(CHAT_HISTORY_FILE)
    return jsonify({"success": True})


@app.route("/api/ai-balance", methods=["GET"])
def api_ai_balance():
    """Query DeepSeek API balance"""
    try:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        with requests.Session() as session:
            session.trust_env = False
            resp = session.get(
                "https://api.deepseek.com/user/balance",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=10,
            )
            status_code = resp.status_code
            data = resp.json() if status_code == 200 else None
        if status_code == 200:
            if data.get("balance_infos"):
                info = data["balance_infos"][0]
                return jsonify({
                    "balance": info.get("total_balance", "0"),
                    "topped_up": info.get("topped_up_balance", "0"),
                    "granted": info.get("granted_balance", "0"),
                    "currency": info.get("currency", "CNY"),
                })
            # Alternative response format
            return jsonify({
                "balance": data.get("total_balance", data.get("balance", "0")),
                "topped_up": data.get("topped_up_balance", "0"),
                "granted": data.get("granted_balance", "0"),
                "currency": data.get("currency", "CNY"),
            })
        return jsonify({"error": f"API returned {status_code}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ===== 新功能 API =====

@app.route("/api/profit-curve", methods=["GET"])
def api_profit_curve():
    """获取收益曲线数据"""
    try:
        from tools.portfolio_tracker import _load_history
        period = request.args.get("period", "all")
        history = _load_history()
        if not history:
            return jsonify({"data": [], "stats": {}})

        from datetime import timedelta
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

        stats = {}
        if filtered:
            first = filtered[0]
            last = filtered[-1]
            stats = {
                "period_return": round(last["profit_pct"] - first["profit_pct"], 2),
                "total_return": last["profit_pct"],
                "total_value": last["total_value"],
                "total_profit": last["profit"],
                "days": len(filtered),
            }
            # Max drawdown
            peak = filtered[0]["total_value"]
            max_dd = 0
            for h in filtered:
                if h["total_value"] > peak:
                    peak = h["total_value"]
                dd = (h["total_value"] / peak - 1) * 100
                if dd < max_dd:
                    max_dd = dd
            stats["max_drawdown"] = round(max_dd, 2)

        return jsonify({"data": filtered, "stats": stats})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/profit-curve/record", methods=["POST"])
def api_record_snapshot():
    """手动触发记录今日快照"""
    try:
        from tools.portfolio_tracker import record_daily_snapshot
        snapshot = record_daily_snapshot()
        if not snapshot:
            return jsonify({"error": "无持仓数据"}), 400
        return jsonify({"success": True, "snapshot": snapshot})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/sector-rotation", methods=["GET"])
def api_sector_rotation():
    """获取板块轮动数据"""
    try:
        from tools.sector_rotation import _get_sector_prices, SECTOR_ETFS
        prices = _get_sector_prices()
        sectors = []
        for sector_name, etf_code in SECTOR_ETFS.items():
            if etf_code in prices:
                data = prices[etf_code]
                sectors.append({
                    "name": sector_name,
                    "price": data["price"],
                    "change_pct": data["change_pct"],
                    "turnover": data["turnover"],
                    "amount": data["amount"],
                })
        sectors.sort(key=lambda x: x["change_pct"], reverse=True)
        return jsonify({"sectors": sectors})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/sector-rotation/history", methods=["GET"])
def api_sector_rotation_history():
    """获取板块近期走势对比"""
    try:
        from tools.sector_rotation import get_sector_history
        histories = get_sector_history(20)
        result = {}

        for sector_name, klines in histories.items():
            try:
                if len(klines) < 5:
                    continue
                closes = [float(k[2]) for k in klines]
                base = closes[0]
                result[sector_name] = {
                    "dates": [k[0] for k in klines],
                    "returns": [round((close / base - 1) * 100, 2) for close in closes],
                    "pct_5d": round((closes[-1] / closes[-5] - 1) * 100, 2),
                    "pct_20d": round((closes[-1] / closes[0] - 1) * 100, 2),
                }
            except (IndexError, TypeError, ValueError, ZeroDivisionError):
                continue

        return jsonify({"sectors": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/backtest", methods=["POST"])
def api_backtest():
    """运行回测"""
    try:
        data = request.get_json(silent=True) or {}
        strategy = data.get("strategy", "ma")
        symbol = data.get("symbol", "515290")
        capital = float(data.get("capital", 10000))

        if strategy == "ma":
            from tools.backtest import backtest_ma_strategy
            result = backtest_ma_strategy.invoke({
                "symbol": symbol,
                "short_period": int(data.get("short_period", 5)),
                "long_period": int(data.get("long_period", 20)),
                "initial_capital": capital,
            })
        elif strategy == "grid":
            from tools.backtest import backtest_grid_strategy
            result = backtest_grid_strategy.invoke({
                "symbol": symbol,
                "grid_pct": float(data.get("grid_pct", 3.0)),
                "initial_capital": capital,
            })
        elif strategy == "dca":
            from tools.backtest import backtest_dca_strategy
            result = backtest_dca_strategy.invoke({
                "symbol": symbol,
                "interval_days": int(data.get("interval_days", 22)),
                "amount_per_time": float(data.get("amount_per_time", 2000)),
            })
        else:
            return jsonify({"error": f"未知策略: {strategy}"}), 400

        return jsonify({"result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/valuation", methods=["GET"])
def api_valuation():
    """获取估值分析"""
    try:
        symbol = request.args.get("symbol", "515290")
        from tools.stock_screener import analyze_valuation
        result = analyze_valuation.invoke({"symbol": symbol})
        return jsonify({"result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/screener", methods=["GET"])
def api_screener():
    """低估值筛选"""
    try:
        sector = request.args.get("sector", "银行")
        from tools.stock_screener import screen_undervalued_stocks
        result = screen_undervalued_stocks.invoke({"sector": sector})
        return jsonify({"result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/position/grid", methods=["POST"])
def api_grid_plan():
    """计算网格交易方案"""
    try:
        data = request.get_json(silent=True) or {}
        from tools.position_manager import calculate_grid_trading
        result = calculate_grid_trading.invoke({
            "symbol": data.get("symbol", "515290"),
            "grid_count": int(data.get("grid_count", 5)),
            "grid_range_pct": float(data.get("grid_range_pct", 10.0)),
            "amount_per_grid": float(data.get("amount_per_grid", 1000.0)),
        })
        return jsonify({"result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/position/add-calc", methods=["POST"])
def api_add_position_calc():
    """计算加仓方案"""
    try:
        data = request.get_json(silent=True) or {}
        from tools.position_manager import calculate_add_position
        result = calculate_add_position.invoke({
            "symbol": data.get("symbol", "515290"),
            "target_cost": float(data.get("target_cost", 0)),
            "add_amount": float(data.get("add_amount", 2000.0)),
        })
        return jsonify({"result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ===== 公司预期变化研究 API =====

@app.route("/api/company-research/tasks", methods=["POST"])
def api_company_research_create():
    """创建后台研究任务，立即返回 task_id，避免阻塞其他写接口。"""
    try:
        data = request.get_json(silent=True) or {}
        company = str(data.get("company", "")).strip()
        symbol = str(data.get("symbol", "")).strip()
        depth = str(data.get("search_depth", "basic")).strip()
        if not company:
            return jsonify({"error": "请输入公司名称或股票代码"}), 400
        if symbol and (len(symbol) != 6 or not symbol.isdigit()):
            return jsonify({"error": "股票代码必须是6位数字"}), 400
        if depth not in {"basic", "deep"}:
            return jsonify({"error": "检索深度必须是 basic 或 deep"}), 400
        from research_system.web_jobs import start_job
        job = start_job({
            "company": company,
            "symbol": symbol,
            "search_depth": depth,
        })
        job.pop("result", None)
        return jsonify(job), 202
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/company-research/tasks/<task_id>", methods=["GET"])
def api_company_research_status(task_id):
    from research_system.web_jobs import get_job
    job = get_job(task_id)
    if not job:
        return jsonify({"error": "研究任务不存在或已过期"}), 404
    return jsonify(job)


@app.route("/api/company-research/analyses", methods=["GET"])
def api_company_research_analyses():
    try:
        limit = min(max(int(request.args.get("limit", 10)), 1), 50)
        from research_system.repository import ResearchRepository
        return jsonify({"analyses": ResearchRepository().list_analyses(limit)})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/company-research/analyses/<analysis_id>", methods=["GET"])
def api_company_research_analysis(analysis_id):
    from research_system.repository import ResearchRepository
    result = ResearchRepository().get_analysis(analysis_id)
    if not result:
        return jsonify({"error": "研究记录不存在"}), 404
    stages = result.pop("stages", {})
    result.update(stages)
    return jsonify(result)


_signal_monitor_lock = threading.Lock()
_signal_monitor_stop = threading.Event()
_signal_monitor_thread = None


def _extract_signal_direction(signal_text):
    if "重仓加仓" in signal_text:
        return "add_heavy"
    if "适合加仓" in signal_text:
        return "add"
    if "小幅加仓" in signal_text:
        return "add_light"
    if "止盈" in signal_text:
        return "take_profit"
    if "减仓" in signal_text:
        return "sell"
    return "hold"


def _signal_monitor_loop():
    last_directions = {}
    while not _signal_monitor_stop.is_set():
        try:
            now = datetime.now()
            if now.weekday() < 5 and 9 <= now.hour <= 14:
                from agents import RiskMonitor
                monitor = RiskMonitor()
                signals = monitor.get_signals()
                current_directions = {}
                has_actionable = False
                for signal in signals:
                    signal_text = signal["signal"]
                    current_directions[signal["code"]] = _extract_signal_direction(signal_text)
                    if any(keyword in signal_text for keyword in ["加仓", "止盈", "减仓"]):
                        has_actionable = True

                has_change = any(
                    last_directions.get(code) != direction
                    for code, direction in current_directions.items()
                )
                if has_change and has_actionable:
                    last_directions.update(current_directions)
                    message = f"📊 操作信号更新 ({now.strftime('%H:%M')})\n\n"
                    for signal in signals:
                        held = "[持仓]" if signal["code"] in monitor.portfolio else "[观察]"
                        message += f"▸ {signal['name']}({signal['code']}) {held}\n"
                        message += f"  {signal['signal']}\n\n"
                    _send_notification("DataBoard 操作信号", message)
                elif has_change:
                    last_directions.update(current_directions)
        except Exception:
            pass

        # Event.wait 可被 stop 立即唤醒，避免 sleep 导致线程无法及时退出。
        _signal_monitor_stop.wait(30 * 60)


def start_signal_monitor():
    """幂等启动后台信号监控，同一进程最多一个线程。"""
    global _signal_monitor_thread
    with _signal_monitor_lock:
        if _signal_monitor_thread and _signal_monitor_thread.is_alive():
            return _signal_monitor_thread
        _signal_monitor_stop.clear()
        _signal_monitor_thread = threading.Thread(
            target=_signal_monitor_loop,
            name="signal-monitor",
            daemon=True,
        )
        _signal_monitor_thread.start()
        return _signal_monitor_thread


def stop_signal_monitor(timeout=2):
    """停止后台信号监控，主要供嵌入式运行和测试清理使用。"""
    global _signal_monitor_thread
    with _signal_monitor_lock:
        thread = _signal_monitor_thread
        if not thread:
            return
        _signal_monitor_stop.set()
    thread.join(timeout=timeout)
    with _signal_monitor_lock:
        if not thread.is_alive() and _signal_monitor_thread is thread:
            _signal_monitor_thread = None


def _send_notification(title, content):
    """统一推送：走 Bark（iOS），未配置或失败时打印到控制台。"""
    bark_url = os.getenv("BARK_URL")
    if bark_url:
        try:
            with requests.Session() as session:
                session.trust_env = False
                # 用 POST 表单提交，避免长文本/换行撑爆 URL
                session.post(
                    bark_url.rstrip("/"),
                    data={
                        "title": title,
                        "body": content,
                        "group": "DataBoard",
                        "icon": "https://img.icons8.com/color/96/combo-chart.png",
                    },
                    timeout=10,
                )
            return True
        except Exception:
            pass

    # 未配置 Bark 或推送失败：打印到控制台
    print(f"\n[通知] {title}: {content}\n")
    return False


if __name__ == "__main__":
    start_signal_monitor()
    # 安全：默认绑定本机、关闭 debug（Werkzeug 调试器可致远程代码执行）
    _host = os.getenv("DASHBOARD_HOST", "127.0.0.1")
    _debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(host=_host, port=5000, debug=_debug)
