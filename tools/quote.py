"""
统一行情数据层。
集中处理：交易所前缀（含北交所）、腾讯实时行情、腾讯K线、真实涨跌家数，
并带轻量 TTL 缓存，避免各模块重复实现与重复请求。
"""
import re
import time
import threading
from collections import OrderedDict

import requests

# 行情/K线缓存均有硬容量上限；固定分片锁避免按用户输入永久创建 Lock。
_QUOTE_CACHE = OrderedDict()   # prefix_code -> (ts, fields)
_KLINE_CACHE = OrderedDict()   # prefix_code -> (ts, fetched_days, closes)
_BREADTH_CACHE = {"ts": 0, "ttl": 0, "data": None}
_LOCK = threading.Lock()
_FETCH_LOCK_STRIPES = 64
_QUOTE_FETCH_LOCKS = [threading.Lock() for _ in range(_FETCH_LOCK_STRIPES)]
_KLINE_FETCH_LOCKS = [threading.Lock() for _ in range(_FETCH_LOCK_STRIPES)]
_BREADTH_FETCH_LOCK = threading.Lock()

QUOTE_TTL = 15      # 实时行情缓存秒数
KLINE_TTL = 300     # K线缓存秒数
BREADTH_TTL = 60    # 涨跌家数缓存秒数
BREADTH_FAILURE_TTL = 15  # 上游异常短暂负缓存，避免连续请求阻塞首页
QUOTE_CACHE_MAXSIZE = 2048
KLINE_CACHE_MAXSIZE = 512
_SYMBOL_RE = re.compile(r"^(?:(?:sh|sz|bj)\d{6}|hk\d{5}|\d{5,6})$", re.IGNORECASE)

# 腾讯行情字段索引（~ 分隔）常量，避免魔法数字
F_NAME = 1
F_CODE = 2
F_PRICE = 3
F_PREV_CLOSE = 4
F_OPEN = 5
F_VOLUME = 6
F_CHANGE_AMT = 31
F_CHANGE_PCT = 32
F_HIGH = 33
F_LOW = 34
F_AMOUNT = 37       # 成交额(万元)
F_TURNOVER = 38     # 换手率
F_PE = 39
F_PB = 46


def _session():
    s = requests.Session()
    s.trust_env = False
    return s


def is_valid_symbol(symbol: str) -> bool:
    """只接受 A/港股数字代码及明确的交易所前缀，阻止任意缓存键。"""
    return bool(_SYMBOL_RE.fullmatch((symbol or "").strip()))


def code_prefix(symbol: str) -> str:
    """给股票/基金代码加正确的交易所前缀（支持沪/深/北交所及港交所）。"""
    s = (symbol or "").strip().lower()
    if not s:
        return s
    market = s[:2]
    if market in ("sh", "sz", "bj", "hk"):
        return market + s[2:]
    # 港股使用 5 位数字代码；A 股、ETF 和北交所代码均为 6 位。
    if s.isdigit() and len(s) == 5:
        return "hk" + s
    c = s[0]
    if c == "6" or c == "9":          # 沪市A股 / B股
        return "sh" + s
    if c == "5":                        # 沪市基金/ETF
        return "sh" + s
    if c in ("0", "3"):                # 深市A股/创业板
        return "sz" + s
    if c == "1":                        # 深市基金/ETF/LOF
        return "sz" + s
    if c in ("4", "8"):                # 北交所
        return "bj" + s
    return ""


def _cache_get(cache, key, ttl, now=None):
    """调用方持有 _LOCK；命中时刷新 LRU，过期时立即删除。"""
    cached = cache.get(key)
    if not cached:
        return None
    if (now or time.time()) - cached[0] > ttl:
        cache.pop(key, None)
        return None
    cache.move_to_end(key)
    return cached


def _cache_put(cache, key, value, maxsize):
    """调用方持有 _LOCK；写入并按 LRU 淘汰至硬容量上限。"""
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > maxsize:
        cache.popitem(last=False)


def _striped_locks(symbols, stripes):
    indexes = sorted({hash(symbol) % len(stripes) for symbol in symbols})
    return [stripes[index] for index in indexes]


def _safe_float(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def get_quotes(codes, ttl=QUOTE_TTL) -> dict:
    """批量获取实时行情原始字段；忽略非法代码并使用有界 TTL/LRU 缓存。"""
    prefixed = [code_prefix(c) for c in codes if is_valid_symbol(c)]
    prefixed = [code for code in prefixed if code]
    now = time.time()
    result = {}
    missing = []
    with _LOCK:
        for p in prefixed:
            cached = _cache_get(_QUOTE_CACHE, p, ttl, now)
            if cached:
                result[p] = cached[1]
            else:
                missing.append(p)

    if missing:
        symbols = sorted(set(missing))
        fetch_locks = _striped_locks(symbols, _QUOTE_FETCH_LOCKS)
        for lock in fetch_locks:
            lock.acquire()
        try:
            pending = []
            now = time.time()
            with _LOCK:
                for p in symbols:
                    cached = _cache_get(_QUOTE_CACHE, p, ttl, now)
                    if cached:
                        result[p] = cached[1]
                    else:
                        pending.append(p)
            if pending:
                try:
                    with _session() as session:
                        resp = session.get(f"https://qt.gtimg.cn/q={','.join(pending)}", timeout=10)
                        resp.encoding = "gbk"
                        text = resp.text
                    fetched_at = time.time()
                    with _LOCK:
                        for line in text.strip().split(";"):
                            line = line.strip()
                            if not line:
                                continue
                            match = re.match(r'v_(\w+)="(.+)"', line)
                            if match:
                                code = match.group(1)
                                fields = match.group(2).split("~")
                                if code in pending and len(fields) > F_CHANGE_PCT:
                                    _cache_put(
                                        _QUOTE_CACHE, code, (fetched_at, fields),
                                        QUOTE_CACHE_MAXSIZE,
                                    )
                                    result[code] = fields
                except Exception:
                    pass  # 网络失败时返回已命中缓存的部分
        finally:
            for lock in reversed(fetch_locks):
                lock.release()
    return result


def get_price(symbol: str):
    """获取单只标的现价，取不到返回 None。"""
    p = code_prefix(symbol)
    data = get_quotes([symbol])
    if p in data:
        return _safe_float(data[p][F_PRICE], None)
    return None


def get_closes(symbol: str, days: int = 120, ttl=KLINE_TTL):
    """获取前复权日K收盘价；较长窗口可复用给较短窗口。"""
    if not is_valid_symbol(symbol):
        return None
    p = code_prefix(symbol)
    requested_days = max(1, min(int(days), 1000))
    fetch_days = max(120, requested_days)

    def cached_result(now):
        cached = _cache_get(_KLINE_CACHE, p, ttl, now)
        if cached and cached[1] >= requested_days:
            return cached[2][-requested_days:]
        return None

    with _LOCK:
        cached = cached_result(time.time())
        if cached is not None:
            return cached
    fetch_lock = _KLINE_FETCH_LOCKS[hash(p) % len(_KLINE_FETCH_LOCKS)]

    with fetch_lock:
        with _LOCK:
            cached = cached_result(time.time())
            if cached is not None:
                return cached
        try:
            with _session() as session:
                resp = session.get(
                    f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={p},day,,,{fetch_days},qfq",
                    timeout=10,
                )
                data = resp.json()
            node = data.get("data", {}).get(p, {})
            klines = node.get("day") or node.get("qfqday") or []
            closes = [_safe_float(k[2]) for k in klines if len(k) > 2]
            if closes:
                with _LOCK:
                    _cache_put(
                        _KLINE_CACHE, p, (time.time(), fetch_days, closes),
                        KLINE_CACHE_MAXSIZE,
                    )
                return closes[-requested_days:]
        except Exception:
            pass
    return None


def get_klines_raw(symbol: str, days: int = 120):
    """获取原始日K数组（含开高低收量），取不到返回 []。"""
    if not is_valid_symbol(symbol):
        return []
    p = code_prefix(symbol)
    days = max(1, min(int(days), 1000))
    try:
        with _session() as session:
            resp = session.get(
                f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={p},day,,,{days},qfq",
                timeout=10,
            )
            data = resp.json()
        node = data.get("data", {}).get(p, {})
        return node.get("day") or node.get("qfqday") or []
    except Exception:
        return []


_EM_HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://quote.eastmoney.com/"}


def _read_market_breadth_cache():
    with _LOCK:
        data = _BREADTH_CACHE["data"]
        if (data is not None and
                time.time() - _BREADTH_CACHE["ts"] <= _BREADTH_CACHE["ttl"]):
            return dict(data)
    return None


def get_cached_market_breadth() -> dict:
    """只读取有效缓存，不访问网络；首屏可立即返回。"""
    return _read_market_breadth_cache() or {"up": None, "down": None, "flat": None}


def get_market_breadth(ttl=BREADTH_TTL) -> dict:
    """获取真实涨跌家数；单次请求最多约 6 秒，失败短暂负缓存。"""
    cached = _read_market_breadth_cache()
    if cached is not None:
        return cached

    def remember(data, cache_ttl):
        with _LOCK:
            _BREADTH_CACHE.update({"ts": time.time(), "ttl": cache_ttl, "data": data})
        return data

    # 同一时间只允许一个请求链访问上游，其余请求等待并复用结果。
    with _BREADTH_FETCH_LOCK:
        cached = _read_market_breadth_cache()
        if cached is not None:
            return cached

        unknown = {"up": None, "down": None, "flat": None}
        fs = "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23,m:0+t:81+s:2048"
        base = "https://push2.eastmoney.com/api/qt/clist/get"

        def _count(extra_filter=None):
            params = {
                "pn": "1", "pz": "1", "po": "1", "np": "1",
                "fltt": "2", "invt": "2", "fid": "f3",
                "fs": fs, "fields": "f3",
            }
            if extra_filter:
                params["filter"] = extra_filter
            try:
                with _session() as session:
                    response = session.get(
                        base, params=params, timeout=3, headers=_EM_HEADERS,
                    )
                if response.status_code == 200:
                    return response.json().get("data", {}).get("total", 0) or 0
            except Exception:
                pass
            return None

        total = _count()
        if not total:
            return remember(unknown, BREADTH_FAILURE_TTL)

        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=2) as executor:
            up_future = executor.submit(_count, "(f3>0)")
            down_future = executor.submit(_count, "(f3<0)")
            up = up_future.result()
            down = down_future.result()

        if up is None or down is None or up >= total or down >= total or (up + down) > total:
            return remember(unknown, BREADTH_FAILURE_TTL)

        stats = {"up": up, "down": down, "flat": max(0, total - up - down)}
        return remember(stats, ttl)


def compute_rsi(closes, period: int = 14):
    """标准 Wilder RSI。数据不足返回 None。"""
    if not closes or len(closes) < period + 1:
        return None
    gains, losses = [], []
    for i in range(1, len(closes)):
        d = closes[i] - closes[i - 1]
        gains.append(max(d, 0.0))
        losses.append(max(-d, 0.0))
    # 首个平均
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    # Wilder 平滑
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _ema_series(values, period):
    k = 2 / (period + 1)
    ema = values[0]
    out = [ema]
    for v in values[1:]:
        ema = v * k + ema * (1 - k)
        out.append(ema)
    return out


def compute_macd(closes, fast=12, slow=26, signal=9):
    """标准 MACD，返回 (dif, dea, hist)。数据不足返回 (None, None, None)。"""
    if not closes or len(closes) < slow + signal:
        return None, None, None
    ema_fast = _ema_series(closes, fast)
    ema_slow = _ema_series(closes, slow)
    dif_series = [f - s for f, s in zip(ema_fast, ema_slow)]
    dea_series = _ema_series(dif_series, signal)
    dif = dif_series[-1]
    dea = dea_series[-1]
    hist = (dif - dea) * 2
    return dif, dea, hist
