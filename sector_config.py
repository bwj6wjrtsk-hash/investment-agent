"""
Sector auto-detection and configuration.
Given a stock/ETF code, automatically determine its sector and provide
representative stocks + news keywords.
"""

# Mapping of known ETF codes to sectors
ETF_SECTOR_MAP = {
    # 银行
    "515290": "银行", "512700": "银行", "512800": "银行", "516820": "银行",
    # 医药
    "512010": "医药", "159929": "医药", "512290": "医药", "159992": "医药",
    # 半导体/芯片
    "512480": "半导体", "159995": "半导体", "512760": "半导体",
    # 新能源
    "516160": "新能源", "159875": "新能源", "515790": "新能源",
    # 消费
    "159928": "消费", "510150": "消费", "516170": "消费",
    # 军工
    "512660": "军工", "512810": "军工", "159516": "军工",
    # 证券
    "512880": "证券", "512000": "证券", "159841": "证券",
    # 房地产
    "512200": "房地产", "159768": "房地产",
    # 科技/信息技术
    "515000": "科技", "159939": "科技",
    # 食品饮料
    "515170": "食品饮料", "159843": "食品饮料",
    # 汽车
    "516110": "汽车", "159845": "汽车",
    # 电力
    "159611": "电力", "562350": "电力",
    # AI/人工智能
    "515070": "人工智能", "159819": "人工智能",
}

# Sector representative stocks (for sector panel display)
SECTOR_STOCKS = {
    "银行": ["sh601398", "sh601939", "sh601288", "sh600036", "sh601166", "sh600015", "sh601328", "sh601818"],
    "医药": ["sh600276", "sz000538", "sh603259", "sz300760", "sz300347", "sh688180"],
    "半导体": ["sh688981", "sz002049", "sh688396", "sz300661", "sz002371", "sh603501"],
    "新能源": ["sz300750", "sh601012", "sz002459", "sh600438", "sz300274", "sz002709"],
    "消费": ["sh600519", "sz000858", "sh603288", "sz002304", "sz000568", "sh600887"],
    "军工": ["sh600893", "sh601989", "sz002179", "sh600760", "sz000768", "sh601698"],
    "证券": ["sh601377", "sh600030", "sz000166", "sh601688", "sh601211", "sh600999"],
    "房地产": ["sh600048", "sz000002", "sz001979", "sh600383", "sh600606", "sz000069"],
    "科技": ["sz002415", "sh603019", "sz000977", "sz002230", "sh688111", "sz300496"],
    "食品饮料": ["sh600519", "sz000858", "sh600887", "sz002304", "sh603369", "sz000568"],
    "汽车": ["sz002594", "sh601238", "sh600104", "sz000625", "sz002920", "sh601127"],
    "电力": ["sh600900", "sh601985", "sh600025", "sz003816", "sh600886", "sz000883"],
    "人工智能": ["sz002230", "sh603019", "sz300474", "sz300418", "sh688787", "sz002049"],
    "机械": ["sh601766", "sh601100", "sz000157", "sh600031", "sz000425", "sh601169"],
    "家电": ["sz000651", "sz000333", "sh600690", "sz000921", "sh603486", "sz002032"],
}

# Sector news keywords
SECTOR_KEYWORDS = {
    "银行": ["银行", "利率", "央行", "降息", "降准", "LPR", "房贷", "信贷", "存款", "贷款", "净息差", "MLF"],
    "医药": ["医药", "创新药", "集采", "医保", "药品审批", "CXO", "临床试验", "FDA", "生物医药"],
    "半导体": ["半导体", "芯片", "晶圆", "光刻", "EDA", "国产替代", "华为", "AI芯片", "存储"],
    "新能源": ["新能源", "锂电", "光伏", "风电", "储能", "碳中和", "电池", "充电桩"],
    "消费": ["消费", "白酒", "零售", "内需", "社零", "品牌", "涨价"],
    "军工": ["军工", "国防", "航空", "导弹", "卫星", "军费", "装备"],
    "证券": ["证券", "券商", "IPO", "注册制", "成交量", "两融", "基金发行"],
    "房地产": ["房地产", "楼市", "房价", "限购", "房企", "土地", "保交楼"],
    "科技": ["科技", "人工智能", "AI", "大模型", "算力", "数据中心", "云计算"],
    "食品饮料": ["食品", "饮料", "白酒", "消费升级", "涨价", "旺季"],
    "汽车": ["汽车", "新能源车", "销量", "出口", "智能驾驶", "电动车"],
    "电力": ["电力", "电网", "水电", "核电", "电价", "用电量"],
    "人工智能": ["AI", "人工智能", "大模型", "ChatGPT", "算力", "GPU", "机器人"],
    "机械": ["中车", "高铁", "铁路", "轨道交通", "机械", "装备制造", "工程机械", "盾构", "城轨"],
    "家电": ["家电", "空调", "冰箱", "洗衣机", "小家电", "以旧换新"],
}

# Stock code prefix to exchange mapping
# 6xx = Shanghai, 0xx/3xx = Shenzhen, 4xx/8xx/9xx = Beijing
STOCK_SECTOR_HINTS = {
    # Some well-known individual stocks -> sector
    "601398": "银行", "601939": "银行", "601288": "银行", "600036": "银行",
    "600519": "消费", "000858": "消费",
    "300750": "新能源", "002594": "汽车",
    "600276": "医药", "000538": "医药",
    "600893": "军工", "601989": "军工",
    "601377": "证券", "600030": "证券",
    "601766": "机械", "601698": "军工", "000651": "家电",
}


_ONLINE_SECTOR_MISSES = set()


def detect_sector(code: str, name: str = "", allow_online: bool = False) -> str:
    """识别板块；页面请求默认只走本地规则，在线扫描必须显式启用。"""
    # Check ETF map first
    if code in ETF_SECTOR_MAP:
        return ETF_SECTOR_MAP[code]

    # Check known stocks
    if code in STOCK_SECTOR_HINTS:
        return STOCK_SECTOR_HINTS[code]

    # Guess from name
    name_hints = {
        "银行": "银行", "医药": "医药", "半导体": "半导体", "芯片": "半导体",
        "新能源": "新能源", "锂电": "新能源", "光伏": "新能源",
        "消费": "消费", "白酒": "消费", "食品": "食品饮料",
        "军工": "军工", "国防": "军工",
        "证券": "证券", "券商": "证券",
        "地产": "房地产", "房": "房地产",
        "科技": "科技", "AI": "人工智能", "智能": "人工智能",
        "汽车": "汽车", "电力": "电力",
        "中车": "机械", "机械": "机械", "铁路": "机械", "高铁": "机械",
    }
    for keyword, sector in name_hints.items():
        if keyword in name:
            return sector

    if not allow_online or code in _ONLINE_SECTOR_MISSES:
        return ""

    # 显式启用时才扫描新浪板块；成功和失败都缓存，避免重复遍历。
    sector = _online_detect_sector(code)
    if sector:
        STOCK_SECTOR_HINTS[code] = sector
        return sector
    _ONLINE_SECTOR_MISSES.add(code)
    return ""


def _online_detect_sector(code: str) -> str:
    """Query Sina API to find which sector a stock belongs to"""
    import requests
    import json

    s = requests.Session()
    s.trust_env = False

    from tools import quote as _q
    prefix = _q.code_prefix(code)

    # Map of Sina nodes to our sector names
    node_to_sector = {
        "new_dzxx": "电子信息", "new_yysw": "医药生物", "new_jrhy": "金融",
        "new_zqhy": "证券", "new_fdc": "房地产", "new_dlhy": "电力",
        "new_qcgy": "汽车", "new_spyl": "食品饮料", "new_jzjc": "建筑建材",
        "new_dqhy": "电气设备", "new_cmyl": "传媒娱乐", "new_blhy": "玻璃",
        "new_fdsb": "电力设备", "new_ylqx": "医疗器械", "new_cbzz": "船舶制造",
        "new_fzhy": "纺织服装", "new_gfjs": "军工", "new_jtys": "交通运输",
        "new_hghy": "化工", "new_jxhy": "机械", "new_jshy": "钢铁",
        "new_kchy": "矿产", "new_lyjd": "旅游酒店", "new_nfhy": "农林牧渔",
        "new_ysjs": "有色金属", "new_sxhy": "石油化工", "new_rjhy": "软件",
        "new_zthy": "造纸印刷", "new_slhy": "塑料制品", "new_jdhy": "家电",
    }

    for node, sector_name in node_to_sector.items():
        try:
            r = s.get(
                "https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/Market_Center.getHQNodeData",
                params={"page": "1", "num": "80", "sort": "symbol", "asc": "1", "node": node},
                timeout=2
            )
            if r.status_code == 200 and r.text.strip() != "[]":
                items = json.loads(r.text)
                codes = [i["symbol"] for i in items]
                if prefix in codes:
                    return sector_name
        except Exception:
            continue

    return ""


def get_sector_config(sector_name: str) -> dict:
    """Get full config for a sector. If not preconfigured, fetch dynamically."""
    stocks = SECTOR_STOCKS.get(sector_name, [])
    keywords = SECTOR_KEYWORDS.get(sector_name, [])

    # If no preset stocks, try to fetch from Sina
    if not stocks:
        stocks = _fetch_sector_stocks(sector_name)

    # If no keywords, use sector name as keyword
    if not keywords:
        keywords = [sector_name]

    return {
        "name": sector_name,
        "stocks": stocks,
        "keywords": keywords,
    }


def _fetch_sector_stocks(sector_name: str) -> list:
    """Fetch top stocks in a sector from Sina API"""
    import requests
    import json

    # Reverse map: sector name -> Sina node
    sector_to_node = {
        "电子信息": "new_dzxx", "医药生物": "new_yysw", "金融": "new_jrhy",
        "证券": "new_zqhy", "房地产": "new_fdc", "电力": "new_dlhy",
        "汽车": "new_qcgy", "食品饮料": "new_spyl", "建筑建材": "new_jzjc",
        "电气设备": "new_dqhy", "传媒娱乐": "new_cmyl", "玻璃": "new_blhy",
        "电力设备": "new_fdsb", "医疗器械": "new_ylqx", "船舶制造": "new_cbzz",
        "纺织服装": "new_fzhy", "军工": "new_gfjs", "交通运输": "new_jtys",
        "化工": "new_hghy", "机械": "new_jxhy", "钢铁": "new_jshy",
        "矿产": "new_kchy", "旅游酒店": "new_lyjd", "农林牧渔": "new_nfhy",
        "有色金属": "new_ysjs", "石油化工": "new_sxhy", "软件": "new_rjhy",
        "造纸印刷": "new_zthy", "塑料制品": "new_slhy", "家电": "new_jdhy",
    }

    node = sector_to_node.get(sector_name, "")
    if not node:
        return []

    try:
        s = requests.Session()
        s.trust_env = False
        r = s.get(
            "https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/Market_Center.getHQNodeData",
            params={"page": "1", "num": "8", "sort": "nmc", "asc": "0", "node": node},
            timeout=5
        )
        if r.status_code == 200 and r.text.strip() != "[]":
            items = json.loads(r.text)
            # Convert to Tencent format codes
            stocks = []
            for item in items[:8]:
                symbol = item.get("symbol", "")
                if symbol.startswith("sh"):
                    stocks.append(symbol)
                elif symbol.startswith("sz"):
                    stocks.append(symbol)
            return stocks
    except:
        pass

    return []
