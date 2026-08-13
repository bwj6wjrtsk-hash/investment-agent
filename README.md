# A股投资分析 Agent

一个基于 DeepSeek + LangChain 的个人A股投资助手，支持实时行情、技术分析、新闻汇总、风险监控、仓位管理、收益追踪、板块轮动、策略回测和AI选股。

## 功能

- 🔍 **实时行情查询** - 个股行情、大盘概况、热门股票
- 📊 **技术分析** - MA均线、MACD、RSI 等技术指标
- 📰 **新闻资讯** - 财经新闻、个股新闻、板块资金流向
- 📈 **基本面分析** - 市盈率、市净率、营收增速等
- ⚠️ **风险监控** - 止损止盈预警、异动提醒
- 📝 **知识管理** - 存储和检索投资笔记、研报
- 📅 **定时任务** - 每日日报、盘前概览、定时风控检查
- 📱 **消息推送** - Server酱/企业微信通知到手机
- 💰 **仓位管理** - 网格交易计算、加仓分析、定投模拟
- 📉 **收益追踪** - 每日持仓快照、收益曲线、周/月/年统计
- 🔄 **板块轮动** - 板块强弱排名、轮动信号、板块对比
- 🧪 **策略回测** - 均线策略、网格策略、定投策略回测
- 🎯 **AI选股估值** - PE/PB分位数分析、低估值筛选、AI综合分析

## 快速开始

### 1. 安装依赖

```bash
cd investment-agent
pip install -r requirements.txt
```

### 2. 配置 API Key

复制示例配置后填写自己的密钥（`.env` 已被 Git 忽略）：

```bash
copy .env.example .env
```

```env
DEEPSEEK_API_KEY=你的DeepSeek_API_Key
```

DeepSeek 注册地址: https://platform.deepseek.com/

### 3. 运行

```bash
# Web 界面（推荐，可视化操作）
python main.py --web

# 交互模式（命令行对话）
python main.py

# 立即生成日报
python main.py --report

# 检查持仓风险
python main.py --monitor

# 启动定时任务（后台运行）
python main.py --schedule
```

## 使用示例

```
你: 帮我分析一下 000001 平安银行
你: 今天大盘怎么样？
你: 哪些板块资金流入最多？
你: 最近有什么财经新闻？
你: 帮我记个笔记：平安银行在12元附近有支撑

# 新功能示例
你: 帮我算一下515290的网格交易方案
你: 如果加仓2000元，成本变成多少？
你: 模拟一下515290每月定投2000的效果
你: 记录一下今天的持仓快照
你: 看看我的收益曲线
你: 今天哪个板块最强？
你: 检测一下板块轮动信号
你: 对比一下银行和证券板块
你: 回测一下515290的均线交叉策略
你: 回测网格策略效果怎么样？
你: 分析一下515290的估值
你: 帮我筛选银行板块低估值股票
你: 用AI分析一下600036招商银行
```

## 配置持仓监控

真实持仓、自选、交易和聊天数据不会提交到 Git。首次使用时复制示例文件：

```bash
copy portfolio.example.json portfolio.json
copy watchlist.example.json watchlist.json
copy trades.example.json trades.json
```

然后编辑 `portfolio.json`：

```json
[
  {
    "code": "515290",
    "name": "银行ETF天弘",
    "cost": 1.392,
    "shares": 1000,
    "stop_loss": 1.25,
    "take_profit": 1.60,
    "first_buy_date": "2026-05-11"
  }
]
```

## 配置消息推送（可选）

### Server酱（推送到微信）
1. 访问 https://sct.ftqq.com/ 注册
2. 获取 SendKey
3. 填入 `.env` 的 `SERVERCHAN_KEY`

### 企业微信机器人
1. 在企业微信群中添加机器人
2. 获取 Webhook URL
3. 填入 `.env` 的 `WECHAT_WEBHOOK`

## 项目结构

```
investment-agent/
├── main.py              # 主入口
├── scheduler.py         # 定时任务调度
├── sector_config.py     # 板块配置（ETF映射、关键词）
├── portfolio.json       # 持仓配置
├── .env                 # 配置文件
├── requirements.txt     # 依赖
├── agents/
│   ├── coordinator.py   # 核心 Agent（对话+调度）
│   └── risk_monitor.py  # 风控监控
├── tools/
│   ├── market_data.py       # 行情数据工具
│   ├── news_fetcher.py      # 新闻抓取工具
│   ├── notify.py            # 消息推送工具
│   ├── position_manager.py  # 仓位管理（网格/加仓/定投）
│   ├── portfolio_tracker.py # 收益追踪（快照/曲线/统计）
│   ├── sector_rotation.py   # 板块轮动（强弱/信号/对比）
│   ├── backtest.py          # 策略回测（均线/网格/定投）
│   └── stock_screener.py    # 选股估值（PE-PB/筛选/AI分析）
├── knowledge/
│   └── rag.py           # RAG 知识库
├── web/
│   ├── app.py           # Flask Web 服务
│   └── templates/
│       └── index.html   # Web 界面
└── data/
    ├── chromadb/            # 知识库存储
    └── portfolio_history.json  # 收益历史记录
```

## Web API 新接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/profit-curve` | GET | 收益曲线数据 (参数: period=week/month/year/all) |
| `/api/profit-curve/record` | POST | 手动记录今日快照 |
| `/api/sector-rotation` | GET | 板块轮动排名 |
| `/api/sector-rotation/history` | GET | 板块近期走势对比 |
| `/api/backtest` | POST | 运行回测 (strategy=ma/grid/dca) |
| `/api/valuation` | GET | 估值分析 (参数: symbol) |
| `/api/screener` | GET | 低估值筛选 (参数: sector) |
| `/api/position/grid` | POST | 网格交易方案 |
| `/api/position/add-calc` | POST | 加仓计算 |

## 定时任务

启动 `python main.py --schedule` 后自动执行：

| 时间 | 任务 |
|------|------|
| 每日 9:20 | 盘前市场概览 |
| 交易日 每30分钟 | 风险检查 |
| 每日 15:10 | 记录持仓快照（收益曲线数据源）|
| 每日 17:00 | 生成每日投资简报 |

## 进阶扩展

- 添加更多数据源（同花顺、雪球等）
- 接入量化回测框架（backtrader）
- 增加 AI 选股策略
- 接入交易接口（模拟盘）
- 部署到服务器 24 小时运行


## 上市公司预期变化研究（MVP）

新增 `research_system/`，目标不是自动推荐股票，而是完成：信息 → 事实 → 行业/公司变化 → 故事变化 → 市场预期 → 盈利与估值 → 反证 → 概率化审判。

### 架构

固定 DAG：`Research → Financial → Industry → Company → Story Evolution → Price-Implied Expectations → Programmatic Valuation → Catalyst → Red Team → Judge`。所有 Agent 使用 DeepSeek，Agent 间只传 Pydantic Schema。系统显式维护投资逻辑树、财务兑现树、风险树三棵可审查逻辑树，分别描述为什么可能上涨、收入到 EPS 的兑现路径，以及独立的下行风险路径。

系统采用“Research Engine Engineering”分工：LLM 负责理解事实、提出增长率/利润率/费用率/估值倍数等假设及解释；Python 计算引擎负责当前价格隐含盈利增长路径、三年 Revenue→Gross Profit→Opex→Net Profit 财务桥、估值和概率；SQLite 负责保存历史阶段状态、Evidence 与权威预期序列。预期比较固定为三条预测序列：**Model-Implied（模型反推）**、**Market Consensus（可靠市场一致预期）**、**Agent Forecast（估值引擎 Base 情景）**，并可附 Actual 实际值用于事后校验；旧 `expectation_timeline` 仅作兼容。价格隐含路径明确是给定要求回报率和退出倍数假设下的反推结果，不冒充市场一致预期。

硬规则包括：所有增长率、利润率、费用率、要求回报率和概率统一使用 `0..1` 小数约定；情景调整概率按 `base_probability × evidence_factor` 计算，再对 Bear/Base/Bull 归一化。净利润大于0才允许 PE，亏损时自动切换 EV/EBITDA、EV/Sales 或 PB；核心当前结论必须满足“当前有效 A 级证据至少1条，或来自两个独立来源的当前有效 B 级证据至少2条”。Evidence 的 `available_at` 不得晚于 `analysis_date`，否则触发 Look-ahead Bias 失效；市场一致预期还必须通过来源独立性、时点可用性与 Market Consistency Gate，未通过时不得保存为可用共识。故事概率按 DAG 祖先闭包中的 `P(A) × P(B|A) × P(C|A,B)…` 联合计算，共享祖先只计算一次；Judge 必须同时回答“市场为什么可能低估”和“Agent 最可能错在哪里”。系统同时保留可观察 Catalyst 与因果链 Red Team，并输出 Bear/Base/Bull 各情景目标价、价格回报率，以及概率加权目标价和预期回报率。

### 配置与启动

1. 执行 `.venv\Scripts\pip.exe install -r requirements.txt`。
2. 在 `.env` 配置 `DEEPSEEK_API_KEY`；可选配置 `TAVILY_API_KEY` 以补充公告、研报、政策和产业链检索。
3. 双击项目根目录的 `start.bat`（或桌面快捷启动脚本）。
4. DataBoard 桌面窗口会自动打开；公司研究位于“公司研究”页签。系统只监听 `127.0.0.1:5000`，无需单独启动 API 或 Streamlit。

研究接口集成在原 DataBoard：

- `POST /api/company-research/tasks`：提交后台研究任务
- `GET /api/company-research/tasks/<task_id>`：查询任务进度
- `GET /api/company-research/analyses`：查询历史研究
- `GET /api/company-research/analyses/<analysis_id>`：读取完整研究结果

研究数据库位于 `data/research.db`（已被 `.gitignore` 排除）。输出仅供研究，不构成投资建议，也不包含自动交易能力。
