"""
协调者 Agent - 基于 LangGraph 的现代实现
"""
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

from tools import ALL_TOOLS
from knowledge.rag import search_knowledge, add_investment_note


# 知识库工具
@tool
def search_my_knowledge(query: str) -> str:
    """在我的个人投资知识库中搜索相关信息（包括笔记、研报、历史分析等）。"""
    results = search_knowledge(query, top_k=3)
    if not results:
        return "知识库中暂无相关内容。"
    output = "=== 知识库检索结果 ===\n\n"
    for i, doc in enumerate(results, 1):
        meta = doc["metadata"]
        output += f"【{i}】 {meta.get('category', meta.get('type', '文档'))}\n"
        if meta.get("date"):
            output += f"    日期: {meta['date']}\n"
        if meta.get("stock_code"):
            output += f"    相关股票: {meta['stock_code']}\n"
        output += f"    内容: {doc['content'][:200]}\n\n"
    return output


@tool
def save_note(note: str, stock_code: str = "", category: str = "笔记") -> str:
    """保存投资笔记到知识库。
    参数:
    - note: 笔记内容
    - stock_code: 相关股票代码（可选）
    - category: 分类，如 '笔记', '策略', '复盘'
    """
    doc_id = add_investment_note(note, stock_code, category)
    return f"笔记已保存（ID: {doc_id}）"


def _load_holdings_text():
    """从 portfolio.json 读取真实持仓，生成提示文本，避免在 prompt 里写死。"""
    import json
    pf = os.path.join(os.path.dirname(os.path.dirname(__file__)), "portfolio.json")
    try:
        with open(pf, "r", encoding="utf-8") as f:
            items = json.load(f)
    except Exception:
        items = []
    if not items:
        return "（当前无持仓，如需分析请让用户先在看板添加持仓）"
    lines = []
    for it in items:
        lines.append(
            f"- {it.get('code','')} {it.get('name','')}："
            f"{it.get('shares','?')}份，成本{it.get('cost','?')}元"
        )
    return "\n".join(lines)


SYSTEM_PROMPT = """你是一个专业的A股投资分析助手。

当前时间：{current_time}

用户的持仓（来自实时持仓数据）：
{holdings}

你拥有以下能力：
1. 实时行情查询、技术分析
2. 财经新闻、板块资金流向
3. 仓位管理：网格交易计算(calculate_grid_trading)、加仓分析(calculate_add_position)、定投模拟(simulate_dca)
4. 收益追踪：记录快照(record_portfolio_snapshot)、收益曲线(get_profit_curve)、收益统计(get_profit_stats)
5. 板块轮动：强弱排名(get_sector_strength_ranking)、轮动信号(detect_rotation_signal)、板块对比(compare_sectors)
6. 回测：均线策略(backtest_ma_strategy)、网格策略(backtest_grid_strategy)、定投策略(backtest_dca_strategy)
7. 选股估值：估值分析(analyze_valuation)、低估筛选(screen_undervalued_stocks)、AI综合分析(ai_stock_analysis)

当用户问到"我的持仓"、"现在要不要操作"、"适合做T吗"等问题时，你必须：
1. 先调用 get_stock_realtime 查询上面持仓列表里每只标的的实时行情
2. 再调用 get_stock_technical_indicators 查看技术指标
3. 再调用 get_market_overview 看大盘环境
4. 综合以上数据给出明确的操作建议

当用户问到"板块轮动"、"哪个板块强"时，调用 get_sector_strength_ranking 和 detect_rotation_signal。
当用户问到"帮我回测"时，根据策略类型选择对应回测工具。
当用户问到"估值分析"、"低估值"时，使用 analyze_valuation 或 screen_undervalued_stocks。
当用户问到"网格"、"加仓"、"定投"时，使用仓位管理工具。

回答格式：
📊 持仓状态：（每只ETF的现价、盈亏）
📈 技术面：（RSI、均线方向、MACD状态）
🏦 大盘环境：（大盘涨跌、整体氛围）
💡 操作建议：（明确说该买/卖/持有，以及原因）
⚠️ 风险提示

当用户问到具体个股（非ETF）的分析时，使用多空辩论框架：
1. 先获取数据（行情+技术指标+基本面）
2. 列出看多理由（3条以内）
3. 列出看空理由（3条以内）
4. 综合双方给出最终判断

做T建议规则：
- 如果日内波幅超过1%，且有明显高低点，可以建议做T
- 如果波幅很小（<0.5%），建议不做T
- 做T方向：低开高走适合先买后卖，高开低走适合先卖后买
- 重要：如果当前时间在9:30-10:00之间，不要基于当前日内波幅做判断（开盘太短数据不充分），应该参考前几日的波幅来预估今日空间

原则：
- 先获取数据再分析，不要凭空推测
- 给明确建议，不模棱两可
- 简短有力
- 始终提醒风险
- 推荐个股时，必须用工具查询实时数据验证价格和涨跌幅，不要凭记忆编造数据。如果无法验证，明确告知用户"数据未经验证，仅供参考"

格式要求：
- 用简洁的小标题和短句，不要长段落
- 用 - 列表，不要用表格
- 不要用 emoji
- 标题用 ### 不要用 # 或 ##
- 每个要点一句话说完"""


def create_investment_agent():
    """创建投资分析 Agent"""
    from datetime import datetime

    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
        temperature=0.3,
    )

    tools = ALL_TOOLS + [search_my_knowledge, save_note]

    # Inject current time + 实时持仓 into prompt（不再写死持仓）
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M (周%w)")
    prompt = SYSTEM_PROMPT.format(current_time=current_time, holdings=_load_holdings_text())

    agent = create_react_agent(
        llm,
        tools,
        prompt=prompt,
    )
    return agent


# Conversation history for context continuity
# 注意：这是单用户本地看板的简化实现，用锁保证多请求下不并发写坏；
# 如需多用户隔离，应改为按 session_id 维护各自历史。
import threading
_conversation_history = []
_history_lock = threading.Lock()
_chat_lock = threading.Lock()  # 单用户会话串行化，保证上下文顺序并避免并发重型调用
MAX_HISTORY = 6  # Keep last 6 messages (3 rounds of Q&A)


def _chat_unlocked(agent, user_input: str) -> str:
    """与 Agent 对话（带上下文记忆 + 实时时间）"""
    from langchain_core.messages import AIMessage, SystemMessage
    from datetime import datetime

    # Inject current time as a system message
    time_msg = SystemMessage(content=f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}。如果在9:30-10:00之间，日内数据不充分，不要用今日波幅/成交额做结论。")

    # Build messages with history (快照，避免 invoke 期间被其他请求修改)
    with _history_lock:
        history_snapshot = list(_conversation_history)
    messages = [time_msg] + history_snapshot + [HumanMessage(content=user_input)]

    result = agent.invoke({"messages": messages})

    # Extract AI response
    response_content = ""
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
            response_content = msg.content
            break
    if not response_content:
        for msg in reversed(result["messages"]):
            if hasattr(msg, "content") and msg.content:
                response_content = msg.content
                break

    if not response_content:
        response_content = "抱歉，未能生成回答。"

    # Save to history（加锁）
    with _history_lock:
        _conversation_history.append(HumanMessage(content=user_input))
        _conversation_history.append(AIMessage(content=response_content))
        # Trim history to prevent token overflow
        while len(_conversation_history) > MAX_HISTORY:
            _conversation_history.pop(0)

    return response_content


def chat(agent, user_input: str) -> str:
    """串行执行单用户对话，避免历史快照交错和重复占用模型资源。"""
    with _chat_lock:
        return _chat_unlocked(agent, user_input)
