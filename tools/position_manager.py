"""
仓位管理 & 定投策略工具
- 网格交易计算
- 加仓/减仓建议
- 定投均价模拟
"""
import json
import os
import requests
import re
from langchain_core.tools import tool

from tools import quote as _quote


def _load_portfolio():
    """Load portfolio from file"""
    portfolio_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "portfolio.json")
    if os.path.exists(portfolio_file):
        with open(portfolio_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def _get_current_price(code: str) -> float:
    """Get current price for a stock/ETF"""
    s = requests.Session()
    s.trust_env = False
    prefix = _quote.code_prefix(code)
    resp = s.get(f"https://qt.gtimg.cn/q={prefix}", timeout=10)
    resp.encoding = "gbk"
    match = re.match(r'v_\w+="(.+)"', resp.text.strip().split(";")[0].strip())
    if match:
        fields = match.group(1).split("~")
        if len(fields) > 3 and fields[3]:
            return float(fields[3])
    return 0


@tool
def calculate_grid_trading(symbol: str, grid_count: int = 5, grid_range_pct: float = 10.0, amount_per_grid: float = 1000.0) -> str:
    """计算网格交易方案。
    参数:
    - symbol: 股票/ETF代码
    - grid_count: 网格数量（默认5档）
    - grid_range_pct: 网格总范围百分比（默认上下各10%）
    - amount_per_grid: 每格买入金额（默认1000元）
    """
    try:
        current_price = _get_current_price(symbol)
        if current_price == 0:
            return f"无法获取 {symbol} 的当前价格"

        upper = current_price * (1 + grid_range_pct / 100)
        lower = current_price * (1 - grid_range_pct / 100)
        grid_step = (upper - lower) / grid_count

        result = f"=== {symbol} 网格交易方案 ===\n\n"
        result += f"当前价格: {current_price:.3f}\n"
        result += f"网格范围: {lower:.3f} ~ {upper:.3f} (±{grid_range_pct}%)\n"
        result += f"网格数量: {grid_count}档\n"
        result += f"每格金额: {amount_per_grid:.0f}元\n"
        result += f"总资金需求: {amount_per_grid * grid_count:.0f}元\n\n"

        result += "【买入网格（价格下跌时买入）】\n"
        for i in range(1, grid_count + 1):
            buy_price = current_price - grid_step * i
            shares = int(amount_per_grid / buy_price / 100) * 100  # 整百份
            if shares < 100:
                shares = 100
            result += f"  第{i}档: 价格 {buy_price:.3f} (跌{grid_step*i/current_price*100:.1f}%) → 买入 {shares}份 (约{buy_price*shares:.0f}元)\n"

        result += "\n【卖出网格（价格上涨时卖出）】\n"
        for i in range(1, grid_count + 1):
            sell_price = current_price + grid_step * i
            result += f"  第{i}档: 价格 {sell_price:.3f} (涨{grid_step*i/current_price*100:.1f}%) → 卖出\n"

        result += f"\n💡 建议: 银行ETF波动较小，可适当缩小网格间距(3-5%)提高收益率"
        return result
    except Exception as e:
        return f"计算网格方案失败: {e}"


@tool
def calculate_add_position(symbol: str, target_cost: float = 0, add_amount: float = 2000.0) -> str:
    """计算加仓后的成本变化。
    参数:
    - symbol: 股票/ETF代码
    - target_cost: 目标成本价（0表示不设目标，只计算当前加仓效果）
    - add_amount: 加仓金额（默认2000元）
    """
    try:
        portfolio = _load_portfolio()
        position = None
        for item in portfolio:
            if item["code"] == symbol:
                position = item
                break

        current_price = _get_current_price(symbol)
        if current_price == 0:
            return f"无法获取 {symbol} 的当前价格"

        result = f"=== {symbol} 加仓分析 ===\n\n"
        result += f"当前价格: {current_price:.3f}\n"

        if position:
            old_cost = position["cost"]
            old_shares = position["shares"]
            old_total = old_cost * old_shares

            # 计算加仓后
            add_shares = int(add_amount / current_price / 100) * 100
            if add_shares < 100:
                add_shares = 100
            actual_add_amount = add_shares * current_price

            new_shares = old_shares + add_shares
            new_cost = (old_total + actual_add_amount) / new_shares

            result += f"\n【当前持仓】\n"
            result += f"  {position['name']}: {old_shares}份 × 成本{old_cost:.3f} = {old_total:.0f}元\n"
            result += f"  当前盈亏: {(current_price/old_cost-1)*100:+.2f}%\n"

            result += f"\n【加仓 {add_amount:.0f}元 后】\n"
            result += f"  新增: {add_shares}份 × {current_price:.3f} = {actual_add_amount:.0f}元\n"
            result += f"  新总: {new_shares}份 × 成本{new_cost:.3f} = {new_cost*new_shares:.0f}元\n"
            result += f"  成本变化: {old_cost:.3f} → {new_cost:.3f} ({(new_cost/old_cost-1)*100:+.2f}%)\n"

            # 如果设定目标成本
            if target_cost > 0 and target_cost < old_cost:
                needed_amount = (old_cost * old_shares - target_cost * old_shares) / (1 - target_cost / current_price)
                needed_shares = int(needed_amount / current_price / 100) * 100
                result += f"\n【达到目标成本 {target_cost:.3f} 需要】\n"
                result += f"  需加仓: {needed_shares}份 ≈ {needed_shares*current_price:.0f}元\n"
        else:
            # 新建仓
            shares = int(add_amount / current_price / 100) * 100
            if shares < 100:
                shares = 100
            result += f"\n【新建仓】\n"
            result += f"  买入: {shares}份 × {current_price:.3f} = {shares*current_price:.0f}元\n"

        return result
    except Exception as e:
        return f"计算加仓失败: {e}"


@tool
def simulate_dca(symbol: str, monthly_amount: float = 2000.0, months: int = 12) -> str:
    """模拟定投策略（基于历史数据回算）。
    参数:
    - symbol: 股票/ETF代码
    - monthly_amount: 每月定投金额（默认2000元）
    - months: 定投月数（默认12个月）
    """
    try:
        s = requests.Session()
        s.trust_env = False
        prefix = _quote.code_prefix(symbol)

        # 获取足够的历史数据
        days_needed = months * 22  # 每月约22个交易日
        resp = s.get(
            f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={prefix},day,,,{days_needed},qfq",
            timeout=10
        )
        data = resp.json()
        klines = data.get("data", {}).get(prefix, {}).get("day", [])
        if not klines:
            klines = data.get("data", {}).get(prefix, {}).get("qfqday", [])

        if not klines or len(klines) < 20:
            return f"历史数据不足，无法模拟定投"

        # 每月取第一个交易日的收盘价作为定投价格
        monthly_prices = []
        last_month = ""
        for k in klines:
            date_str = k[0]  # "2025-01-02"
            month = date_str[:7]
            if month != last_month:
                monthly_prices.append({"date": date_str, "price": float(k[2])})
                last_month = month

        # 取最近 months 个月
        if len(monthly_prices) > months:
            monthly_prices = monthly_prices[-months:]

        result = f"=== {symbol} 定投模拟 ({len(monthly_prices)}个月) ===\n\n"
        result += f"每月投入: {monthly_amount:.0f}元\n\n"

        total_cost = 0
        total_shares = 0
        records = []

        for mp in monthly_prices:
            shares = int(monthly_amount / mp["price"] / 100) * 100
            if shares < 100:
                shares = 100
            actual_cost = shares * mp["price"]
            total_cost += actual_cost
            total_shares += shares
            avg_cost = total_cost / total_shares
            records.append({
                "date": mp["date"],
                "price": mp["price"],
                "shares": shares,
                "total_shares": total_shares,
                "avg_cost": avg_cost,
            })

        # 最终结果
        current_price = _get_current_price(symbol)
        if current_price == 0:
            current_price = monthly_prices[-1]["price"]

        final_value = total_shares * current_price
        profit = final_value - total_cost
        profit_pct = (final_value / total_cost - 1) * 100

        result += "【定投记录】\n"
        for r in records:
            result += f"  {r['date']} | 价格{r['price']:.3f} | 买{r['shares']}份 | 累计{r['total_shares']}份 | 均价{r['avg_cost']:.3f}\n"

        result += f"\n【定投结果】\n"
        result += f"  总投入: {total_cost:.0f}元\n"
        result += f"  总份额: {total_shares}份\n"
        result += f"  平均成本: {total_cost/total_shares:.3f}\n"
        result += f"  当前市值: {final_value:.0f}元\n"
        result += f"  盈亏: {profit:+.0f}元 ({profit_pct:+.2f}%)\n"

        # 对比一次性买入
        one_time_price = monthly_prices[0]["price"]
        one_time_shares = int(total_cost / one_time_price / 100) * 100
        one_time_value = one_time_shares * current_price
        one_time_profit_pct = (one_time_value / (one_time_shares * one_time_price) - 1) * 100

        result += f"\n【对比一次性买入】\n"
        result += f"  如果{monthly_prices[0]['date']}一次买入: 盈亏 {one_time_profit_pct:+.2f}%\n"
        result += f"  定投 vs 一次性: {'定投更优' if profit_pct > one_time_profit_pct else '一次性更优'}\n"

        return result
    except Exception as e:
        return f"模拟定投失败: {e}"
