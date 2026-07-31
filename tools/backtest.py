"""
回测工具
- 基于历史数据验证交易策略
- 支持均线策略、网格策略、定投策略
- 输出回测结果和关键指标
"""
import re
import requests
from langchain_core.tools import tool

from tools import quote as _q


def _fetch_klines(symbol: str, days: int = 250) -> list:
    """Fetch historical kline data"""
    s = requests.Session()
    s.trust_env = False
    prefix = _q.code_prefix(symbol)
    resp = s.get(
        f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={prefix},day,,,{days},qfq",
        timeout=10
    )
    data = resp.json()
    klines = data.get("data", {}).get(prefix, {}).get("day", [])
    if not klines:
        klines = data.get("data", {}).get(prefix, {}).get("qfqday", [])
    return klines


def _calc_max_drawdown(values: list) -> float:
    """Calculate max drawdown from a list of portfolio values"""
    peak = values[0]
    max_dd = 0
    for v in values:
        if v > peak:
            peak = v
        dd = (v / peak - 1) * 100
        if dd < max_dd:
            max_dd = dd
    return max_dd


@tool
def backtest_ma_strategy(symbol: str, short_period: int = 5, long_period: int = 20, initial_capital: float = 10000.0) -> str:
    """回测均线交叉策略（金叉买入，死叉卖出）。
    参数:
    - symbol: 股票/ETF代码
    - short_period: 短期均线周期（默认5日）
    - long_period: 长期均线周期（默认20日）
    - initial_capital: 初始资金（默认10000元）
    """
    try:
        klines = _fetch_klines(symbol, 250)
        if not klines or len(klines) < long_period + 10:
            return f"历史数据不足（需要至少{long_period+10}天），无法回测"

        closes = [float(k[2]) for k in klines]
        dates = [k[0] for k in klines]

        result = f"=== {symbol} 均线交叉策略回测 ===\n"
        result += f"策略: MA{short_period} 上穿 MA{long_period} 买入，下穿卖出\n"
        result += f"回测区间: {dates[0]} ~ {dates[-1]} ({len(dates)}个交易日)\n"
        result += f"初始资金: {initial_capital:.0f}元\n\n"

        # 回测逻辑
        cash = initial_capital
        shares = 0
        trades = []
        portfolio_values = []

        for i in range(long_period, len(closes)):
            ma_short = sum(closes[i-short_period+1:i+1]) / short_period
            ma_long = sum(closes[i-long_period+1:i+1]) / long_period
            prev_ma_short = sum(closes[i-short_period:i]) / short_period
            prev_ma_long = sum(closes[i-long_period:i]) / long_period

            current_price = closes[i]
            portfolio_value = cash + shares * current_price
            portfolio_values.append(portfolio_value)

            # 金叉买入
            if prev_ma_short <= prev_ma_long and ma_short > ma_long and shares == 0:
                shares = int(cash / current_price / 100) * 100
                if shares >= 100:
                    cost = shares * current_price
                    cash -= cost
                    trades.append({"date": dates[i], "action": "买入", "price": current_price, "shares": shares})

            # 死叉卖出
            elif prev_ma_short >= prev_ma_long and ma_short < ma_long and shares > 0:
                revenue = shares * current_price
                profit = revenue - trades[-1]["price"] * shares if trades else 0
                cash += revenue
                trades.append({"date": dates[i], "action": "卖出", "price": current_price, "shares": shares, "profit": profit})
                shares = 0

        # 最终清仓
        final_value = cash + shares * closes[-1]
        total_return = (final_value / initial_capital - 1) * 100

        # 对比买入持有
        buy_hold_return = (closes[-1] / closes[long_period] - 1) * 100

        # 统计
        buy_trades = [t for t in trades if t["action"] == "买入"]
        sell_trades = [t for t in trades if t["action"] == "卖出"]
        win_trades = [t for t in sell_trades if t.get("profit", 0) > 0]

        result += f"【回测结果】\n"
        result += f"  最终资产: {final_value:.2f}元\n"
        result += f"  总收益率: {total_return:+.2f}%\n"
        result += f"  买入持有收益: {buy_hold_return:+.2f}%\n"
        result += f"  策略超额收益: {total_return - buy_hold_return:+.2f}%\n"

        if portfolio_values:
            max_dd = _calc_max_drawdown(portfolio_values)
            result += f"  最大回撤: {max_dd:.2f}%\n"

        result += f"\n【交易统计】\n"
        result += f"  总交易次数: {len(buy_trades)}买 + {len(sell_trades)}卖\n"
        if sell_trades:
            result += f"  胜率: {len(win_trades)}/{len(sell_trades)} ({len(win_trades)/len(sell_trades)*100:.0f}%)\n"

        result += f"\n【交易明细】\n"
        for t in trades[-10:]:  # 最多显示10条
            if t["action"] == "买入":
                result += f"  {t['date']} 买入 {t['shares']}份 @ {t['price']:.3f}\n"
            else:
                emoji = "✅" if t.get("profit", 0) > 0 else "❌"
                result += f"  {t['date']} 卖出 {t['shares']}份 @ {t['price']:.3f} {emoji} {t.get('profit',0):+.0f}元\n"

        if len(trades) > 10:
            result += f"  ... (共{len(trades)}条交易)\n"

        result += f"\n💡 结论: {'策略优于买入持有' if total_return > buy_hold_return else '买入持有更优'}"
        return result
    except Exception as e:
        return f"回测失败: {e}"


@tool
def backtest_grid_strategy(symbol: str, grid_pct: float = 3.0, initial_capital: float = 10000.0) -> str:
    """回测网格交易策略。
    参数:
    - symbol: 股票/ETF代码
    - grid_pct: 网格间距百分比（默认3%）
    - initial_capital: 初始资金（默认10000元）
    """
    try:
        klines = _fetch_klines(symbol, 250)
        if not klines or len(klines) < 30:
            return "历史数据不足，无法回测"

        closes = [float(k[2]) for k in klines]
        dates = [k[0] for k in klines]

        result = f"=== {symbol} 网格策略回测 ===\n"
        result += f"策略: 每下跌{grid_pct}%加仓，每上涨{grid_pct}%减仓\n"
        result += f"回测区间: {dates[0]} ~ {dates[-1]} ({len(dates)}个交易日)\n"
        result += f"初始资金: {initial_capital:.0f}元\n\n"

        # 初始建仓50%
        cash = initial_capital
        base_price = closes[0]
        initial_shares = int((cash * 0.5) / base_price / 100) * 100
        if initial_shares < 100:
            initial_shares = 100
        cash -= initial_shares * base_price
        shares = initial_shares
        last_trade_price = base_price

        trades = []
        trades.append({"date": dates[0], "action": "建仓", "price": base_price, "shares": initial_shares})
        portfolio_values = []
        total_grid_profit = 0

        for i in range(1, len(closes)):
            current_price = closes[i]
            portfolio_values.append(cash + shares * current_price)

            change_from_last = (current_price / last_trade_price - 1) * 100

            # 下跌超过grid_pct，买入
            if change_from_last <= -grid_pct and cash > current_price * 100:
                buy_shares = int(min(cash * 0.3, initial_capital * 0.1) / current_price / 100) * 100
                if buy_shares >= 100:
                    cash -= buy_shares * current_price
                    shares += buy_shares
                    last_trade_price = current_price
                    trades.append({"date": dates[i], "action": "网格买入", "price": current_price, "shares": buy_shares})

            # 上涨超过grid_pct，卖出
            elif change_from_last >= grid_pct and shares > 100:
                sell_shares = min(int(shares * 0.2 / 100) * 100, shares - 100)
                if sell_shares >= 100:
                    revenue = sell_shares * current_price
                    grid_profit = sell_shares * (current_price - last_trade_price)
                    total_grid_profit += grid_profit
                    cash += revenue
                    shares -= sell_shares
                    last_trade_price = current_price
                    trades.append({"date": dates[i], "action": "网格卖出", "price": current_price, "shares": sell_shares, "profit": grid_profit})

        # 最终结果
        final_value = cash + shares * closes[-1]
        total_return = (final_value / initial_capital - 1) * 100
        buy_hold_return = (closes[-1] / closes[0] - 1) * 100

        buy_count = len([t for t in trades if "买" in t["action"]])
        sell_count = len([t for t in trades if "卖" in t["action"]])

        result += f"【回测结果】\n"
        result += f"  最终资产: {final_value:.2f}元 (现金{cash:.0f} + 持仓{shares*closes[-1]:.0f})\n"
        result += f"  总收益率: {total_return:+.2f}%\n"
        result += f"  网格利润: {total_grid_profit:+.0f}元\n"
        result += f"  买入持有收益: {buy_hold_return:+.2f}%\n"
        result += f"  策略超额收益: {total_return - buy_hold_return:+.2f}%\n"

        if portfolio_values:
            max_dd = _calc_max_drawdown(portfolio_values)
            result += f"  最大回撤: {max_dd:.2f}%\n"

        result += f"\n【交易统计】\n"
        result += f"  网格买入: {buy_count}次\n"
        result += f"  网格卖出: {sell_count}次\n"
        result += f"  当前持仓: {shares}份\n"
        result += f"  剩余现金: {cash:.0f}元\n"

        result += f"\n【最近交易】\n"
        for t in trades[-8:]:
            result += f"  {t['date']} {t['action']} {t['shares']}份 @ {t['price']:.3f}"
            if "profit" in t:
                result += f" (利润{t['profit']:+.0f})"
            result += "\n"

        # 适用性分析
        price_range = (max(closes) - min(closes)) / min(closes) * 100
        result += f"\n💡 分析:\n"
        result += f"  价格波动范围: {price_range:.1f}%\n"
        if price_range < grid_pct * 3:
            result += f"  波动太小，网格间距建议缩小到 {price_range/5:.1f}%\n"
        elif total_return > buy_hold_return:
            result += f"  网格策略表现优于持有，适合震荡市\n"
        else:
            result += f"  单边行情中网格不如持有，适合震荡区间操作\n"

        return result
    except Exception as e:
        return f"回测失败: {e}"


@tool
def backtest_dca_strategy(symbol: str, interval_days: int = 22, amount_per_time: float = 2000.0) -> str:
    """回测定投策略。
    参数:
    - symbol: 股票/ETF代码
    - interval_days: 定投间隔天数（默认22天≈每月）
    - amount_per_time: 每次投入金额（默认2000元）
    """
    try:
        klines = _fetch_klines(symbol, 500)
        if not klines or len(klines) < 60:
            return "历史数据不足，无法回测"

        closes = [float(k[2]) for k in klines]
        dates = [k[0] for k in klines]

        result = f"=== {symbol} 定投策略回测 ===\n"
        result += f"策略: 每{interval_days}个交易日投入{amount_per_time:.0f}元\n"
        result += f"回测区间: {dates[0]} ~ {dates[-1]} ({len(dates)}个交易日)\n\n"

        total_cost = 0
        total_shares = 0
        trades = []
        portfolio_values = []

        for i in range(0, len(closes), interval_days):
            price = closes[i]
            buy_shares = int(amount_per_time / price / 100) * 100
            if buy_shares < 100:
                buy_shares = 100
            actual_cost = buy_shares * price
            total_cost += actual_cost
            total_shares += buy_shares
            trades.append({"date": dates[i], "price": price, "shares": buy_shares})

            # 记录每个定投日的组合价值
            portfolio_values.append(total_shares * price)

        # 最终结果
        final_price = closes[-1]
        final_value = total_shares * final_price
        avg_cost = total_cost / total_shares
        total_return = (final_value / total_cost - 1) * 100

        # 买入持有对比
        buy_hold_shares = int(total_cost / closes[0] / 100) * 100
        buy_hold_value = buy_hold_shares * final_price
        buy_hold_return = (buy_hold_value / (buy_hold_shares * closes[0]) - 1) * 100

        result += f"【定投结果】\n"
        result += f"  定投次数: {len(trades)}次\n"
        result += f"  总投入: {total_cost:.0f}元\n"
        result += f"  总份额: {total_shares}份\n"
        result += f"  平均成本: {avg_cost:.3f}\n"
        result += f"  最终市值: {final_value:.0f}元\n"
        result += f"  总收益: {final_value-total_cost:+.0f}元 ({total_return:+.2f}%)\n"

        if portfolio_values:
            max_dd = _calc_max_drawdown(portfolio_values)
            result += f"  最大回撤: {max_dd:.2f}%\n"

        result += f"\n【对比买入持有】\n"
        result += f"  同等资金一次性买入: {buy_hold_return:+.2f}%\n"
        result += f"  定投 vs 持有: {'定投更优' if total_return > buy_hold_return else '持有更优'} (差{abs(total_return-buy_hold_return):.2f}%)\n"

        # 定投效果分析
        result += f"\n【成本分布】\n"
        prices_at_buy = [t["price"] for t in trades]
        result += f"  最低买入价: {min(prices_at_buy):.3f}\n"
        result += f"  最高买入价: {max(prices_at_buy):.3f}\n"
        result += f"  平均买入价: {sum(prices_at_buy)/len(prices_at_buy):.3f}\n"
        result += f"  当前价格: {final_price:.3f}\n"

        result += f"\n💡 定投适合波动大、长期向上的标的。银行ETF波动小，定投平滑效果有限。"
        return result
    except Exception as e:
        return f"回测失败: {e}"
