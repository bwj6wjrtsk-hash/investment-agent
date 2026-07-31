"""
定时任务调度器 - 自动执行日报生成、风险监控等
"""
import os
import sys
from datetime import datetime
from apscheduler.schedulers.blocking import BlockingScheduler
from dotenv import load_dotenv

load_dotenv()

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(__file__))

from agents import create_investment_agent, chat, RiskMonitor
from tools.notify import send_serverchan, send_wechat_bot
from tools.portfolio_tracker import record_daily_snapshot


def generate_daily_report():
    """生成每日投资简报（每天 17:00 执行）"""
    print(f"\n[{datetime.now()}] 开始生成每日投资简报...")

    executor = create_investment_agent()
    report = chat(executor, """
        请生成今日投资简报，包含：
        1. 大盘指数表现（上证、深证、创业板）
        2. 涨跌家数统计
        3. 成交额 TOP5
        4. 行业板块资金流向前5
        5. 今日重要财经新闻（3-5条）
        6. 简要总结今日市场情绪
    """)

    print(report)

    # 推送通知
    title = f"📊 {datetime.now().strftime('%m月%d日')} A股日报"
    if os.getenv("SERVERCHAN_KEY"):
        send_serverchan(title, report)
    elif os.getenv("WECHAT_WEBHOOK"):
        send_wechat_bot(f"## {title}\n\n{report}")


# 预警冷却记录：{alert_key: {"level": level, "date": date_str}}
_alert_cooldown = {}


def _should_notify(alert) -> bool:
    """判断是否需要发送通知（同一预警当天只发一次，除非级别升级）"""
    global _alert_cooldown
    today = datetime.now().strftime("%Y-%m-%d")

    # 生成唯一 key：股票 + 预警类型关键词
    stock = alert.get("stock", "unknown")
    # 用 message 前缀区分类型（止损/止盈/大跌/大涨）
    msg = alert.get("message", "")
    if "止损" in msg:
        alert_type = "stop_loss"
    elif "止盈" in msg:
        alert_type = "take_profit"
    elif "下跌" in msg:
        alert_type = "drop"
    elif "上涨" in msg:
        alert_type = "rise"
    else:
        alert_type = "other"

    key = f"{stock}_{alert_type}"

    prev = _alert_cooldown.get(key)
    if prev and prev["date"] == today:
        # 同一天已发过，只有级别升级才再发
        level_priority = {"info": 0, "warning": 1, "critical": 2}
        if level_priority.get(alert["level"], 0) <= level_priority.get(prev["level"], 0):
            return False

    # 记录本次发送
    _alert_cooldown[key] = {"level": alert["level"], "date": today}
    return True


def check_risk():
    """风险检查（交易时间每30分钟执行一次）"""
    monitor = RiskMonitor()

    # 在这里配置你的持仓
    # monitor.add_position("000001", "平安银行", 12.5, 1000, stop_loss=11.25, take_profit=15.0)

    if not monitor.portfolio:
        return

    alerts = monitor.check_all()
    critical_alerts = [a for a in alerts if a["level"] in ("critical", "warning")]

    # 过滤：只发今天还没通知过的预警
    new_alerts = [a for a in critical_alerts if _should_notify(a)]

    if new_alerts:
        message = "⚠️ 持仓预警\n\n"
        for alert in new_alerts:
            message += f"[{alert['level'].upper()}] {alert.get('stock', '')}\n"
            message += f"{alert['message']}\n"
            message += f"建议: {alert.get('action', '')}\n\n"

        print(message)
        # 推送预警
        if os.getenv("SERVERCHAN_KEY"):
            send_serverchan("⚠️ 持仓预警", message)
        elif os.getenv("WECHAT_WEBHOOK"):
            send_wechat_bot(message)
    elif critical_alerts:
        print(f"[{datetime.now().strftime('%H:%M')}] 有预警但今日已通知过，跳过推送")


def record_snapshot():
    """记录每日持仓快照（收盘后执行）"""
    print(f"\n[{datetime.now()}] 记录持仓快照...")
    snapshot = record_daily_snapshot()
    if snapshot:
        print(f"  总市值: {snapshot['total_value']:.2f}元, 盈亏: {snapshot['profit_pct']:+.2f}%")
    else:
        print("  无持仓数据")


def start_scheduler():
    """启动定时任务"""
    scheduler = BlockingScheduler()

    # 每个交易日 17:00 生成日报
    scheduler.add_job(
        generate_daily_report,
        'cron',
        day_of_week='mon-fri',
        hour=17,
        minute=0,
        id='daily_report'
    )

    # 交易时间每30分钟检查风险（9:30-15:00）
    scheduler.add_job(
        check_risk,
        'cron',
        day_of_week='mon-fri',
        hour='9-14',
        minute='0,30',
        id='risk_check'
    )

    # 开盘前市场概览（9:20）
    scheduler.add_job(
        lambda: print(chat(create_investment_agent(), "给我看看今天的市场概况和热门股票")),
        'cron',
        day_of_week='mon-fri',
        hour=9,
        minute=20,
        id='morning_overview'
    )

    # 收盘后记录持仓快照（15:10）
    scheduler.add_job(
        record_snapshot,
        'cron',
        day_of_week='mon-fri',
        hour=15,
        minute=10,
        id='portfolio_snapshot'
    )

    print("📅 定时任务已启动:")
    print("  - 每日 9:20 盘前概览")
    print("  - 交易时间每30分钟风险检查")
    print("  - 每日 15:10 记录持仓快照")
    print("  - 每日 17:00 生成日报")
    print("\n按 Ctrl+C 停止\n")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print("\n定时任务已停止")


if __name__ == "__main__":
    start_scheduler()
