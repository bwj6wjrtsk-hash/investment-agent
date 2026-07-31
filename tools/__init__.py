from .market_data import (
    get_stock_realtime,
    get_stock_history,
    get_stock_technical_indicators,
    get_market_overview,
    get_stock_fundamental,
)
from .news_fetcher import (
    get_financial_news,
    get_stock_news,
    get_sector_fund_flow,
    get_hot_stocks,
)
from .notify import send_notification
from .position_manager import (
    calculate_grid_trading,
    calculate_add_position,
    simulate_dca,
)
from .portfolio_tracker import (
    record_portfolio_snapshot,
    get_profit_curve,
    get_profit_stats,
)
from .sector_rotation import (
    get_sector_strength_ranking,
    detect_rotation_signal,
    compare_sectors,
)
from .backtest import (
    backtest_ma_strategy,
    backtest_grid_strategy,
    backtest_dca_strategy,
)
from .stock_screener import (
    analyze_valuation,
    screen_undervalued_stocks,
    ai_stock_analysis,
)

ALL_TOOLS = [
    # 行情数据
    get_stock_realtime,
    get_stock_history,
    get_stock_technical_indicators,
    get_market_overview,
    get_stock_fundamental,
    # 新闻
    get_financial_news,
    get_stock_news,
    get_sector_fund_flow,
    get_hot_stocks,
    # 通知
    send_notification,
    # 仓位管理 & 定投
    calculate_grid_trading,
    calculate_add_position,
    simulate_dca,
    # 收益追踪
    record_portfolio_snapshot,
    get_profit_curve,
    get_profit_stats,
    # 板块轮动
    get_sector_strength_ranking,
    detect_rotation_signal,
    compare_sectors,
    # 回测
    backtest_ma_strategy,
    backtest_grid_strategy,
    backtest_dca_strategy,
    # 选股 & 估值
    analyze_valuation,
    screen_undervalued_stocks,
    ai_stock_analysis,
]
