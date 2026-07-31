from .risk_monitor import RiskMonitor


def create_investment_agent():
    from .coordinator import create_investment_agent as _create
    return _create()


def chat(agent, user_input: str) -> str:
    from .coordinator import chat as _chat
    return _chat(agent, user_input)
