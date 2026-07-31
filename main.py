"""
DataBoard - Main Entry
"""
import os
import sys
import argparse
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(__file__))


def interactive_mode():
    from agents import create_investment_agent, chat

    print("=" * 60)
    print("DataBoard - Interactive Mode")
    print("=" * 60)
    print()
    print("Type 'quit' or 'exit' to quit")
    print("=" * 60)
    print()

    if not os.getenv("DEEPSEEK_API_KEY") or os.getenv("DEEPSEEK_API_KEY") == "your_api_key_here":
        print("Please configure DEEPSEEK_API_KEY in .env file")
        print()
        return

    executor = create_investment_agent()
    print("Ready.\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                print("\nBye!")
                break

            print("\nAnalyzing...\n")
            response = chat(executor, user_input)
            print(f"\nAssistant:\n{response}\n")
            print("-" * 40 + "\n")

        except KeyboardInterrupt:
            print("\n\nBye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


def report_mode():
    from scheduler import generate_daily_report
    generate_daily_report()


def monitor_mode():
    from agents import RiskMonitor

    monitor = RiskMonitor()
    if not monitor.portfolio:
        print("No portfolio configured. Use add_position() to add positions.")
        return

    print(monitor.get_portfolio_summary())
    print()

    alerts = monitor.check_all()
    for alert in alerts:
        level = alert["level"].upper()
        print(f"[{level}] {alert.get('stock', '')} {alert['message']}")
        if alert.get("action"):
            print(f"  -> {alert['action']}")
        print()


def schedule_mode():
    from scheduler import start_scheduler
    start_scheduler()


def web_mode(port=5000):
    from web.app import app
    # 默认仅绑定本机，避免局域网内任何人访问无鉴权的写接口
    host = os.getenv("DASHBOARD_HOST", "127.0.0.1")
    app.run(host=host, port=port, debug=False)


def main():
    parser = argparse.ArgumentParser(description="DataBoard")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--monitor", action="store_true")
    parser.add_argument("--schedule", action="store_true")
    parser.add_argument("--web", action="store_true")
    parser.add_argument("--port", type=int, default=5000)

    args = parser.parse_args()

    if args.web:
        web_mode(args.port)

    elif args.report:
        report_mode()
    elif args.monitor:
        monitor_mode()
    elif args.schedule:
        schedule_mode()
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
