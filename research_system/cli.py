import argparse
import json

from dotenv import load_dotenv

from .orchestrator import ResearchOrchestrator
from .schemas import ResearchRequest


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="上市公司预期变化研究")
    parser.add_argument("company", help="公司名称或股票代码")
    parser.add_argument("--symbol", default="", help="六位股票代码")
    parser.add_argument("--deep", action="store_true", help="启用深度搜索")
    args = parser.parse_args()
    request = ResearchRequest(
        company=args.company,
        symbol=args.symbol,
        search_depth="deep" if args.deep else "basic",
    )
    result = ResearchOrchestrator().run(request)
    print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
