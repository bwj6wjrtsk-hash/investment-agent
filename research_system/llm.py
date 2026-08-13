import json
import os
import re
from typing import TypeVar

from langchain_openai import ChatOpenAI
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class DeepSeekClient:
    def __init__(self) -> None:
        api_key = os.getenv("DEEPSEEK_API_KEY", "")
        if not api_key or api_key == "your_api_key_here":
            raise RuntimeError("请在 .env 中配置 DEEPSEEK_API_KEY")
        self.llm = ChatOpenAI(
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            api_key=api_key,
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            temperature=0.1,
            timeout=120,
            max_retries=2,
            max_tokens=int(os.getenv("DEEPSEEK_MAX_TOKENS", "8192")),
        )

    def generate(self, schema: type[T], system: str, payload: dict) -> T:
        schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False)
        base_prompt = (
            f"{system}\n\n只返回一个合法、完整、可解析的 JSON 对象，不要 Markdown。"
            f"严格符合此 JSON Schema：{schema_json}\n"
            "保持内容精炼、去除重复，单条文字尽量不超过120字；所有数组和对象必须完整闭合。\n\n输入："
            f"{json.dumps(payload, ensure_ascii=False, default=str)}"
        )
        prompt = base_prompt
        last_error: Exception | None = None
        for attempt in range(2):
            response = self.llm.invoke(prompt)
            text = response.content if isinstance(response.content, str) else str(response.content)
            try:
                start = text.find("{")
                end = text.rfind("}")
                if start < 0 or end <= start:
                    raise ValueError("模型未返回完整 JSON 对象")
                return schema.model_validate_json(text[start:end + 1])
            except Exception as exc:
                last_error = exc
                if attempt == 0:
                    prompt = (
                        base_prompt
                        + "\n\n上一次输出无法通过 Schema 校验，可能被截断或字段格式错误。"
                        "请重新生成更精炼的完整 JSON，务必闭合所有括号，不要解释。"
                        f"校验错误摘要：{str(exc)[:500]}"
                    )
        raise ValueError(f"模型连续两次未返回符合 Schema 的完整 JSON：{last_error}") from last_error
