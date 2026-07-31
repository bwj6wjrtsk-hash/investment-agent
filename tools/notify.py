"""
消息推送工具 - 支持 Server酱 和 企业微信机器人
"""
import os
import requests
from langchain_core.tools import tool


def send_serverchan(title: str, content: str) -> bool:
    """通过 Server酱 推送消息到微信"""
    key = os.getenv("SERVERCHAN_KEY")
    if not key:
        return False
    url = f"https://sctapi.ftqq.com/{key}.send"
    data = {"title": title, "desp": content}
    resp = requests.post(url, data=data, timeout=10)
    return resp.status_code == 200


def send_wechat_bot(content: str) -> bool:
    """通过企业微信机器人推送消息"""
    webhook = os.getenv("WECHAT_WEBHOOK")
    if not webhook:
        return False
    data = {
        "msgtype": "markdown",
        "markdown": {"content": content}
    }
    resp = requests.post(webhook, json=data, timeout=10)
    return resp.status_code == 200


@tool
def send_notification(title: str, content: str) -> str:
    """发送通知消息。会尝试通过 Server酱 或企业微信机器人推送。
    参数:
    - title: 消息标题
    - content: 消息内容（支持 Markdown）
    """
    success = False

    if os.getenv("SERVERCHAN_KEY"):
        success = send_serverchan(title, content)
        if success:
            return f"通知已通过 Server酱 发送: {title}"

    if os.getenv("WECHAT_WEBHOOK"):
        success = send_wechat_bot(f"## {title}\n\n{content}")
        if success:
            return f"通知已通过企业微信发送: {title}"

    if not success:
        # 没配置推送渠道，打印到控制台
        print(f"\n{'='*50}")
        print(f"📢 通知: {title}")
        print(f"{'='*50}")
        print(content)
        print(f"{'='*50}\n")
        return f"通知已输出到控制台（未配置推送渠道）: {title}"
