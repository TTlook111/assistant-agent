from langchain.tools import tool
from typing import List

@tool("send_email", description="通过邮件 API 发送电子邮件。要求使用格式正确的电子邮件地址。")
def send_email(
    to: List[str],  # 电子邮件地址列表
    subject: str,
    body: str,
    cc: List[str] = []
) -> str:
    # 占位符：在实际应用中，这里会调用 SendGrid, Gmail API 等。
    return f"邮件已发送至 {', '.join(to)} - 主题: {subject}"