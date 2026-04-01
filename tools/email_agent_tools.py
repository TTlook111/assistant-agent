import json
import os
from langchain.tools import tool
from typing import List

EMAIL_DB_FILE = "email_db.json"

def _save_email(email_record):
    events = []
    if os.path.exists(EMAIL_DB_FILE):
        with open(EMAIL_DB_FILE, "r", encoding="utf-8") as f:
            events = json.load(f)
            
    events.append(email_record)
    with open(EMAIL_DB_FILE, "w", encoding="utf-8") as f:
        json.dump(events, f, ensure_ascii=False, indent=4)

@tool("send_email")
def send_email(
    to: List[str],  # 电子邮件地址列表
    subject: str,
    body: str,
    cc: List[str] = []
) -> str:
    """
    通过邮件 API 发送电子邮件。要求使用格式正确的电子邮件地址。
    
    参数:
        to: 收件人的电子邮件地址列表。
        subject: 邮件主题。
        body: 邮件正文内容。
        cc: 抄送人的电子邮件地址列表（可选）。
    """
    email_record = {
        "to": to,
        "cc": cc,
        "subject": subject,
        "body": body
    }
    _save_email(email_record)
    print(f"\n[Email Tool] 📧 成功模拟发送邮件并保存记录，主题: {subject}")
    return f"邮件已发送至 {', '.join(to)} - 主题: {subject}"