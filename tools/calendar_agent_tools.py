import json
import os
from langchain.tools import tool
from typing import List

CALENDAR_DB_FILE = "calendar_db.json"

def _load_events():
    if os.path.exists(CALENDAR_DB_FILE):
        with open(CALENDAR_DB_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def _save_event(event):
    events = _load_events()
    events.append(event)
    with open(CALENDAR_DB_FILE, "w", encoding="utf-8") as f:
        json.dump(events, f, ensure_ascii=False, indent=4)

@tool("create_calendar_event")
def create_calendar_event(
    title: str,
    start_time: str,       # ISO 格式: "2024-01-15T14:00:00"
    end_time: str,         # ISO 格式: "2024-01-15T15:00:00"
    attendees: List[str],  # 电子邮件地址列表
    location: str = ""
) -> str:
    """
    创建一个日历事件。
    要求严格使用 ISO 日期时间格式。
    
    参数:
        title: 事件的标题。
        start_time: 开始时间，ISO 格式 (如 "2024-01-15T14:00:00")。
        end_time: 结束时间，ISO 格式 (如 "2024-01-15T15:00:00")。
        attendees: 参与者的电子邮件地址列表。
        location: 事件地点（可选）。
    """
    event = {
        "title": title,
        "start_time": start_time,
        "end_time": end_time,
        "attendees": attendees,
        "location": location
    }
    _save_event(event)
    print(f"\n[Calendar Tool] 🗓️ 成功将事件写入本地存储: {title}")
    return f"已创建事件：{title}，时间从 {start_time} 到 {end_time}，有 {len(attendees)} 位参与者"

@tool("get_available_time_slots")
def get_available_time_slots(
    attendees: List[str],
    date: str,  # ISO 格式: "2024-01-15"
    duration_minutes: int
) -> List[str]:
    """
    检查特定日期给定参与者的日历空闲时间。
    
    参数:
        attendees: 参与者的电子邮件地址列表。
        date: 日期，ISO 格式 (如 "2024-01-15")。
        duration_minutes: 会议持续时间（分钟）。
    """
    print(f"\n[Calendar Tool] 🔍 正在查询 {date} 的空闲时间，参与者: {attendees}")
    # 占位符：在实际应用中，这里会查询日历 API
    return ["09:00", "14:00", "16:00"]