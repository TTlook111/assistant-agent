from langchain.tools import tool
from typing import List

@tool("create_calendar_event", description="创建一个日历事件。要求严格使用 ISO 日期时间格式。")
def create_calendar_event(
    title: str,
    start_time: str,       # ISO 格式: "2024-01-15T14:00:00"
    end_time: str,         # ISO 格式: "2024-01-15T15:00:00"
    attendees: List[str],  # 电子邮件地址列表
    location: str = ""
) -> str:
    # 占位符：在实际应用中，这里会调用 Google Calendar API, Outlook API 等。
    return f"已创建事件：{title}，时间从 {start_time} 到 {end_time}，有 {len(attendees)} 位参与者"

@tool("get_available_time_slots", description="检查特定日期给定参与者的日历空闲时间。")
def get_available_time_slots(
    attendees: List[str],
    date: str,  # ISO 格式: "2024-01-15"
    duration_minutes: int
) -> List[str]:
    # 占位符：在实际应用中，这里会查询日历 API
    return ["09:00", "14:00", "16:00"]