from langchain.tools import tool
from langgraph.types import Command

@tool("update_email_draft")
def update_email_draft(draft_content: str) -> Command:
    """
    当需要向邮件代理（Email Agent）传递或更新邮件草稿上下文时使用。
    必须在调用 manage_email 之前调用此工具来保存草稿状态。
    
    参数:
        draft_content: 生成的邮件草稿内容或需要传递给邮件代理的上下文信息。
    """
    return Command(
        update={"email_draft": draft_content}
    )

@tool("update_calendar_status")
def update_calendar_status(status_content: str) -> Command:
    """
    当需要向日历代理（Calendar Agent）传递或更新日历状态上下文时使用。
    必须在调用 schedule_event 之前调用此工具来保存日历相关的上下文信息。
    
    参数:
        status_content: 需要传递给日历代理的上下文或状态信息。
    """
    return Command(
        update={"calendar_status": status_content}
    )
