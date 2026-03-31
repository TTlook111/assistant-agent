import os
from langchain.agents import create_agent
from langchain_community.chat_models.tongyi import ChatTongyi
from langgraph.checkpoint.memory import InMemorySaver
from core.prompts import SUPERVISOR_PROMPT
from core.state import AgentState
from core.config import DASHSCOPE_API_KEY

# 直接从子代理文件导入包装好的 Tool
from agent.calendar_agent import schedule_event
from agent.email_agent import manage_email
from tools.supervisor_tools import update_email_draft, update_calendar_status

supervisor_agent = create_agent(
    model=ChatTongyi(
        model="qwen3-max",
        api_key=DASHSCOPE_API_KEY
    ),
    tools=[schedule_event, manage_email, update_email_draft, update_calendar_status],
    system_prompt=SUPERVISOR_PROMPT,
    checkpointer=InMemorySaver(),
    # 注意：在较新的 langchain 版本中，create_agent/create_react_agent 期望传入类型本身。
    # 如果 IDE 提示“应为类型 TypedDict”，我们可以直接传递类本身。
    state_schema=AgentState,
)