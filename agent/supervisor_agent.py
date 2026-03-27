from langchain.agents import create_agent
from langchain_community.chat_models.tongyi import ChatTongyi
from langgraph.checkpoint.memory import InMemorySaver
from core.prompts import SUPERVISOR_PROMPT

# 直接从子代理文件导入包装好的 Tool
from agent.calendar_agent import schedule_event
from agent.email_agent import manage_email

supervisor_agent = create_agent(
    model=ChatTongyi(
        model="qwen3-max",
    ),
    tools=[schedule_event, manage_email],
    system_prompt=SUPERVISOR_PROMPT,
    checkpointer=InMemorySaver(),
)