import os
from langchain.agents import create_agent
from langchain_community.chat_models.tongyi import ChatTongyi
from langgraph.checkpoint.memory import InMemorySaver
from core.prompts import SUPERVISOR_PROMPT

# 直接从子代理文件导入包装好的 Tool
from agent.calendar_agent import schedule_event
from agent.email_agent import manage_email
# 加载环境变量
from dotenv import load_dotenv
load_dotenv()


supervisor_agent = create_agent(
    model=ChatTongyi(
        model="qwen3-max",
        api_key=os.getenv("DASHSCOPE_API_KEY")
    ),
    tools=[schedule_event, manage_email],
    system_prompt=SUPERVISOR_PROMPT,
    checkpointer=InMemorySaver(),
)