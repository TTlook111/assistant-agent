import os
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_community.chat_models import ChatTongyi
from langchain.tools import tool
from core.prompts import EMAIL_AGENT_PROMPT
from tools.email_agent_tools import send_email
# 加载环境变量
from dotenv import load_dotenv
load_dotenv()

email_agent = create_agent(
    model=ChatTongyi(
        model="qwen3-max",
        api_key=os.getenv("DASHSCOPE_API_KEY")
    ),
    tools=[send_email],
    system_prompt=EMAIL_AGENT_PROMPT,
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={"send_email": True},
            description_prefix="发件邮件待批准",
        ),
    ],
)

@tool("manage_email", description=(
    "使用自然语言发送电子邮件。\n"
    "当用户想要发送通知、提醒或任何电子邮件通信时使用此工具。\n"
    "处理收件人提取、主题生成和电子邮件撰写。\n"
    "输入：自然语言的电子邮件请求（例如：'给他们发个关于会议的提醒邮件'）"
))
def manage_email(request: str) -> str:
    result = email_agent.invoke({
        "messages": [{"role": "user", "content": request}]
    })
    return result["messages"][-1].text