import os
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_community.chat_models import ChatTongyi
from langchain.tools import tool
from typing import Annotated
from langgraph.prebuilt import InjectedState

from core.prompts import EMAIL_AGENT_PROMPT
from tools.email_agent_tools import send_email
from core.config import DASHSCOPE_API_KEY

email_agent = create_agent(
    model=ChatTongyi(
        model="qwen3-max",
        api_key=DASHSCOPE_API_KEY
    ),
    tools=[send_email],
    system_prompt=EMAIL_AGENT_PROMPT,
    # 恢复人机协同拦截器：当要执行 send_email 时，强制中断！
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={"send_email": True},
            description_prefix="发送邮件待批准",
        ),
    ],
)

# 注意这里：我们使用了 InjectedState 魔法！
# 告诉框架从全局 State 中提取 email_draft（与 core.state.py 里的字段对应），大模型完全感知不到这个参数。
@tool("manage_email", description=(
    "使用自然语言发送电子邮件。\n"
    "当用户想要发送通知、提醒或任何电子邮件通信时使用此工具。\n"
    "处理收件人提取、主题生成和电子邮件撰写。\n"
    "输入：自然语言的电子邮件请求（例如：'给他们发个关于会议的提醒邮件'）"
))
def manage_email(
    request: str, 
    # 依赖注入：大模型不需要生成这个参数，框架会自动把 state 里的 email_draft 塞进来
    email_draft: Annotated[str, InjectedState("email_draft")] = ""
) -> str:
    # 这一步，我们把主代理通过 State 传来的“草稿状态”拼接到了用户的请求里，
    # 这就是在模拟“State”在不同 Agent 之间的流转，且无需传递完整的历史消息。
    full_request = request
    if email_draft:
        full_request += f"\n\n【主Agent补充的邮件草稿/上下文】：\n{email_draft}"
        
    result = email_agent.invoke({
        "messages": [{"role": "user", "content": full_request}]
    })
    return result["messages"][-1].text