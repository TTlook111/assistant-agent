import os
# 加载环境变量
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_community.chat_models import ChatTongyi
from langchain.tools import tool
from core.prompts import CALENDAR_AGENT_PROMPT
from tools.calendar_agent_tools import create_calendar_event, get_available_time_slots

load_dotenv()

calendar_agent = create_agent(
    model=ChatTongyi(
        model="qwen3-max",
        api_key=os.getenv("DASHSCOPE_API_KEY")
    ),
    tools=[create_calendar_event, get_available_time_slots],
    system_prompt=CALENDAR_AGENT_PROMPT,
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={"create_calendar_event": True},
            description_prefix="日历活动待批准",
        ),
    ],
)

@tool("schedule_event", description=(
    "使用自然语言安排日历事件。\n"
    "当用户想要创建、修改或查看日历预约时使用此工具。\n"
    "处理日期/时间解析、空闲时间检查和事件创建。\n"
    "输入：自然语言的日程安排请求（例如：'下周二下午2点和设计团队开会'）"
))
def schedule_event(request: str) -> str:
    result = calendar_agent.invoke({
        "messages": [{"role": "user", "content": request}]
    })
    return result["messages"][-1].text