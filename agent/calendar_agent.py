import os
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_community.chat_models import ChatTongyi
from langchain.tools import tool
from langchain_core.messages import ToolMessage
from typing import Annotated
from langgraph.prebuilt import InjectedState

from core.prompts import CALENDAR_AGENT_PROMPT
from tools.calendar_agent_tools import create_calendar_event, get_available_time_slots
from core.config import DASHSCOPE_API_KEY

calendar_agent = create_agent(
    model=ChatTongyi(
        model="qwen3-max",
        api_key=DASHSCOPE_API_KEY
    ),
    tools=[create_calendar_event, get_available_time_slots],
    system_prompt=CALENDAR_AGENT_PROMPT,
    # 恢复人机协同拦截器：当要执行 create_calendar_event 时，强制中断！
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={"create_calendar_event": True},
            description_prefix="日历活动待批准",
        ),
    ],
)

from langgraph.types import Command
from langchain.tools import ToolRuntime

@tool("schedule_event", description=(
    "使用自然语言安排日历事件。\n"
    "当用户想要创建、修改或查看日历预约时使用此工具。\n"
    "处理日期/时间解析、空闲时间检查和事件创建。\n"
    "输入：自然语言的日程安排请求（例如：'下周二下午2点和设计团队开会'）"
))
def schedule_event(
    request: str,
    runtime: ToolRuntime,
    # 同样使用依赖注入，让主 Agent 可以传递日历相关的上下文 (对应 core.state.py 中的 calendar_status)
    calendar_status: Annotated[str, InjectedState("calendar_status")] = ""
) -> Command:
    full_request = request
    if calendar_status:
        full_request += f"\n\n【主Agent补充的日历状态/上下文】：\n{calendar_status}"
        
    result = calendar_agent.invoke({
        "messages": [{"role": "user", "content": full_request}]
    })
    
    # 核心修复：使用 Command 返回并闭环构造 ToolMessage
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=result["messages"][-1].text,
                    name="schedule_event",
                    tool_call_id=runtime.tool_call_id,
                )
            ]
        }
    )