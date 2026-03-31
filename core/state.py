from typing import TypedDict, Annotated, List, Any
import operator
from langchain_core.messages import BaseMessage

# 定义整个多智能体系统的全局状态 (State)
# 就像一个公文包，所有的节点都在读写这个字典
class AgentState(TypedDict):
    # 1. 公共字段：消息历史
    # Annotated 和 operator.add 意味着：当某个节点返回新的 messages 时，
    # LangGraph 不会覆盖原来的列表，而是把新消息追加 (append) 到列表末尾
    messages: Annotated[List[BaseMessage], operator.add]
    
    # 2. 路由字段：决定下一步该谁走
    next_agent: str
    
    # 3. 业务专有字段：只供特定的子代理读写（比如给邮件代理暂存草稿）
    email_draft: str
    calendar_status: str
