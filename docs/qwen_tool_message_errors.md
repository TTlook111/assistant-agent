# LangGraph 结合通义千问 (Qwen) 常见踩坑与修复记录

在使用 LangGraph 和 `ChatTongyi` (阿里云百炼通义千问) 构建多智能体时，由于大模型底层的消息历史格式校验非常严格，会遇到一些经典的 `ToolMessage` 同步报错。这里记录了最常见的错误及其修复方案。

## 1. 缺失 ToolMessage 回执

### 报错信息
## 5. 多 Agent 架构下终端打印“助理回复为空”的问题

### 现象描述
在代码跑通、没有报错的情况下，终端可能输出如下内容：
```text
你: 给王五发消息，下午三点来找我 
Agent 思考中... 
助理: 
你: 
```

### 原因分析：消息折叠与 `ToolMessage` 
在 `main.py` 的简单测试脚本中，原本打印助理回复的代码通常是：
```python
print(f"\n助理: {response['messages'][-1].text}")
```
在多 Agent 的架构中，如果主 Agent 调用了子 Agent（例如调用了 `manage_email` 工具），并且该工具为了更新状态返回了 `Command` 和 `ToolMessage`。那么当整个图流转结束时，**`messages` 列表中的最后一条消息往往是一个 `ToolMessage`，而不是大模型说的话 (`AIMessage`)**。

`ToolMessage` 的内容存储在 `.content` 属性中，而它的 `.text` 属性通常为空。因此，使用 `.text` 打印就会输出空白。

同时，由于大模型在调用工具时，如果缺少参数（如缺少王五的邮箱地址），它会在生成的工具输入中或者干脆在它自己的 `AIMessage` 中询问，然后工具的执行结果（`ToolMessage`）会被追加到末尾。如果你只打印最后一条，就会漏掉大模型的提问过程。

### 修复方案
在遍历和打印消息时，需要智能判断消息的类型，如果是 `ToolMessage` 则打印 `.content`，如果是 `AIMessage` 则打印 `.text` 或 `.content`。

**修复后的打印代码示例：**
```python
def print_assistant_response(response):
    """
    智能地打印助理的回复。
    如果最后一条消息是 ToolMessage（工具回执），则打印其内容。
    如果是普通的 AIMessage，则打印其 text 或 content。
    """
    last_message = response['messages'][-1]
    if hasattr(last_message, 'text') and last_message.text:
        print(f"\n助理: {last_message.text}")
    else:
        # 如果是 ToolMessage 或没有 text 属性的消息，打印 content
        print(f"\n助理 [工具返回]: {last_message.content}")

# 在 invoke 后调用
response = supervisor_agent.invoke(...)
print_assistant_response(response)
```text
[错误]: Expected to have a matching ToolMessage in Command.update for tool 'update_email_draft', got: []. Every tool call (LLM requesting to call a tool) in the message history MUST have a corresponding ToolMessage.
```

### 错误原因
当 Agent 决定调用某个 Tool 时，LangGraph 会将一个带有 `tool_calls` 的 `AIMessage` 追加到对话历史中。按照 LLM 的格式规范，下一条消息**必须**是 `ToolMessage`（也就是工具的执行结果），否则对话历史就不完整，模型就会拒绝继续生成。

如果你的工具仅仅返回了 `Command(update={"some_state": "value"})`，而没有顺便把 `ToolMessage` 写进 `messages` 列表里，就会触发此报错。

### 修复方案
在工具函数的参数中，使用 `InjectedToolCallId` 拿到当前工具调用的 ID，并在返回的 `Command.update` 中，手动追加一条 `ToolMessage`。

## 2. ToolMessage 缺失 name 参数 (Qwen 严格校验)

### 报错信息
```text
[错误]: request_id: 1dd8d829-b38c-97bf-94a9-94d8df1515d7 
status_code: 400 
code: InvalidParameter 
message: <400> InternalError.Algo.InvalidParameter: An assistant message with "tool_calls" must be followed by tool messages responding to each "tool_call_id". The following tool_call_ids did not have response messages: message[3].role
```

### 错误原因
这是在上一个错误修复后，紧接着容易遇到的坑。
虽然你返回了 `ToolMessage(content="...", tool_call_id=...)`，但是**阿里云通义千问 (Qwen) 的 API 对消息格式校验极其严格**。它要求 `ToolMessage` 不仅要有 `tool_call_id`，还必须明确带有 **`name`** 参数（即被调用的工具名称）。如果没有 `name`，Qwen API 就会认为这条消息是不合法的，并抛出 400 错误。

### 修复方案
在实例化 `ToolMessage` 时，务必显式传入 `name="你的工具名称"`。

```python
from typing import Annotated
from langchain.tools import tool
from langchain_core.tools import InjectedToolCallId
from langchain_core.messages import ToolMessage
from langgraph.types import Command

@tool("update_email_draft")
def update_email_draft(
    draft_content: str, 
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    return Command(
        update={
            "email_draft": draft_content,
            "messages": [
                ToolMessage(
                    content="邮件草稿已成功保存到状态中", 
                    name="update_email_draft", # 【关键修复】：必须带上工具名称！
                    tool_call_id=tool_call_id
                )
            ]
        }
    )
```

## 3. 修复代码后仍然报 400 InvalidParameter (SQLite 幽灵记忆)

### 现象描述
你在代码中已经按照上述方案添加了带有 `name` 和 `tool_call_id` 的 `ToolMessage`，但重新运行 `main.py` 时，**依然报一模一样的 400 错误**！

### 错误原因：LangGraph 的持久化机制 (Checkpointer)
这不是代码没修好，而是**“幽灵记忆”**在作祟！
由于你在 `main.py` 中使用了 `SqliteSaver` 进行真正的硬盘级持久化，并固定了 `thread_id = "test-user-001"`。
在你修复代码**之前**，Agent 已经把那段“残缺/错误的消息历史”保存到了 `checkpoints.sqlite` 数据库中。
当你修复代码并重新运行程序时，LangGraph 会去数据库里把**那段包含错误格式的历史记录读取出来**，再次原封不动地发给大模型，导致大模型再次拒绝并报错。

## 4. `create_agent` 内部工具返回 Command 时缺失 ToolMessage 响应

### 报错信息
```text
[错误]: request_id: 54854b61-6044-96bd-81dc-55529aa4a9bc 
status_code: 400 
code: InvalidParameter 
message: <400> InternalError.Algo.InvalidParameter: An assistant message with "tool_calls" must be followed by tool messages responding to each "tool_call_id". The following tool_call_ids did not have response messages: message[5].role
```

### 错误原因
当你使用 LangChain 的 `create_agent` 高阶 API 创建子 Agent（如 `email_agent` 或 `calendar_agent`），并在调用它们的 Tool 函数（如 `manage_email` 或 `schedule_event`）中直接返回业务字符串时，LangChain 框架会**自动**为你包装一个 `ToolMessage` 发送给大模型。

**但是**，一旦你为了更新全局图状态（如 `AgentState`），将 Tool 的返回值**从普通的 `str` 改为了 `Command(update={...})`**，LangChain 就**不再会自动包装工具结果了**！它认为你已经接管了控制流。此时如果你只在 `update` 中修改了业务字段，而**没有手动组装并返回 `ToolMessage`**，大模型在下一轮对话中就会因为找不到与上一轮 `tool_call_id` 匹配的工具回执而报 400 错误。

### 修复方案
在使用 `Command` 接管状态更新后，**必须手动构造 `ToolMessage`**，并且需要引入 `ToolRuntime` 来获取大模型生成的 `tool_call_id`。

```python
from langgraph.types import Command
from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import InjectedState
from typing import Annotated

@tool("manage_email")
def manage_email(
    request: str, 
    runtime: ToolRuntime, # 注入运行时以获取 tool_call_id
    email_draft: Annotated[str, InjectedState("email_draft")] = ""
) -> Command: # 注意返回值变成了 Command
    
    # 1. 业务逻辑...
    result = email_agent.invoke(...)
    
    # 2. 核心修复：使用 Command 返回时，必须闭环构造 ToolMessage
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=result["messages"][-1].text, # 真正的工具执行结果
                    name="manage_email", # 必须带上工具名 (Qwen 强制要求)
                    tool_call_id=runtime.tool_call_id, # 必须原封不动返回大模型给的 ID
                )
            ]
        }
    )
```