# LangGraph 结合通义千问 (Qwen) 常见踩坑与修复记录

在使用 LangGraph 和 `ChatTongyi` (阿里云百炼通义千问) 构建多智能体时，由于大模型底层的消息历史格式校验非常严格，会遇到一些经典的 `ToolMessage` 同步报错。这里记录了最常见的错误及其修复方案。

## 1. 缺失 ToolMessage 回执

### 报错信息
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

### 修复方案
你有三种选择来开启一段“干净”的对话历史：
1. **(推荐)** 在 `main.py` 中换一个新的 `thread_id`，例如改为 `"test-user-002"`，开启一个全新的会话。
2. 直接删除项目根目录下的 `checkpoints.sqlite` 文件，清空所有人的记忆。
3. （高阶用法）使用 LangGraph 的 `update_state` API 删除/修正数据库里最后那条错误的消息。