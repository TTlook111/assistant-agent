# Agent 记忆与状态管理 (LangChain & LangGraph)

在开发多智能体 (Multi-Agent) 架构时，理解**记忆 (Memory)** 和**状态 (State)** 的流转是核心难点。本文档将为你梳理 LangChain 和 LangGraph 中关于记忆和状态的底层逻辑。

---

## 1. 记忆的两个维度

大模型本质上是**无状态的 (Stateless)**。所有的“记忆”，都是我们在调用模型前，将历史信息拼接到 Prompt 中的结果。

### 1.1 短时记忆 (Short-term Memory)
- **定义**：当前会话 (Session) 的上下文，通常是 `messages` 列表（包含 User, AI, Tool 的多轮对话）。
- **作用**：防止 Agent 在 ReAct 循环（思考 -> 调用工具 -> 观察）中“失忆”，维持当前对话的连贯性。
- **管理方式**：
  - 在 LangChain 早期，开发者需要手动维护一个 List，每次请求前拼接。
  - 在 LangGraph 中，短时记忆被内置到了 `State` 中，通过配置 `Checkpointer` (如 `InMemorySaver` 或 SQLite) 自动实现对话历史的追加和管理。

### 1.2 长期记忆 (Long-term Memory)
- **定义**：跨越不同会话、不同时间的知识持久化（如用户偏好、过往事实）。
- **管理方式**：将重要信息提取为摘要或向量 (Embeddings)，存入外部数据库（关系型数据库或向量数据库）。每次对话前，通过 RAG (检索增强生成) 将相关记忆查出，作为系统提示词 (System Prompt) 补充给大模型。

---

## 2. Checkpoint 与 LangChain 的协同关系

你总结得非常准确：**LangGraph 的 Checkpoint 机制与 LangChain 的短时记忆是相辅相成的。**

### 2.1 它们的明确分工
- **LangGraph (图流转与状态管理)**：
  - 负责**“大局”**。它定义了流程怎么走，并在每个节点执行完后，把整个 `State`（包含消息历史）“拍个快照”存到 Checkpoint 数据库里。
  - **核心机制：thread_id**：在实际业务中，一个人可能有多个独立的聊天窗口。**标准的最佳实践是使用 `session_id`（对话框ID）作为 `thread_id`**，确保每个聊天窗口的状态严格隔离。

### 2.2 Checkpoint 是如何保存和拼接记忆的？
这是一个非常核心的问题，涉及到 Checkpoint 在底层的工作原理：

1. **它是如何保存的？（多轮对话的追加机制）**
   - Checkpoint 并不是每次只保存一句话，而是保存了**当前这一步的完整 `State` 快照**。
   - 当你在 `State` 定义中使用了 `Annotated[list, operator.add]`（如 `messages` 字段），这告诉 LangGraph：“每次有新消息进来，不要覆盖，而是追加（Append）到旧列表中”。
   - 因此，同一个聊天框（`thread_id`）下进行多次对话时，数据库里保存的是：
     - Step 1: `[用户: "你好"]`
     - Step 2: `[用户: "你好", AI: "你好，需要什么帮助？"]`
     - Step 3: `[用户: "你好", AI: "你好，需要什么帮助？", 用户: "发邮件"]`
   - 每次调用 `invoke` 时，LangGraph 会去数据库里查这个 `thread_id` **最新的一条快照**，从而拿到完整的历史消息列表。

2. **它会自动拼接到主 Agent 的 Prompt 吗？**
   - **是的，会自动拼接，但这得益于底层的机制配合**。
   - 当你在主程序调用 `agent.invoke({"messages": [{"role": "user", "content": "新消息"}]})` 时，LangGraph 引擎会在后台做两件事：
     1. 从数据库读取最新的历史 `messages` 列表。
     2. 将你的“新消息”追加到历史列表的末尾。
     3. 将这个**完整的列表**交给主 Agent 的大模型去阅读。大模型收到的是 `[历史消息1, 历史消息2, ..., 新消息]`，从而表现出“拥有记忆”的能力。

### 2.3 历史记忆是主 Agent 独享，还是子 Agent 也知道？
这是一个非常好的架构思考！在实际的多智能体设计中：

- **主 Agent (Supervisor/Router)**：必须拥有**全局的、完整的历史记忆**。因为它需要结合之前的上下文来理解用户的最新意图，决定分配给哪个子 Agent。
- **子 Agent (Worker/Tool)**：**通常不需要完整的历史记忆，因为它“只做任务”**。
  - 如果把几十轮不相关的对话都发给负责发邮件的子 Agent，不仅浪费 Token，还容易让它产生幻觉（“走神”）。
  - **最佳实践：记忆隔离与状态按需拼接**。
    1. 主 Agent 在决定调用子 Agent 时，**不应该把完整的 `messages` 传过去**。
    2. 如果子 Agent 需要“额外的知识”或“先前的上下文”（比如上一轮对话中提到的会议地点），**主 Agent 应该负责提取这些信息，并将其写入 State 中某个专门的字段**（例如 `State["meeting_context"]`）。
    3. 当流程流转到子 Agent 节点时，子 Agent 只从 State 里读取 `meeting_context` 这个特定字段，然后在它自己的内部逻辑中，**将这段短小精悍的知识拼接到它自己的 System Prompt 或指令中**。这样既获得了所需的上下文，又避免了被无关的冗长聊天记录干扰。

### 2.4 具体代码实现：子 Agent 如何获取 State 并拼接 Prompt？

你提到了一个非常核心的落地问题：“**在 Tool 里面，还需要用 ToolRuntime 去获取 State 吗？拼接操作写在哪里？**”

随着 LangChain/LangGraph 的演进，我们不再推荐使用复杂的 `ToolRuntime`（那通常是较老版本或需要极深度定制时的做法）。现在最优雅、最标准的方式是使用 **依赖注入 (`InjectedState` 或 `RunnableConfig`)**。

#### 场景设定
假设主 Agent 在 `State` 里存了一个 `State["meeting_context"] = "明天下午两点，在 A 栋会议室"`。现在我们要调用发邮件的工具。

#### 步骤 1：在 Tool 中注入 State (或特定字段)
你可以直接在 `@tool` 装饰的函数参数里，通过 `Annotated` 和 `InjectedState` 告诉框架：“**这个参数不要让大模型去生成，请框架直接把 State 里的值塞进来。**”

```python
from typing import Annotated
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

@tool("manage_email", description="发送邮件的工具")
def manage_email(
    request: str, # 这是大模型需要生成的参数
    # 核心魔法：告诉框架，把 State 里的 meeting_context 塞给这个参数。大模型完全看不见这个参数。
    meeting_context: Annotated[str, InjectedState("meeting_context")] 
) -> str:
    # --- 这里就是你问的“拼接操作主要写在哪里”的答案 ---
    # 拼接操作就写在 Tool 函数的内部，在真正调用子 Agent 的 invoke 之前。
    
    # 1. 构造完整的、携带上下文的 Prompt
    full_request = f"{request}\n\n【补充的会议上下文信息】：{meeting_context}"
    
    # 2. 调用子 Agent (email_agent) 去干活
    result = email_agent.invoke({
        "messages": [{"role": "user", "content": full_request}]
    })
    
    return result["messages"][-1].text
```

#### 为什么这种方式最好？
1. **无需 ToolRuntime**：代码非常纯粹，就是一个普通的 Python 函数，通过类型注解就完成了状态的获取。
2. **对大模型隐身**：因为加了 `InjectedState`，大模型在阅读 Tool 的 Schema 时，根本看不到 `meeting_context` 这个参数，它只会乖乖地生成 `request`。这就避免了模型产生幻觉（比如它自己瞎编一个上下文传进来）。
3. **职责清晰**：拼接 Prompt 的逻辑被封装在了工具函数内部，子 Agent（`email_agent`）完全不用改代码，它收到的就是一个完美的、包含了上下文的完整指令。
由于每次都会把完整的短时记忆发给大模型，如果不加以限制，很快就会导致 **Token 成本爆炸** 或 **超出模型上下文窗口报错**。因此，必须对历史记忆进行压缩：
- **方案 A (滑动窗口)**：通过在 StateGraph 中返回 `RemoveMessage(id=...)`，动态删除太久远的消息，只保留最近 N 轮对话。
- **方案 B (摘要压缩)**：使用一个小模型，定期将前面的多轮废话总结为一句话（例如：“用户想安排会议，但最终取消了”），作为系统提示词保留，同时删除原始冗长的消息。

---

## 3. 人机协同 (Human-in-the-loop)
在执行关键操作（如真实发送邮件、支付扣款）前，必须让人类审批，这就用到了人机协同。
- **工作原理**：当执行到加了拦截器（如 `HumanInTheLoopMiddleware`）的工具时，LangGraph 会自动抛出 `GraphInterrupt` 异常，并将当前状态（停在执行工具前的这一秒）保存为 Checkpoint。
- **恢复执行**：程序捕获异常后，挂起等待用户输入。当用户确认同意后，开发者只需调用 `invoke(None, config=config)`（传入 `None` 而不是新消息），LangGraph 就会拿着之前的 `thread_id` 去数据库里唤醒刚才的断点，继续把剩下的工具执行完。

---

## 3. 深入理解 LangGraph 的 State (状态)

在 LangGraph 中，**State 是整个流转过程中的唯一数据载体**。它就像是一个在各个 Agent 之间传递的“公文包”。

### 3.1 State 只能有一个吗？
**是的，整个 Graph（图）在顶层只有一个全局的 State 对象**。这个对象通常是一个 Python 的 `TypedDict`（类型字典）或 Pydantic 模型。

```python
from typing import TypedDict, Annotated
import operator

# 这是一个全局 State 的定义
class AgentState(TypedDict):
    # 所有的 Agent 共用这个消息历史
    messages: Annotated[list, operator.add]
    
    # 专门给 Calendar Agent 用的业务数据
    calendar_status: str
    
    # 专门给 Email Agent 用的业务数据
    email_draft: str
```

### 3.2 为什么多 Agent 场景下依然是单一 State？
你可能会觉得：“不同的子 Agent 处理不一样的业务，为什么不用不同的对象？”

因为如果状态分散了，主 Agent 就无法掌控全局。在 LangGraph 中，我们通过**“单一状态字典，多字段按需读写”**来解决这个问题：

1. **共享字段**：所有 Agent 都可以读取和追加 `messages`（对话历史），这是它们共享的短时记忆。
2. **专属字段**：每个子 Agent 只关心 `State` 里属于自己的那部分字段。
   - 当任务流转到 `Calendar Agent` 节点时，它读取 `messages`，处理完后，只修改 `calendar_status` 字段，然后把公文包传回给主 Agent。
   - 当任务流转到 `Email Agent` 时，它无视 `calendar_status`，只处理自己关心的 `email_draft` 字段。

这种设计类似于前端开发中的 Redux 或 Vuex 状态管理模式：**单一数据源 (Single Source of Truth)**，但各个组件（Agent）各取所需。

---

## 4. 总结

1. **短期记忆**：由 LangGraph 的 `State` 维护，通过 `Checkpoint` 机制持久化（如 `thread_id` 区分）。
2. **长期记忆**：需要额外开发，在进入图的流转前，通过检索数据库，将结果动态注入到 Prompt 或 `State` 中。
3. **状态对象 (State)**：全局唯一，像一个公文包。不同的子 Agent 从中读取自己需要的字段，处理完后将结果写回对应的字段，实现多智能体之间的数据协同。
