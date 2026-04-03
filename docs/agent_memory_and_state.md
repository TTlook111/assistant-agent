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

1. **底层数据库结构是怎样的？**
   在 `checkpoints.sqlite` 数据库中，有一张核心表 `checkpoints`，它记录了图在每一步执行后的状态。它不是简单地把每一句话存成一行，而是**存储整个公文包（State）的快照**。
   - `thread_id`: 聊天窗口ID（如 `test-user-001`）。
   - `checkpoint_id`: 状态的版本号（UUID）。每执行完一个节点，就会生成一个新版本。
   - `checkpoint` (BLOB): **这是核心！** 它是一个被序列化（变成二进制）的大字典，里面包含了你定义的 `AgentState` 的所有内容，包括 `messages` 列表、`email_draft` 等。

2. **多轮对话是如何追加的？**
   - 当你在 `State` 定义中使用了 `Annotated[list, operator.add]`（如 `messages` 字段），这告诉 LangGraph：“每次有新消息进来，不要覆盖，而是追加（Append）到旧列表中”。
   - 假设我们在同一个 `thread_id` 下聊天：
     - **第 1 轮**：你说“你好”。图执行完后，存入数据库，生成 `checkpoint_id=v1`。此时里面反序列化出来的 `messages` 是 `[User("你好"), AI("你好！")]`。
     - **第 2 轮**：你说“发邮件”。LangGraph 拿着 `thread_id` 去查数据库，拿到最新的 `v1` 快照。它把新消息追加进去，图执行完后，存入新记录 `checkpoint_id=v2`。此时 `messages` 是 `[User("你好"), AI("你好！"), User("发邮件"), AI("发给谁？")]`。
   - **注意**：数据库里会同时存在 `v1` 和 `v2` 两条记录（这被称为“状态旅行”机制，允许你让 Agent 回滚到之前的状态）。

3. **前端如何展示本窗口的对话？**
   前端**不需要（也不应该）**去解析 SQLite 里的 BLOB 字段。通常有两种方式：
   - **方式 A（流式输出）**：前端只负责展示用户当前输入的话，以及后端流式返回的 AI 当前生成的回复。前端自己维护一个本地的 UI 列表。
   - **方式 B（调用获取状态 API）**：如果你想在前端刷新页面时恢复历史记录，可以在后端提供一个接口，调用 `agent.get_state(config)`。
     ```python
     # 只要传入 thread_id，就能拿到这个聊天框的完整历史
     current_state = supervisor_agent.get_state({"configurable": {"thread_id": "test-user-001"}})
     history_messages = current_state.values["messages"] # 这是一个列表，包含所有对话
     # 将 history_messages 转化为 JSON 发给前端渲染即可
     ```

### 2.3 历史记忆是主 Agent 独享，还是子 Agent 也知道？
这是一个非常好的架构思考！在实际的多智能体设计中：

- **主 Agent (Supervisor/Router)**：必须拥有**全局的、完整的历史记忆**。因为它需要结合之前的上下文来理解用户的最新意图，决定分配给哪个子 Agent。
- **子 Agent (Worker/Tool)**：**通常不需要完整的历史记忆，因为它“只做任务”**。
  - 如果把几十轮不相关的对话都发给负责发邮件的子 Agent，不仅浪费 Token，还容易让它产生幻觉（“走神”）。
  - **最佳实践：记忆隔离与状态按需拼接**。
    1. 主 Agent 在决定调用子 Agent 时，**不应该把完整的 `messages` 传过去**。
    2. 如果子 Agent 需要“额外的知识”或“先前的上下文”（比如上一轮对话中提到的会议地点），**主 Agent 应该负责提取这些信息，并将其写入 State 中某个专门的字段**（例如 `State["meeting_context"]`）。
    3. 当流程流转到子 Agent 节点时，子 Agent 只从 State 里读取 `meeting_context` 这个特定字段，然后在它自己的内部逻辑中，**将这段短小精悍的知识拼接到它自己的 System Prompt 或指令中**。这样既获得了所需的上下文，又避免了被无关的冗长聊天记录干扰。

### 2.5 主 Agent 什么时候去赋值 State？

这是一个非常关键的逻辑闭环问题：“既然子 Agent 需要从 State 里拿 `meeting_context`，那主 Agent 到底是在哪一步把这个值写进 State 的呢？”

在 LangGraph 的多智能体架构中，主 Agent（Supervisor）通常有两种方式来更新 State：

#### 方式 A：通过 Tool 调用更新 (当前项目的模式)
如果你使用的是 `create_agent` 高层封装，主 Agent 本身就是一个大模型循环。你可以给主 Agent 提供一个**专门用来“写纸条”的内部工具**。

1. **定义一个写状态的工具**：
   ```python
   @tool("update_meeting_context")
   def update_meeting_context(context: str) -> Command:
       # 这里使用了 Command 对象，它不仅能返回值，还能直接更新图的全局状态！
       return Command(
           update={"meeting_context": context}
       )
   ```
2. **主 Agent 的行为**：主 Agent 在思考时发现：“哦，用户想发邮件，但我得先把会议地点告诉邮件 Agent”。于是它**先调用 `update_meeting_context` 工具**，把地点写进 State。写完后，它**再调用 `manage_email` 工具**。

#### 方式 B：通过原生节点 (Node) 返回值更新 (进阶架构)
如果你不使用 `create_agent`，而是自己手写 `StateGraph` 的节点（这在复杂的企业级项目中更常见）。主 Agent 就是一个普通的 Python 函数节点：

```python
def supervisor_node(state: AgentState) -> Command:
    # 1. 主 Agent 阅读所有历史消息，决定下一步做什么
    messages = state["messages"]
    response = llm.invoke(messages) 
    
    # 2. 假设大模型决定要发邮件，并提取出了上下文
    # 在原生 LangGraph 中，节点返回一个 Command 对象就可以直接更新 State 并路由！
    return Command(
        update={
            # 把提取出的上下文写入 State
            "meeting_context": "明天下午两点，在A栋",
            # 记录下一步该谁走
            "next_agent": "email_agent"
        },
        goto="email_agent" # 告诉框架：下一步去执行 email_agent 节点
    )
```

**总结**：
无论是用内部 Tool 还是原生 Node，核心思想都是：**在主 Agent 决定交棒给子 Agent 之前的那一刻**，主 Agent 负责把提取好的关键信息“塞进公文包（State）”里，然后再把公文包递给子 Agent。

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

### 2.6 “时间旅行” (Time Travel)：如何撤回与修改历史？

你提到了一个非常高级的真实场景：“如果在第 5 轮发现大模型偏题了，我想直接退回到第 1 轮并修改我的提问，该怎么实现？”

这在 LangGraph 中被称为 **Time Travel (时间旅行)** 或 **Forking (状态分叉)**。这正是 Checkpoint 机制的终极威力！

#### 核心原理：基于 checkpoint_id 的链式存储
正如我们在 2.2 节提到的，数据库不是覆盖存储，而是追加存储。在同一个 `thread_id` 下，每一步都会生成一个独立的 `checkpoint_id`（版本号）。这就像 Git 的 Commit 记录一样。

- v1 (第 1 轮用户输入)
- v2 (第 1 轮大模型回复)
- ...
- v9 (第 5 轮用户输入)
- v10 (第 5 轮大模型胡言乱语)

#### 具体实现步骤

1. **获取历史快照列表**
   首先，你需要通过 `get_state_history` 获取该线程下所有的历史版本，找到你想要回退的那个“时间点”。
   ```python
   # 遍历历史记录，找到你想回退的那个 checkpoint_id
   history = list(supervisor_agent.get_state_history({"configurable": {"thread_id": "test-user-001"}}))
   
   # 假设你想回到第 1 轮大模型回复之前的状态（即只保留了你第一轮的问题）
   # 假设你找到了那个特定历史节点的配置信息
   target_config = {"configurable": {"thread_id": "test-user-001", "checkpoint_id": "v1"}}
   ```

2. **更新历史状态 (Update State)**
   找到目标历史节点后，你可以直接修改那个节点里的内容（比如修改你当时的问题）。
   ```python
   # 我们使用 update_state 来覆盖 v1 节点里的内容
   # 注意：你需要告诉框架你是作为哪个身份在修改（比如伪装成用户的输入节点）
   supervisor_agent.update_state(
       target_config,
       {"messages": [{"role": "user", "content": "修改后的第一轮提问"}]},
   )
   ```

3. **从修改后的历史节点“分叉”执行 (Fork & Replay)**
   修改完成后，你只需要拿着那个历史 `config` 再次调用 `invoke` 即可！
   ```python
   # 传入 None，表示不需要提供新消息，直接从修改后的 v1 节点继续往下跑！
   response = supervisor_agent.invoke(None, config=target_config)
   ```

#### 数据库里发生了什么？
当你执行上述操作后，LangGraph **不会删除** v2 到 v10 的错误记录（它保留了所有的犯罪证据）。
相反，它会基于你修改后的 v1，**开辟一条新的时间线（Fork）**，并生成新的 `checkpoint_id` (比如 v11, v12)。
这就是极其强大的**无损时间旅行机制**，它允许你在复杂的 Agent 任务中反复试错、回退、修改参数，而不用每次都从头开始！

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

**核心机制：** `InjectedState` 里能填什么字符串，完全取决于你在创建主 Agent（图的入口）时，传给它的 `state_schema`（即契约）是什么。如果在 `AgentState` 里没有定义 `user_age`，那么在工具里写 `InjectedState("user_age")` 就会导致注入失败。

这种设计类似于前端开发中的 Redux 或 Vuex 状态管理模式：**单一数据源 (Single Source of Truth)**，但各个组件（Agent）各取所需。

---

## 4. 总结

1. **短期记忆**：由 LangGraph 的 `State` 维护，通过 `Checkpoint` 机制持久化（如 `thread_id` 区分）。
2. **长期记忆**：需要额外开发，在进入图的流转前，通过检索数据库，将结果动态注入到 Prompt 或 `State` 中。
3. **状态对象 (State)**：全局唯一，像一个公文包。不同的子 Agent 从中读取自己需要的字段，处理完后将结果写回对应的字段，实现多智能体之间的数据协同。

---

## 5. LangChain 抽象层 vs LangGraph 原生层：状态注入对象全解析

在阅读官方文档或不同版本的代码时，你可能会看到很多用于获取状态和上下文的对象（如 `InjectedState`、`ToolRuntime`、`@before_model` 等）。**它们目前都在使用中，但分别属于不同的抽象层和使用场景。**

### 5.1 原生 LangGraph 模式：`InjectedState`
**适用场景**：这是目前开发 LangGraph 原生工具（Tools）时**最推荐、最干净**的做法（如代码 `email_agent.py#L40` 所示）。
- **作用**：精准地从全局 `State` 中提取某个具体的业务字段（如 `email_draft`、`messages`），注入到工具函数的参数中。
- **优势**：
  1. **对大模型隐身**：大模型在查看工具的 JSON Schema 时，完全看不到被 `InjectedState` 标记的参数，它只需要按需生成真正的业务参数。
  2. **极度解耦**：工具函数不需要关心底层框架的运行机制，只需声明自己需要什么数据。

### 5.2 高阶 `create_agent` 模式：`Runtime` 与 Middleware
**适用场景**：如果你使用的是 LangChain 最新推出的高阶封装 API `langchain.agents.create_agent`（它底层包裹了 LangGraph），你才会用到以下对象：

1. **获取 `tool_call_id` 的两种方式：`InjectedToolCallId` vs `ToolRuntime`**：
   在需要手动构造 `ToolMessage` 闭环时，我们需要大模型生成的 ID。LangChain 提供了两种注入方式：
   - **方式 A (推荐)：`tool_call_id: Annotated[str, InjectedToolCallId]`**。这是一种更现代、更纯粹的依赖注入方式（类似于 `InjectedState`）。如果你**仅仅只需要**拿到这个 ID 字符串，这是最佳选择，代码类型非常清晰。
   - **方式 B：`runtime: ToolRuntime`**。注入整个运行时对象，然后通过 `runtime.tool_call_id` 获取。如果你不仅需要 ID，还需要访问跨会话的长期记忆存储 (`runtime.store`) 或配置上下文 (`runtime.context`)，则必须使用这种方式。两者在获取 ID 这一目的上是等价的。

2. **`@dynamic_prompt` 与 `ModelRequest`**：
   - **何时使用**：专门用于 `create_agent` 的中间件。在请求真正发给大模型（LLM）的前一刻，用来**动态修改系统提示词**。比如根据 `request.runtime.context.user_name` 动态把提示词改为“你好，张三”。

3. **`@before_model` / `@after_model` 与 `Runtime`**：
   - **何时使用**：也是 `create_agent` 的中间件钩子。用于在调用大模型前后进行拦截。比如记录日志、审计，或者在调用模型前强制检查/修改当前图的 `AgentState` 和 `Runtime` 上下文。

**总结与选型建议**：
- 如果你想在 Tool 里读取业务变量（如草稿），用 **`InjectedState`**。
- 如果你想在 Tool 里拿到调用 ID 来构造消息，优先用 **`InjectedToolCallId`**；如果还需要操作底层存储/上下文，用 **`ToolRuntime`**。
- 只有在需要全局拦截请求、动态修改提示词时，才去研究中间件钩子。

---

## 6. Tool 返回值的差异：何时返回 `Command`，何时返回 `str`

在编写 `@tool` 时，你可能会发现有些工具直接返回普通的字符串（或字典），而有些工具却返回了 `Command` 对象。这本质上取决于**这个工具扮演的角色和层级**。

### 6.1 返回普通字符串或字典（底层干活 Tool）
- **适用场景**：真正的底层业务执行工具。比如 `send_email`（发送邮件的脚本）、`get_weather`（调用查天气的 API）。
- **底层机制**：当工具返回 `str` 时，LangChain 的 `ToolNode` 会拦截这个返回值，并**自动**为你包装成一个 `ToolMessage(content="你的返回值", tool_call_id="xxx")`，然后追加到全局 `State["messages"]` 的末尾。
- **作用**：这些工具只负责执行具体的物理动作，不关心整个系统的“图”怎么流转。大模型通过阅读更新后的 `messages` 列表就能看到工具执行的结果。

### 6.2 返回 `Command` 对象（路由与状态管理 Tool）
- **适用场景**：用来管理多 Agent 状态和流程的高阶工具（或称代理间通信桥梁）。比如你项目中的 `manage_email` 和 `schedule_event`。
- **底层机制**：一旦返回 `Command`，你就是在对框架说：“**我接管了这个节点的状态更新，不要再自动帮我包装 ToolMessage 了**”。
- **作用**：
  1. **跨 Agent 传递状态**：你需要更新全局业务状态。例如，不仅告诉系统邮件发完了，还要顺便 `update={"email_draft": "新草稿"}`。
  2. **动态路由 (Handoff)**：你可能需要控制图的下一步走向，使用 `goto="next_agent_node"`。
- **代价与责任**：正如第 4 节所述，既然你接管了更新，你**必须手动构造 `ToolMessage`** 并塞进 `update={"messages": [...]}` 中，否则大模型就会因为找不到匹配的回执而报 400 InvalidParameter 错误。
