# 🤖 Assistant Agent

![Python Version](https://img.shields.io/badge/python-3.13%2B-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-orange)
![License](https://img.shields.io/badge/License-MIT-green)

**Assistant Agent** 是一个基于 [LangGraph](https://python.langchain.com/docs/langgraph/) 和大语言模型（默认配置为通义千问 `qwen3-max`）构建的多智能体协作系统（Multi-Agent System）。它演示了在真实场景下如何利用 **Supervisor-Worker** 架构、**人机协同 (Human-in-the-Loop)** 以及 **SQLite 持久化记忆** 来打造一个功能强大且安全的个人助理。

***

## ✨ 核心特性

- 🧠 **多智能体架构 (Multi-Agent)**: 采用 `Supervisor Agent` 负责意图识别与任务路由，将具体操作下发给专属领域的 `Calendar Agent` 和 `Email Agent`，实现任务解耦。
- 💾 **持久化记忆 (Persistent Memory)**: 结合 `SqliteSaver` 和 `thread_id`，实现真正的跨会话记忆保持。系统重启后，Agent 依然记得你的对话历史。
- ⏸️ **人机协同 (Human-in-the-Loop)**: 在执行高危操作（如发送邮件、预定日程）前主动拦截请求，支持用户的**批准 (Approve)**、**拒绝 (Reject)** 和 **编辑 (Edit)** 决策，防止大模型“自作主张”。
- 🔄 **全局状态共享 (State Management)**: 使用强类型的 `AgentState` (TypedDict) 进行上下文传递，配合 `@tool` 的 `InjectedState` 实现跨 Agent 的状态按需注入。

## 📂 项目结构

```text
assistant-agent/
├── agent/                  # 智能体定义目录
│   ├── supervisor_agent.py # 主代理 (路由与调度)
│   ├── calendar_agent.py   # 日程处理子代理
│   └── email_agent.py      # 邮件处理子代理
├── core/                   # 核心配置与状态管理
│   ├── config.py           # 环境变量与配置加载
│   ├── prompts.py          # 系统 Prompt 集中管理
│   └── state.py            # 全局 AgentState 定义
├── tools/                  # 工具函数目录 (Tools)
│   ├── supervisor_tools.py # 供主代理调用的工具
│   ├── calendar_agent_tools.py 
│   └── email_agent_tools.py
├── docs/                   # 进阶技术文档
│   ├── agent_memory_and_state.md
│   └── langgraph_human_in_the_loop.md
├── main.py                 # 项目入口文件 (终端交互)
├── pyproject.toml          # uv 依赖管理配置
└── .env                    # 环境变量配置 (需手动创建)
```

## 🚀 快速开始

### 1. 环境准备

确保你的机器上安装了 [Python 3.13+](https://www.python.org/) 和 [uv 包管理器](https://github.com/astral-sh/uv)。

### 2. 克隆与安装依赖

```bash
# 1. 克隆项目
git clone git@github.com:TTlook111/assistant-agent.git
cd assistant-agent

# 2. 使用 uv 同步依赖
uv sync
```

### 3. 配置环境变量

在项目根目录创建一个 `.env` 文件，并填入必要的 API Keys：

```env
# 必填: 阿里云百炼 (通义千问) API Key
DASHSCOPE_API_KEY="sk-your-dashscope-api-key"

# 可选: 用于 LangSmith 运行追踪 (推荐)
LANGCHAIN_TRACING_V2="true"
LANGCHAIN_API_KEY="ls__your-langchain-api-key"
```

### 4. 运行系统

```bash
# 启动你的个人助理
uv run python main.py
```

## 💡 演示示例

启动后，在终端与 Agent 交互。下面展示了一个**人机协同**的拦截场景：

```text
🤖 个人助理 Agent 已启动！(输入 'exit' 退出)
--------------------------------------------------
[系统提示] 已连接到 SQLite 数据库，当前会话 ID: test-user-001
[系统提示] 你可以随时按 Ctrl+C 强制退出，再次运行程序时，它依然记得你！

你: 帮我发一封邮件给老板，说我明天请假。

Agent 思考中...
⚠️ [拦截器触发] 助理请求人工审批！
中断原因: Interrupted by tool 'manage_email'

是否同意执行该操作？(y: 批准 / n: 拒绝 / e: 编辑): e
进入编辑模式...
请输入修改后的指令 (例如：发给张三，内容改为开会): 发给老板，内容改为我明天上午请假去医院。

助理: 邮件已经发送给老板了，内容是：“老板您好，我明天上午需要请假去医院，特此请假，望批准。”
```

## 📚 进阶学习指南

本项目不仅是一个代码实现，更是一个 LangGraph 架构的最佳实践示例。想要深入理解底层的设计理念，请阅读以下文档：

- 📖 [Agent 的状态与记忆机制详解](./docs/agent_memory_and_state.md)：深入理解 `TypedDict`、`InjectedState` 和 `SqliteSaver` 的工作原理。
- 📖 [人机协同 (HITL) 深度剖析](./docs/langgraph_human_in_the_loop.md)：拆解如何使用 `GraphInterrupt` 和 `Command(resume=...)` 实现优雅的断点恢复与决策干预。

## 📄 许可证

本项目基于 [MIT License](./LICENSE) 开源，你可以自由使用、修改和分发。
