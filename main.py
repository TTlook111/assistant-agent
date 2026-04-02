# 1. 第一步：加载环境变量 (必须在导入任何 agent 之前)
# 我们已经在 core.config 中完成了 dotenv 的加载，所以只要导入它，环境变量就会被初始化
import core.config
import uuid
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.errors import GraphInterrupt
from langgraph.types import Command

# 2. 导入主代理
from agent.supervisor_agent import supervisor_agent

def main():
    print("🤖 个人助理 Agent 已启动！(输入 'exit' 退出)")
    print("-" * 50)
    
    # 模拟一个固定的用户 ID，这样即使你重启程序，只要 ID 相同，记忆就会恢复！
    # 在真实项目中，这个可能是数据库里的 user_id
    # 注意：因为之前的测试中 SQLite 保存了格式错误的对话历史，所以这里换一个新的 ID 测试。
    thread_id = "test-user-010" 
    config = {"configurable": {"thread_id": thread_id}}
    
    # 3. 初始化 SQLite Checkpointer (真正的硬盘持久化记忆)
    conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
    memory = SqliteSaver(conn)
    
    # 将真实的持久化数据库传入（覆盖掉原本代码里的 InMemorySaver）
    # 注意：在真实的 LangGraph 重构后，我们会在建图的时候把 memory 传进去。
    # 这里只是为了演示在现有架构下如何替换
    supervisor_agent.checkpointer = memory
    
    # 建立数据库表结构
    memory.setup()

    print(f"[系统提示] 已连接到 SQLite 数据库，当前会话 ID: {thread_id}")
    print(f"[系统提示] 你可以随时按 Ctrl+C 强制退出，再次运行程序时，它依然记得你！\n")

    while True:
        user_input = input("\n你: ")
        if user_input.lower() in ['exit', 'quit', '退出']:
            print("拜拜！")
            break
            
        if not user_input.strip():
            continue
            
        print("\nAgent 思考中...")
        try:
            # 第一次调用：发送用户的请求
            response = supervisor_agent.invoke(
                {"messages": [{"role": "user", "content": user_input}]},
                config=config
            )
            print(f"\n助理: {response['messages'][-1].text}")
            
        except GraphInterrupt as e:
            # 捕获到中断异常！说明某个子代理触发了 HumanInTheLoopMiddleware
            print("\n⚠️ [拦截器触发] 助理请求人工审批！")
            
            # 此时我们需要从 Checkpointer 里把“到底卡在哪个工具上”的状态读取出来
            # （在真实项目中，这里通常是去拿挂起的工具参数展示给用户看，由于封装层级深，我们先用简单确认）
            print(f"中断原因: {str(e)}")
            
            # 等待用户输入确认
            user_approval = input("\n是否同意执行该操作？(y: 批准 / n: 拒绝 / e: 编辑): ")
            if user_approval.lower() == 'y':
                print("已批准，继续执行...")
                # 关键点：批准后，如何让 Agent 继续？
                # 我们传入 Command(resume=...)，使用精准的动态决策来恢复图的执行！
                resume_command = Command(
                    resume={
                        "decisions": [
                            {
                                "type": "approve",
                            }
                        ]
                    },
                    version="v2"
                )
                resume_response = supervisor_agent.invoke(resume_command, config=config)
                print(f"\n助理: {resume_response['messages'][-1].text}")
            elif user_approval.lower() == 'e':
                # 这里我们模拟一个编辑的场景（实际项目中，这通常是一个前端表单）
                print("进入编辑模式...")
                # 由于当前是被 manage_email 工具拦截，我们可以模拟修改它的参数
                # 注意：真实的参数结构需要与被拦截工具的 schema 一致
                new_request = input("请输入修改后的指令 (例如：发给张三，内容改为开会): ")
                
                edit_command = Command(
                    resume={
                        "decisions": [
                            {
                                "type": "edit",
                                "args": {
                                    # 这里的 key 必须和被拦截工具的参数名一致，
                                    # manage_email 工具接收 'request' 和 'draft_context'
                                    "request": new_request,
                                    "draft_context": "用户手动编辑的参数"
                                }
                            }
                        ]
                    },
                    version="v2"
                )
                resume_response = supervisor_agent.invoke(edit_command, config=config)
                print(f"\n助理: {resume_response['messages'][-1].text}")
            else:
                print("操作已取消，通知 Agent...")
                # 如果拒绝，同样使用 Command 对象，告知底层图引擎取消执行。
                # 这样大模型就能收到“用户拒绝”的反馈，从而给出合理的回复（例如：“好的，我已取消发送”）。
                reject_command = Command(
                    resume={
                        "decisions": [
                            {
                                "type": "reject",
                            }
                        ]
                    },
                    version="v2"
                )
                resume_response = supervisor_agent.invoke(reject_command, config=config)
                print(f"\n助理: {resume_response['messages'][-1].text}")
                
        except Exception as e:
            print(f"\n[错误]: {str(e)}")

if __name__ == "__main__":
    main()