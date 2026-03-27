# 1. 第一步：加载环境变量 (必须在导入任何 agent 之前)
import core.config

# 2. 导入主代理
from agent.supervisor_agent import supervisor_agent

def main():
    print("🤖 个人助理 Agent 已启动！(输入 'exit' 退出)")
    print("-" * 50)
    
    while True:
        user_input = input("\n你: ")
        if user_input.lower() in ['exit', 'quit', '退出']:
            print("拜拜！")
            break
            
        if not user_input.strip():
            continue
            
        print("\nAgent 思考中...")
        try:
            # 调用主代理
            response = supervisor_agent.invoke({
                "messages": [{"role": "user", "content": user_input}]
            })
            # 打印最终回复
            print(f"\n助理: {response['messages'][-1].text}")
        except Exception as e:
            print(f"\n[错误]: {str(e)}")

if __name__ == "__main__":
    main()