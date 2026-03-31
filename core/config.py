import os
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
# 这一步必须在所有初始化大模型和 LangSmith 的代码之前执行
load_dotenv()

# 集中管理和暴露环境变量
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

# 你可以在这里做一些环境变量的检查，确保关键 Key 都存在
if not DASHSCOPE_API_KEY:
    raise ValueError("未找到 DASHSCOPE_API_KEY，请检查 .env 文件是否配置正确。")

if not LANGCHAIN_API_KEY:
    print("警告: 未找到 LANGCHAIN_API_KEY，LangSmith 追踪可能无法工作。")