import os
from typing import TypedDict, Annotated, List
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain.messages import SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

# --- 步骤 1: 加载 API 密钥 ---
load_dotenv()

# --- 步骤 2: 定义 "Agent 团队" 使用的工具 ---
# 研究员 Agent 将使用 Tavily (Tavily) 搜索工具
web_search_tool = TavilySearch(max_results=3)

# --- 步骤 3: 定义 "Agent 状态" (AgentState) ---
# 这就像是团队共享的 "公文包" 或 "项目板"
class AgentState(TypedDict):
    task: str                 # 用户的初始任务
    research_data: List[str]  # 研究员找到的数据
    final_report: str         # 写手生成的最终报告

# --- 步骤 4: 定义 "Agent 节点" (Node) ---
# 我们将使用 ChatOpenAI (GPT-4o/GPT-4) 作为 Agent 的 "大脑"
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# 1. 研究员 (Researcher) 节点
def researcher_node(state: AgentState) -> AgentState:
    print("--- 正在调用 [研究员] ---")
    task = state.get("task")
    
    # 1. 创建研究提示
    prompt = f"""
    你是一个世界级的 AI 研究员。
    请根据以下任务，为我进行深入的网络搜索：
    任务：{task}
    
    请返回 3 个最相关的搜索结果。
    """
    
    # 2. 调用 LLM (大脑) 和 Tool (工具)
    messages = [SystemMessage(content=prompt)]
    research_results = web_search_tool.invoke(task)
    
    print(f"--- [研究员] 找到了 {len(research_results)} 条结果 ---")
    
    # 3. 更新 "公文包" (AgentState)
    return {
        "research_data": research_results
    }

# 2. 写手 (Writer) 节点
def writer_node(state: AgentState) -> AgentState:
    print("--- 正在调用 [写手] ---")
    research_data = state.get("research_data")
    
    # 1. 创建写作提示
    prompt = f"""
    你是一个专业的科技报告写手。
    请根据以下 [研究数据]，撰写一份简洁、专业的总结报告：

    [研究数据]:
    {research_data}
    
    请直接输出这份报告。
    """
    
    # 2. 调用 LLM (大脑)
    messages = [SystemMessage(content=prompt)]
    report = llm.invoke(messages).content
    
    print("--- [写手] 已完成报告 ---")
    
    # 3. 更新 "公文包" (AgentState)
    return {
        "final_report": report
    }

# --- 步骤 5: 构建 "多智能体图" (LangGraph) ---
workflow = StateGraph(AgentState)

# 1. 添加节点 (我们的"团队成员")
workflow.add_node("Researcher", researcher_node)
workflow.add_node("Writer", writer_node)

# 2. 定义流程 (Edges)
workflow.set_entry_point("Researcher") # 任务从 "研究员" 开始
workflow.add_edge("Researcher", "Writer") # "研究员" 完成后, 交给 "写手"
workflow.add_edge("Writer", END) # "写手" 完成后, 流程结束 (END)

# 3. 编译图 (Compile)
# checkpointer 是可选的, 但它能让我们"记住"每一步的状态
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# --- 步骤 6: 运行我们的 "Astra" Agent ---
if __name__ == "__main__":
    print("🚀 Astra Agent 已启动... (M1 版本)")
    
    # 定义一个会话 ID (这样我们可以"记住"进度)
    session_config = {"configurable": {"thread_id": "session-1"}}
    
    # 我们的复杂任务
    task = "调研 2025 年 AI Agent 市场的最新趋势, 特别是关于 Manus, OpenAI 和 Google 的"
    
    # 启动 Agent 团队!
    # app.stream() 会返回每一步的"实时"输出
    
    # 我们用 .invoke() 一次性运行到底
    print(f"--- 任务: {task} ---")
    
    # 触发图的运行
    final_state = app.invoke(
        {"task": task},
        config=session_config
    )
    
    print("\n--- 流程已结束 ---")
    
    # 打印最终报告
    print("\n========= [最终报告] =========")
    print(final_state.get("final_report"))
    print("==============================")
