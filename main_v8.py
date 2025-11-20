# main_v2.py
import os
import operator
from typing import TypedDict, Annotated, List, Union
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Document, StorageContext
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
import sqlalchemy
import json
from tools import python_interpreter

# --- 步骤 1: 加载 API 密钥和 LLM ---
load_dotenv()
llm = ChatOpenAI(model="gpt-4o", temperature=0)
web_search_tool = TavilySearch(max_results=3)

# --- 步骤 1.5: M4 - 设置数据库和 LlamaIndex ---
DB_NAME = "astra_db"
# 注意: Homebrew 默认使用你的 macOS 用户名 (maple) 且无密码
# (请将 "maple" 替换为你的 macOS 用户名, 如果不同的话)
DB_USER = "maple" 
DB_PASSWORD = "" # Homebrew 默认
DB_HOST = "localhost"
DB_PORT = "5432"

# 创建 SQLAlchemy 引擎
db_url = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = sqlalchemy.create_engine(db_url)

# 初始化 LlamaIndex 的 PGVectorStore
# "astra_collection" 是我们 RAG 数据的表名
vector_store = PGVectorStore.from_params(
    database=DB_NAME,
    host=DB_HOST,
    port=DB_PORT,
    user=DB_USER,
    password=DB_PASSWORD,
    table_name="astra_collection",
    embed_dim=1536 # OpenAI 嵌入的维度 (ada-002)
)

# 定义嵌入模型
embed_model = OpenAIEmbedding()

# 创建 LlamaIndex 的"存储上下文"和"索引"
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    [], # 启动时, 索引是空的
    storage_context=storage_context,
    embed_model=embed_model
)

# --- 步骤 2: M2 - 定义新的 "Agent 状态" ---
# 这次, "公文包" 变得更复杂了
# 它需要跟踪 "计划" 和 "步骤"
class AgentState(TypedDict):
    task: str                 # 用户的初始任务
    plan: str                 # 规划师生成的计划
    # research_data: List[str]  # 研究员找到的数据
    final_report: str         # 写手生成的最终报告
    
    # "current_step" 是 M2 的关键, 用于路由
    # 它跟踪我们现在在计划的哪一步
    current_step: str
    analysis_results: List[str]


# --- 步骤 3: M2 - 定义我们的 "专家 Agent" 节点 ---

# 3. 规划师 (Planner) 节点 (O4 终极版: 非生成器)
async def planner_node(state: AgentState) -> dict: # (!! 注意: 返回 dict, 不再是 generator)
    print("--- 正在调用 [规划师 (O4)] ---")
    task = state.get("task")
    
    prompt = f"""
    你是一个专业的项目规划师。
    你的任务是为给定的 [任务] 制定一个清晰、简洁、分步的计划。
    [任务]: {task}
    
    将计划分为以下三个步骤之一或全部:
    1. "Research": 如果任务需要从外部世界获取信息 (搜索)。
    2. "Analyze": 如果任务需要数学计算、数据处理或代码执行。
    3. "Write": 编写最终报告。
    
    请只返回计划步骤的列表, 例如:
    ["Research", "Analyze", "Write"]
    """
    
    messages = [HumanMessage(content=prompt)]
    
    print("--- [规划师 O4] 正在流式生成计划... ---")
    
    # (关键) 我们在内部流式处理, 但不 "yield"
    full_plan_str = ""
    async for chunk in llm.astream(messages): # <--- astream_events 会"侦听"到这个
        full_plan_str += chunk.content or ""
    
    print(f"--- [规划师 O4] 计划生成完毕: {full_plan_str} ---")
    
    try:
        if "```json" in full_plan_str:
             full_plan_str = full_plan_str.split("```json")[1].split("```")[0]
        plan_steps = json.loads(full_plan_str.strip())
    except Exception as e:
        print(f"--- [规划师 O4] 计划解析失败! {e} ---")
        plan_steps = ["Research", "Write"] # 降级
    
    # (!! 关键 !!) 我们 "return" 最终结果, 而不是 "yield"
    return {
        "plan": plan_steps,
        "current_step": plan_steps[0] if plan_steps else "END" 
    }

# 2. 研究员 (Researcher) 节点 (与 M1 相同)
def researcher_node(state: AgentState) -> AgentState:
    print("--- 正在调用 [研究员 (M4)] ---")
    task = state.get("task")
    
    # 1. 仍然是搜索
    try:
        research_results = web_search_tool.invoke(task)
    except Exception as e:
        research_results = [f"搜索失败: {e}"]
    
    # 2. M4 核心: "数据摄入" (Ingestion)
    # 我们不再把数据塞进 state, 而是存入 LlamaIndex (pgvector)
    
    print(f"--- [研究员] 找到了 {len(research_results)} 条结果。正在存入记忆... ---")
    
    # 2. [关键修改] 截断过长的内容! (防止爆 Token)
    # 限制每条结果最多 4000 字符 (约 1000-2000 tokens)
    MAX_CHARS = 4000
    # 将原始搜索结果 (字符串) 转换为 LlamaIndex "Document" 对象
    documents = []
    for result in research_results:
        # 确保 result 是字符串
        content = str(result)
        # 如果太长，就截断
        if len(content) > MAX_CHARS:
            content = content[:MAX_CHARS] + "...(已截断)"
            
        documents.append(
            Document(
                text=content, 
                metadata={"task": task}
            )
        )
    
    # 3. 插入数据库
    if documents:
        index.insert_nodes(documents)
    
    return {
        "current_step": "Analyze" # (或者根据你的路由逻辑)
    }

# 4. 分析师 (Analyst) 节点 (M6 新增)
async def analyst_node(state: AgentState) -> dict:
    print("--- 正在调用 [分析师 (M6)] ---")
    task = state.get("task")
    query_engine = index.as_query_engine(similarity_top_k=3)
    rag_context = await query_engine.aquery(task)
    
    # 这里的 Prompt 旨在让 LLM 生成 Python 代码
    prompt = f"""
    你是一个精通 Python 的数据分析师。
    你的任务是编写 Python 代码来解决以下问题。
    
    [任务]: {task}
    
    [参考数据/背景信息]:
    {rag_context}
    
    请遵循以下步骤:
    1. **仔细阅读 [参考数据]**。如果里面包含了具体的数值（例如年份、金额、单位），**必须**使用这些真实数据来构建你的代码中的变量。
    2. 如果 [参考数据] 中没有具体数值，请尝试根据你的知识进行**合理的估算**（例如北欧电力收益通常在 20万欧元/MW 左右，而不是几十欧元），并在代码注释中说明这是估算值。
    3. 编写 Python 代码，通过 Docker 沙箱执行。
    4. **绘图修正**: 
       - 使用 `sns.lineplot` 时，如果只画一条线，**不要**设置 `palette` 参数，这会导致 UserWarning。可以直接指定 `color='blue'`。
       - 务必保存图片到 `/app/output.png`。
    5. 务必使用 print() 输出最终的关键数据。
    """
    
    # 我们绑定工具! 
    # 这是一个高级技巧: Bind Tools
    analyst_llm = llm.bind_tools([python_interpreter])
    
    messages = [HumanMessage(content=prompt)]
    
    print("--- [分析师] 正在编写并执行代码... ---")
    
    # 调用 LLM, 它会决定是否调用工具
    response = await analyst_llm.ainvoke(messages)
    
    analysis_output = "分析师未执行代码。"
    
    # 检查 LLM 是否想要调用工具
    if response.tool_calls:
        for tool_call in response.tool_calls:
            if tool_call["name"] == "python_interpreter":
                code = tool_call["args"]["code"]
                print(f"--- [分析师] 正在沙箱中运行: \n{code} ---")
                
                # 执行工具 (Docker)
                tool_result = python_interpreter.invoke(code)
                
                print(f"--- [分析师] 沙箱返回: \n{tool_result} ---")
                analysis_output = tool_result
    else:
        # 如果 LLM 直接说话了 (没调工具)，我们就用它的话
        analysis_output = response.content

    return {
        "analysis_results": [analysis_output], # 存入状态
        "current_step": "Write" # 分析完通常去写作 (或由 Supervisor 决定)
    }

# 5. 写手 (Writer) 节点 (O4 终极版: 非生成器)
async def writer_node(state: AgentState) -> dict: # (!! 注意: 返回 dict)
    print("--- 正在调用 [写手 (O4)] ---")
    task = state.get("task")
    
    print("--- [写手 O4] 正在从 pgvector 异步检索... ---")
    query_engine = index.as_query_engine(similarity_top_k=3)
    rag_context = await query_engine.aquery(task) # 异步 RAG
    # 获取分析结果
    analysis_results = state.get("analysis_results", [])
    
    print("--- [写手 O4] 记忆检索完毕 ---")
    
    prompt = f"""
    你是一个专业的科技报告写手。
    根据 [任务]、[RAG 上下文] 和 [数据分析结果], 撰写一份专业的总结报告。
    
    [任务]: {task}
    [RAG 上下文]: {rag_context}
    [数据分析结果]: {analysis_results}
    """
    
    messages = [HumanMessage(content=prompt)]
    
    print("--- [写手 O4] 正在流式生成报告... ---")
    
    full_report = ""
    
    # (关键) 我们在内部流式处理, astream_events 会"侦听"到这个
    async for chunk in llm.astream(messages):
        full_report += chunk.content or ""

    print("--- [写手 O4] 报告生成完毕 ---")
    
    # (!! 关键 !!) 我们 "return" 最终结果
    return {
        "final_report": full_report, 
        "current_step": "END"
    }

# --- 步骤 4: M2 - "老板" (Supervisor) 的路由逻辑 ---
# 这就是 "条件边" (Conditional Edge) 的核心!

def supervisor_router(state: AgentState) -> str:
    """
    这是我们的"老板"(Supervisor)。
    它检查 "current_step" 字段, 然后决定下一步去哪个节点。
    """
    print(f"--- [主管] 正在路由: {state.get('current_step')} ---")
    
    current_step = state.get("current_step")
    
    if current_step == "Research":
        return "Researcher"
    elif current_step == "Analyze": # <--- 新增
        return "Analyst"
    elif current_step == "Write":
        return "Writer"
    elif current_step == "END":
        return "END"

# --- 步骤 5: 构建 M2 的 "智能图" ---
workflow = StateGraph(AgentState)

# 1. 添加所有节点
workflow.add_node("Planner", planner_node)
workflow.add_node("Researcher", researcher_node)
workflow.add_node("Analyst", analyst_node)
workflow.add_node("Writer", writer_node)

# 2. 设置"入口"
workflow.set_entry_point("Planner")

# 3. 添加"条件路由"
# 这是 M2 的魔法!
# 我们添加一个"条件边", 从 "Planner" 节点出发
# 它会去调用 `supervisor_router` 函数
# `supervisor_router` 会返回一个字符串 ("Researcher", "Writer" 或 END)
# 然后图会根据这个字符串, 跳转到对应的节点
workflow.add_conditional_edges(
    "Planner",         # "规划师" 运行完毕后...
    supervisor_router, # ...调用"主管"来决定下一步
    {
        "Researcher": "Researcher", # 如果主管说"Research", 就去"研究员"
        "Analyst": "Analyst",
        "Writer": "Writer",         # 如果主管说"Write", 就去"写手"
        "END": END                  # 如果主管说"END", 就结束
    }
)

# 为 Analyst 添加路由 (分析完去找主管，或者像我们这里简化的: 默认去 Write)
# 为了更智能，我们还是用 supervisor_router
workflow.add_conditional_edges(
    "Analyst",
    supervisor_router,
    {
        "Writer": "Writer", # 通常分析完就写
        "END": END
    }
)

# 4. "研究员" 和 "写手" 跑完后, 怎么办?
# 答案是: 他们也应该回去找"主管"汇报!
# (注意: 在这个简单 M2 中, 我们简化了, 让节点自己决定下一步)
# (在一个更高级的 M3 中, Supervisor 应该在每一步都介入)

# 为了简化 M2, 我们让 "Researcher" 跑完后总是去 "Writer"
# (在 M1 中我们用 add_edge, 在 M2 中我们让节点自己更新 current_step)
# 我们需要一个"通用"的路由点

# 让我们重构一下, 这才是正确 M2 结构:
# "主管" 应该在每一步之后都被调用

# --- 步骤 5 (重构 - 正确的 M2 结构) ---
workflow = StateGraph(AgentState)

# 1. 添加节点
workflow.add_node("Planner", planner_node)
workflow.add_node("Researcher", researcher_node)
workflow.add_node("Writer", writer_node)

# 2. 入口
workflow.set_entry_point("Planner")

# 3. 定义"通用"路由
# (这是一个更高级、更像 Manus 的结构)
# 我们让 "Planner", "Researcher", "Writer" 跑完后
# *全部* 回到 "Supervisor" 这里, 由 "Supervisor" 决定下一步

# (为了让这个 M2 保持简单和可运行, 我们先用 V1 的逻辑)
# (V1 逻辑: 规划 -> [路由] -> 研究 -> 写作 -> [路由] -> 结束)

# --- 步骤 5 (M2 - 可运行的最终版) ---
workflow = StateGraph(AgentState)

workflow.add_node("Planner", planner_node)
workflow.add_node("Researcher", researcher_node)
workflow.add_node("Analyst", analyst_node)
workflow.add_node("Writer", writer_node)

workflow.set_entry_point("Planner")

# 1. 规划师 (Planner) 跑完, 去找 主管 (Supervisor)
workflow.add_conditional_edges(
    "Planner",
    supervisor_router,
    {"Researcher": "Researcher", "Analyst": "Analyst", "Writer": "Writer", "END": END}
)

# 2. 研究员 (Researcher) 跑完, 也去找 主管
workflow.add_conditional_edges(
    "Researcher",
    supervisor_router,
    {"Analyst": "Analyst", "Writer": "Writer", "END": END} # 研究完, 只可能去写作或结束
)

# 3. 写手 (Writer) 跑完, 也去找 主管
workflow.add_conditional_edges(
    "Writer",
    supervisor_router,
    {"END": END} # 写完, 只能结束
)

# 4. 编译
memory = MemorySaver()
app = workflow.compile(
    checkpointer=memory,
    interrupt_after=["Planner"]
)

# --- 步骤 6: 运行 M2 Agent ---
if __name__ == "__main__":
    print("🚀 Astra Agent 已启动... (M2 版本: 智能路由)")
    
    session_config = {"configurable": {"thread_id": "session-2"}}
    
    task = "调研 2025 年 AI Agent 市场的最新趋势, 特别是关于 Manus, OpenAI 和 Google 的"
    
    print(f"--- 任务: {task} ---")
    
    # 使用 .stream() 来"直播" Agent 的每一步
    for step in app.stream({"task": task}, config=session_config):
        # stream() 会返回每一步的节点名称和其输出
        node_name = list(step.keys())[0]
        node_output = step[node_name]
        print(f"--- [流] 节点: {node_name} ---")
        # 打印状态的"增量"
        print(f"--- [流] 输出: {node_output} ---")

    # (我们也可以用 .invoke() 一次性运行到底)
    # final_state = app.invoke({"task": task}, config=session_config)
    # print("\n========= [最终报告] =========")
    # print(final_state.get("final_report"))

